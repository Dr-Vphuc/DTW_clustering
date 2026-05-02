"""DTW and Soft-DTW barycenter algorithms.

Two barycenter methods are implemented:

* :func:`dtw_barycenter_averaging` — DTW Barycenter Averaging (DBA), a
  coordinate-descent algorithm that alternates between (1) aligning each
  series to the current estimate and (2) updating the estimate as the
  per-position mean of aligned samples.

* :func:`soft_dtw_barycenter` — gradient-based barycenter that minimises the
  sum of Soft-DTW distances, solved with ``scipy.optimize.minimize``.

References
----------
Petitjean et al. (2011). "A global averaging method for dynamic time warping."
    Pattern Recognition. https://doi.org/10.1016/j.patcog.2010.09.013

Cuturi & Blondel (2017). "Soft-DTW: a Differentiable Loss Function for
    Time-Series." ICML 2017. https://arxiv.org/abs/1703.01541
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize

from .constraints import BaseConstraint
from .dtw import dtw_path
from .soft_dtw import _local_cost, soft_dtw_grad


# ---------------------------------------------------------------------------
# DBA — DTW Barycenter Averaging
# ---------------------------------------------------------------------------

def dtw_barycenter_averaging(
    X: np.ndarray,
    barycenter_size: Optional[int] = None,
    init: Optional[np.ndarray] = None,
    constraint: Optional[BaseConstraint] = None,
    max_iter: int = 30,
    tol: float = 1e-5,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute the DTW barycenter of a set of time series via DBA.

    DBA alternates between two steps until convergence:

    1. **Assignment** — for each series ``x_i``, compute the DTW alignment
       path between the current barycenter estimate ``z`` and ``x_i``.
    2. **Update** — for each time step ``k`` of ``z``, set ``z[k]`` to the
       (weighted) mean of all ``x_i`` values that are matched to step ``k``.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_timesteps[, n_features])
        Input time series.  All series must have the same length when
        ``barycenter_size`` is ``None``.
    barycenter_size : int, optional
        Length of the output barycenter.  Defaults to ``X.shape[1]``.
    init : ndarray, optional
        Initial barycenter estimate.  Defaults to the arithmetic mean of ``X``.
    constraint : BaseConstraint, optional
        Warping constraint applied during alignment.
    max_iter : int
        Maximum number of DBA iterations.
    tol : float
        Convergence tolerance (L∞ norm of the update step).
    weights : ndarray of shape (n_samples,), optional
        Per-sample non-negative weights.  Defaults to uniform.

    Returns
    -------
    z : ndarray of shape (barycenter_size[, n_features])
        DTW barycenter.
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples = len(X)
    T = X.shape[1]
    multivariate = X.ndim == 3
    d = X.shape[2] if multivariate else 1

    if barycenter_size is None:
        barycenter_size = T

    if weights is None:
        weights = np.ones(n_samples, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (n_samples,) or np.any(weights < 0):
        raise ValueError("weights must be a non-negative 1-D array of length n_samples")
    weights = weights / weights.sum()

    # --- Initialise ---
    if init is None:
        if barycenter_size == T:
            z = np.mean(X, axis=0)
        else:
            # Resample the mean to the target length via linear interpolation.
            mean_x = np.mean(X, axis=0)          # (T, d) or (T,)
            t_old = np.linspace(0, 1, T)
            t_new = np.linspace(0, 1, barycenter_size)
            if multivariate:
                z = np.stack(
                    [np.interp(t_new, t_old, mean_x[:, k]) for k in range(d)],
                    axis=-1,
                )
            else:
                z = np.interp(t_new, t_old, mean_x)
    else:
        z = np.asarray(init, dtype=np.float64).copy()

    # --- DBA iterations ---
    for _ in range(max_iter):
        # Accumulate weighted sums for each barycenter time step.
        if multivariate:
            numerator = np.zeros((barycenter_size, d), dtype=np.float64)
        else:
            numerator = np.zeros(barycenter_size, dtype=np.float64)
        denominator = np.zeros(barycenter_size, dtype=np.float64)

        for i in range(n_samples):
            path, _ = dtw_path(z, X[i], constraint=constraint)
            w_i = weights[i]
            for k, l in path:
                numerator[k] += w_i * X[i][l]
                denominator[k] += w_i

        # Avoid division by zero for unreached barycenter steps.
        mask = denominator > 0
        z_new = np.where(
            mask[:, None] if multivariate else mask,
            numerator / np.maximum(denominator[:, None] if multivariate else denominator, 1e-12),
            z,
        )

        if np.max(np.abs(z_new - z)) < tol:
            z = z_new
            break
        z = z_new

    return z


# ---------------------------------------------------------------------------
# Soft-DTW barycenter
# ---------------------------------------------------------------------------

def soft_dtw_barycenter(
    X: np.ndarray,
    gamma: float = 1.0,
    weights: Optional[np.ndarray] = None,
    init: Optional[np.ndarray] = None,
    max_iter: int = 50,
    tol: float = 1e-5,
    method: str = "L-BFGS-B",
) -> np.ndarray:
    """Compute the Soft-DTW barycenter of a set of time series.

    Minimises the weighted sum of Soft-DTW distances:

        z* = argmin_z  Σ_i w_i · soft_dtw_γ(z, x_i)

    The gradient w.r.t. ``z`` is derived from the soft alignment matrices
    returned by :func:`~dtw_clustering.distances.soft_dtw.soft_dtw_grad`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_timesteps[, n_features])
        Input time series (must have equal length).
    gamma : float
        Soft-DTW smoothing parameter.
    weights : ndarray of shape (n_samples,), optional
        Per-sample non-negative weights.  Defaults to uniform.
    init : ndarray, optional
        Initial barycenter.  Defaults to the arithmetic mean of ``X``.
    max_iter : int
        Maximum number of L-BFGS-B iterations.
    tol : float
        Gradient norm tolerance for convergence.
    method : str
        Optimisation method forwarded to :func:`scipy.optimize.minimize`.

    Returns
    -------
    z : ndarray of shape (n_timesteps[, n_features])
        Soft-DTW barycenter.
    """
    if gamma <= 0:
        raise ValueError("gamma must be strictly positive")

    X = np.asarray(X, dtype=np.float64)
    n_samples = len(X)
    shape = X.shape[1:]           # (T,) or (T, d)
    T = shape[0]

    if weights is None:
        weights = np.ones(n_samples, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()

    if init is None:
        z0 = np.mean(X, axis=0)
    else:
        z0 = np.asarray(init, dtype=np.float64).copy()

    def _objective(z_flat: np.ndarray) -> tuple[float, np.ndarray]:
        z = z_flat.reshape(shape)
        total_val = 0.0
        grad_z = np.zeros_like(z)

        for i in range(n_samples):
            val, E = soft_dtw_grad(z, X[i], gamma=gamma)
            total_val += weights[i] * val

            # ∇_z soft_dtw(z, x_i)[k] = 2 * Σ_l E[k, l] * (z[k] - x_i[l])
            # Vectorised: for univariate z shape (T,), x_i shape (T,):
            #   ∂/∂z[k] C[k,l] = 2*(z[k] - x_i[l])
            #   ∇_z[k] = 2 * Σ_l E[k,l] * (z[k] - x_i[l])
            xi = X[i]
            if z.ndim == 1:
                # z: (T,), xi: (T,)
                # C[k,l] = (z[k] - xi[l])^2
                # ∇_z[k] = 2 * Σ_l E[k,l] * (z[k] - xi[l])
                #         = 2 * (E.sum(axis=1) * z - E @ xi)
                g = 2.0 * (E.sum(axis=1) * z - E @ xi)
            else:
                # z: (T, d), xi: (T, d)
                # C[k,l] = ||z[k] - xi[l]||^2
                # ∇_z[k, :] = 2 * Σ_l E[k,l] * (z[k,:] - xi[l,:])
                #            = 2 * (E.sum(1)[:,None] * z - E @ xi)
                g = 2.0 * (E.sum(axis=1)[:, None] * z - E @ xi)

            grad_z += weights[i] * g

        return total_val, grad_z.ravel()

    result = minimize(
        _objective,
        z0.ravel(),
        method=method,
        jac=True,
        options={"maxiter": max_iter, "gtol": tol},
    )

    return result.x.reshape(shape)
