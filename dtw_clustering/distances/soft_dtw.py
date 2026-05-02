"""Soft-DTW: a differentiable loss function for time series.

Implements the forward pass (value) and backward pass (gradient w.r.t. the
local cost matrix) of Soft-DTW as described in:

    Cuturi & Blondel (2017). "Soft-DTW: a Differentiable Loss Function for
    Time-Series." ICML 2017. https://arxiv.org/abs/1703.01541

Two backends are provided:

* **NumPy** — :func:`soft_dtw_value` and :func:`soft_dtw_grad`.  Suitable for
  classical clustering, barycenter computation, and anywhere PyTorch is not
  available.
* **PyTorch** — :class:`SoftDTW` (``torch.nn.Module``).  Used as a
  reconstruction loss in deep clustering models (DEC, IDEC, …).  Imported
  lazily so that PyTorch is an *optional* dependency.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from joblib import Parallel, delayed

from .base import BaseDistance


# ---------------------------------------------------------------------------
# NumPy backend
# ---------------------------------------------------------------------------

def _local_cost(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Squared-Euclidean local cost matrix (supports multivariate series)."""
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    diff = x[:, None, :] - y[None, :, :]
    return np.sum(diff ** 2, axis=-1)


def _softmin_weights(a: float, b: float, c: float, gamma: float) -> np.ndarray:
    """Numerically stable softmin probability vector for three scalars.

    Returns ``p`` such that ``softmin_γ(a,b,c) = p·[a,b,c]`` and the
    gradient of softmin w.r.t. each argument equals ``p[i]``.
    """
    v = np.array([-a, -b, -c]) / gamma
    v -= v.max()          # subtract max for numerical stability
    p = np.exp(v)
    p /= p.sum()
    return p              # shape (3,)


def _forward(C: np.ndarray, gamma: float) -> np.ndarray:
    """Soft-DTW forward pass; returns the padded accumulated cost matrix R.

    Uses the log-sum-exp form ``softmin_γ(a,b,c) = rmin - γ log Σ exp(-(r-rmin)/γ)``
    where ``rmin = min(a,b,c)``. This avoids the ``0·∞`` indeterminate form that
    arises when boundary cells (initialised to ``+∞``) are weighted by zero
    softmin probabilities.
    """
    n, m = C.shape
    R = np.full((n + 1, m + 1), np.inf)
    R[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            r0 = R[i - 1, j]
            r1 = R[i, j - 1]
            r2 = R[i - 1, j - 1]
            rmin = min(r0, r1, r2)
            softmin = rmin - gamma * np.log(
                np.exp(-(r0 - rmin) / gamma)
                + np.exp(-(r1 - rmin) / gamma)
                + np.exp(-(r2 - rmin) / gamma)
            )
            R[i, j] = C[i - 1, j - 1] + softmin
    return R


def soft_dtw_value(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
) -> float:
    """Compute the Soft-DTW value between two time series.

    Parameters
    ----------
    x, y : array-like of shape (n_timesteps,) or (n_timesteps, n_features)
    gamma : float
        Smoothing parameter.  As γ → 0, Soft-DTW converges to standard DTW.

    Returns
    -------
    float
    """
    if gamma <= 0:
        raise ValueError("gamma must be strictly positive")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, m = x.shape[0], y.shape[0]
    R = _forward(_local_cost(x, y), gamma)
    return float(R[n, m])


def soft_dtw_grad(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Compute Soft-DTW value and gradient w.r.t. the local cost matrix.

    The returned gradient ``E`` is the **soft alignment matrix**: ``E[i, j]``
    is the expected number of times cell ``(i, j)`` is visited on the optimal
    path under the softmin distribution.  It satisfies:

        ∂ soft_dtw(x, y) / ∂ C[i, j]  =  E[i, j]

    Parameters
    ----------
    x, y : array-like of shape (n_timesteps,) or (n_timesteps, n_features)
    gamma : float

    Returns
    -------
    value : float
        Soft-DTW value.
    E : ndarray of shape (n_timesteps_x, n_timesteps_y)
        Soft alignment matrix (gradient of Soft-DTW w.r.t. C).
    """
    if gamma <= 0:
        raise ValueError("gamma must be strictly positive")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, m = x.shape[0], y.shape[0]
    C = _local_cost(x, y)

    # Forward pass
    R = _forward(C, gamma)
    value = float(R[n, m])

    # Backward pass
    # E[i, j] = ∂R[n,m] / ∂R[i,j]  (which equals ∂R[n,m] / ∂C[i-1,j-1])
    # Initialise with 1-padding so boundary cells have zero gradient.
    E = np.zeros((n + 1, m + 1))
    E[n, m] = 1.0

    for i in range(n, 0, -1):
        for j in range(m, 0, -1):
            eij = E[i, j]
            if eij == 0.0:
                continue
            # Softmin weights that produced R[i,j] from its three predecessors:
            #   p[0] ← R[i-1, j]   (move along x / "up")
            #   p[1] ← R[i, j-1]   (move along y / "left")
            #   p[2] ← R[i-1, j-1] (diagonal)
            p = _softmin_weights(R[i - 1, j], R[i, j - 1], R[i - 1, j - 1], gamma)
            E[i - 1, j]     += eij * p[0]
            E[i,     j - 1] += eij * p[1]
            E[i - 1, j - 1] += eij * p[2]

    # Strip the padding row/col; E[1..n, 1..m] → E[0..n-1, 0..m-1] in C-space
    return value, E[1:, 1:]


def soft_dtw_distance_matrix(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    gamma: float = 0.05,
    n_jobs: int = 1,
) -> np.ndarray:
    """Compute a pairwise Soft-DTW distance matrix.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_timesteps[, n_features])
    Y : ndarray of shape (m_samples, n_timesteps[, n_features]), optional
        If ``None``, computes the symmetric ``X``-vs-``X`` matrix and only
        evaluates the upper triangle (O(n²/2) distance calls).
    gamma : float
        Soft-DTW smoothing parameter (default 0.05, close to hard DTW).
    n_jobs : int

    Returns
    -------
    D : ndarray of shape (n_samples, n_samples) or (n_samples, m_samples)
    """
    symmetric = Y is None
    if symmetric:
        Y = X
    n, m = len(X), len(Y)

    def _d(i: int, j: int) -> float:
        return soft_dtw_value(X[i], Y[j], gamma=gamma)

    if symmetric:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        vals = Parallel(n_jobs=n_jobs)(delayed(_d)(i, j) for i, j in pairs)
        D = np.zeros((n, n), dtype=np.float64)
        for (i, j), v in zip(pairs, vals):
            D[i, j] = v
            D[j, i] = v
    else:
        pairs = [(i, j) for i in range(n) for j in range(m)]
        vals = Parallel(n_jobs=n_jobs)(delayed(_d)(i, j) for i, j in pairs)
        D = np.array(vals, dtype=np.float64).reshape(n, m)

    return D


# ---------------------------------------------------------------------------
# BaseDistance-compatible wrapper (NumPy)
# ---------------------------------------------------------------------------

class SoftDTWDistance(BaseDistance):
    """Soft-DTW as a :class:`~dtw_clustering.distances.base.BaseDistance`.

    Parameters
    ----------
    gamma : float
        Smoothing parameter (default 1.0).

    Examples
    --------
    >>> sdtw = SoftDTWDistance(gamma=0.1)
    >>> d = sdtw(x, y)
    >>> D = sdtw.pairwise(X, n_jobs=-1)
    """

    def __init__(self, gamma: float = 1.0) -> None:
        if gamma <= 0:
            raise ValueError("gamma must be strictly positive")
        self.gamma = gamma

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        return soft_dtw_value(x, y, gamma=self.gamma)


# ---------------------------------------------------------------------------
# PyTorch backend (optional dependency)
# ---------------------------------------------------------------------------

def _build_torch_module() -> type:
    """Build and return the PyTorch SoftDTW nn.Module class (lazily imported)."""
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for the SoftDTW nn.Module. "
            "Install it with: pip install torch"
        ) from exc

    class _SoftDTW(nn.Module):
        """Soft-DTW loss as a PyTorch ``nn.Module`` for deep clustering.

        Parameters
        ----------
        gamma : float
            Smoothing parameter.
        normalize : bool
            If ``True``, subtract ``[sdtw(x,x) + sdtw(y,y)] / 2`` so that
            the value is always non-negative (Blondel et al., 2021).

        Input shapes
        ------------
        x, y : Tensor of shape ``(batch, time, features)``

        References
        ----------
        Cuturi & Blondel (2017) https://arxiv.org/abs/1703.01541
        Blondel et al. (2021)   https://arxiv.org/abs/2008.02550
        """

        def __init__(self, gamma: float = 1.0, normalize: bool = False) -> None:
            super().__init__()
            if gamma <= 0:
                raise ValueError("gamma must be strictly positive")
            self.gamma = gamma
            self.normalize = normalize

        @staticmethod
        def _sq_dist(x: "torch.Tensor", y: "torch.Tensor") -> "torch.Tensor":
            diff = x.unsqueeze(2) - y.unsqueeze(1)   # (B, n, m, d)
            return (diff ** 2).sum(dim=-1)             # (B, n, m)

        def _sdtw(
            self,
            x: "torch.Tensor",
            y: "torch.Tensor",
        ) -> "torch.Tensor":
            B, n, _ = x.shape
            m = y.shape[1]
            C = self._sq_dist(x, y)                   # (B, n, m)

            INF = 1e38
            R = torch.full(
                (B, n + 1, m + 1), INF, dtype=x.dtype, device=x.device
            )
            R[:, 0, 0] = 0.0

            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    # Stable softmin via log-sum-exp of negated predecessors.
                    stacked = torch.stack(
                        [-R[:, i - 1, j], -R[:, i, j - 1], -R[:, i - 1, j - 1]],
                        dim=1,
                    ) / self.gamma
                    stacked = stacked - stacked.max(dim=1, keepdim=True).values
                    sm = -self.gamma * torch.logsumexp(stacked, dim=1)
                    R[:, i, j] = C[:, i - 1, j - 1] + sm

            return R[:, n, m]

        def forward(
            self,
            x: "torch.Tensor",
            y: "torch.Tensor",
        ) -> "torch.Tensor":
            """Return per-sample (normalised) Soft-DTW loss, shape ``(batch,)``."""
            val = self._sdtw(x, y)
            if self.normalize:
                val = val - (self._sdtw(x, x) + self._sdtw(y, y)) / 2.0
            return val

    return _SoftDTW


class SoftDTW:
    """Lazy proxy that returns a PyTorch Soft-DTW ``nn.Module`` on instantiation.

    PyTorch is imported only when this class is instantiated, so it remains
    an optional dependency for users who only need the NumPy backend.

    Parameters
    ----------
    gamma : float
    normalize : bool

    Examples
    --------
    >>> loss_fn = SoftDTW(gamma=0.1, normalize=True)
    >>> loss = loss_fn(x_batch, x_recon).mean()
    >>> loss.backward()
    """

    def __new__(cls, gamma: float = 1.0, normalize: bool = False):  # type: ignore[override]
        return _build_torch_module()(gamma=gamma, normalize=normalize)
