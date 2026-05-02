"""Affinity Propagation clustering with DTW-based similarity matrix.

Affinity Propagation (AP) passes two types of messages between data points
— *responsibilities* and *availabilities* — until a consistent set of
exemplars (cluster representatives) emerges.  Unlike K-Medoids, AP
determines the number of clusters automatically through the *preference*
parameter: higher preference → more exemplars → more clusters.

Similarity matrix construction
-------------------------------
The off-diagonal entries are computed via the same Gaussian kernel used
for Spectral Clustering:

.. math::

    S_{ij} = \\exp\\!\\left(-\\frac{\\mathrm{DTW}(s_i, s_j)^2}{2\\sigma^2}\\right),
    \\quad i \\neq j

The diagonal entries S(i,i) encode the *preference* — how likely each
point is to be selected as an exemplar (higher ↔ more self-confident):

* ``"median"`` (default): S(i,i) = median of all off-diagonal S values,
  yielding a moderate number of clusters.
* ``"min"``: S(i,i) = min of all off-diagonal S values, favouring fewer,
  larger clusters.
* A scalar ``float``: all S(i,i) set to that value.
* A 1-D array of length n_samples: per-sample preferences.

Message-passing update rules
------------------------------
**Responsibility** R(i,k) — how well point k serves as an exemplar for i,
accounting for competing candidates:

.. math::

    R(i,k) \\leftarrow S(i,k)
            - \\max_{k' \\neq k}\\bigl[A(i,k') + S(i,k')\\bigr]

**Availability** A(i,k) — how appropriate it is for i to choose k as its
exemplar, accounting for support from other points:

.. math::

    A(i,k) &\\leftarrow \\min\\!\\Bigl(0,\\;R(k,k)
        + \\textstyle\\sum_{i' \\notin \\{i,k\\}} \\max(0, R(i',k))\\Bigr),
    \\quad i \\neq k \\\\
    A(k,k) &\\leftarrow \\textstyle\\sum_{i' \\neq k} \\max(0, R(i',k))

Both messages are damped at each iteration:
:math:`M \\leftarrow \\lambda M_{\\text{prev}} + (1-\\lambda) M_{\\text{new}}`.

References
----------
Frey, B.J. & Dueck, D. (2007). "Clustering by passing messages between
data points." *Science*, 315(5814), 972–976.
https://doi.org/10.1126/science.1136800
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from dtw_clustering.distances.base import BaseDistance
from dtw_clustering.models.base import BaseClusterer


class AffinityPropagationDTW(BaseClusterer):
    """Affinity Propagation clustering using a DTW-based similarity matrix.

    Parameters
    ----------
    preference : {"median", "min"} or float or ndarray of shape (n_samples,)
        Controls the number of clusters by setting the diagonal of the
        similarity matrix (self-confidence of each point as an exemplar).

        * ``"median"`` (default): S(i,i) = median of off-diagonal values.
        * ``"min"``: S(i,i) = min of off-diagonal values.
        * A scalar: all S(i,i) = that value (must be in (0, 1] with the
          Gaussian kernel).
        * A 1-D array: per-sample preferences.
    metric : str or BaseDistance
        Distance metric.  Defaults to ``"dtw"``.
    sigma : float or None
        Gaussian kernel bandwidth σ.  ``None`` (default) uses the median
        heuristic: σ = median of all non-zero pairwise DTW distances.
    damping : float in [0.5, 1)
        Damping factor λ applied to messages to avoid oscillations.
        Higher values slow convergence but improve numerical stability.
    max_iter : int
        Maximum number of message-passing iterations.
    convits : int
        Number of consecutive iterations with unchanged exemplars required
        to declare convergence.
    n_jobs : int
        Parallel workers for distance matrix computation.
    random_state : int or None
    **metric_kwargs
        Forwarded to the metric constructor (e.g. ``constraint="sakoe_chiba"``,
        ``window=10``).

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster index of each sample (set after :meth:`fit`).
    cluster_centers_indices_ : ndarray of shape (n_clusters,)
        Indices of exemplars in the training data.
    n_clusters_ : int
        Number of clusters found (may differ from ``n_clusters`` placeholder).
    affinity_matrix_ : ndarray of shape (n_samples, n_samples)
        Full similarity matrix **S** (Gaussian kernel + preference diagonal).
    n_iter_ : int
        Number of iterations run until convergence (or ``max_iter``).

    Examples
    --------
    >>> from dtw_clustering.models.classical import AffinityPropagationDTW
    >>> model = AffinityPropagationDTW(preference="median", damping=0.7)
    >>> labels = model.fit_predict(X)
    >>> print(f"Found {model.n_clusters_} clusters")
    """

    def __init__(
        self,
        preference: Union[str, float, np.ndarray] = "median",
        metric: Union[str, BaseDistance] = "dtw",
        sigma: Optional[float] = None,
        damping: float = 0.5,
        max_iter: int = 200,
        convits: int = 15,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        **metric_kwargs,
    ) -> None:
        # n_clusters=1 is a placeholder; the actual value is set after fit().
        super().__init__(
            n_clusters=1,
            metric=metric,
            n_jobs=n_jobs,
            random_state=random_state,
            **metric_kwargs,
        )
        if not (isinstance(preference, (str, np.ndarray)) or np.isscalar(preference)):
            raise TypeError(
                "preference must be 'median', 'min', a scalar float, "
                "or a 1-D ndarray"
            )
        if isinstance(preference, str) and preference not in ("median", "min"):
            raise ValueError("string preference must be 'median' or 'min'")
        if isinstance(damping, float) and not (0.5 <= damping < 1.0):
            raise ValueError("damping must be in [0.5, 1.0)")

        self.preference = preference
        self.sigma = sigma
        self.damping = float(damping)
        self.max_iter = max_iter
        self.convits = convits

        self.cluster_centers_indices_: Optional[np.ndarray] = None
        self.n_clusters_: int = 0
        self.affinity_matrix_: Optional[np.ndarray] = None
        self.n_iter_: int = 0
        self._X_fit: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Similarity matrix
    # ------------------------------------------------------------------

    def _build_similarity(self, D: np.ndarray) -> np.ndarray:
        """Build the full AP similarity matrix S from the DTW distance matrix.

        Off-diagonal: Gaussian kernel applied to DTW distances.
        Diagonal: preference values.

        Parameters
        ----------
        D : ndarray of shape (n, n) — pairwise DTW distances.

        Returns
        -------
        S : ndarray of shape (n, n)
        """
        n = len(D)
        off_diag_mask = ~np.eye(n, dtype=bool)

        # --- σ estimation ---
        if self.sigma is not None:
            sigma = float(self.sigma)
        else:
            positive = D[off_diag_mask & (D > 0)]
            sigma = float(np.median(positive)) if len(positive) > 0 else 1.0

        # --- Off-diagonal: Gaussian kernel ---
        S = np.exp(-(D ** 2) / (2.0 * sigma ** 2))
        np.fill_diagonal(S, 0.0)     # temporary; overwritten below

        # --- Diagonal: preference ---
        off_vals = S[off_diag_mask]  # already computed without diagonal
        if isinstance(self.preference, str):
            pref_val: Union[float, np.ndarray]
            if self.preference == "median":
                pref_val = float(np.median(off_vals))
            else:                     # "min"
                pref_val = float(np.min(off_vals))
            np.fill_diagonal(S, pref_val)
        elif np.isscalar(self.preference):
            np.fill_diagonal(S, float(self.preference))
        else:
            pref_arr = np.asarray(self.preference, dtype=np.float64)
            if pref_arr.shape != (n,):
                raise ValueError(
                    f"preference array must have shape ({n},), "
                    f"got {pref_arr.shape}"
                )
            S[np.diag_indices(n)] = pref_arr

        return S

    # ------------------------------------------------------------------
    # Message-passing (vectorised)
    # ------------------------------------------------------------------

    @staticmethod
    def _update_responsibilities(
        S: np.ndarray,
        A: np.ndarray,
        R: np.ndarray,
        damping: float,
    ) -> np.ndarray:
        """Vectorised responsibility update with damping.

        R_new(i,k) = S(i,k) - max_{k'≠k} [A(i,k') + S(i,k')]
        """
        n = len(S)
        AS = A + S                                    # (n, n)

        # For each row i, find the best and second-best k.
        idx_max = np.argmax(AS, axis=1)               # (n,)
        val_max = AS[np.arange(n), idx_max]           # (n,)

        # Temporarily mask the best k to get second-best.
        AS[np.arange(n), idx_max] = -np.inf
        val_max2 = np.max(AS, axis=1)                 # (n,)
        AS[np.arange(n), idx_max] = val_max           # restore

        # R_new[i, k] = S[i,k] - val_max[i]  for all k
        R_new = S - val_max[:, None]
        # Correct the argmax column: subtract second-best instead of best.
        R_new[np.arange(n), idx_max] = (
            S[np.arange(n), idx_max] - val_max2
        )

        return damping * R + (1.0 - damping) * R_new

    @staticmethod
    def _update_availabilities(
        R: np.ndarray,
        A: np.ndarray,
        damping: float,
    ) -> np.ndarray:
        """Vectorised availability update with damping.

        For i ≠ k:
            A_new(i,k) = min(0, R(k,k) + Σ_{i'∉{i,k}} max(0, R(i',k)))

        For i = k:
            A_new(k,k) = Σ_{i'≠k} max(0, R(i',k))
        """
        n = len(R)

        # Clamp off-diagonal to 0; keep diagonal as R(k,k).
        Rp = np.maximum(R, 0.0)
        np.fill_diagonal(Rp, np.diag(R))             # diagonal: R(k,k), not clamped

        # Column sums include the diagonal term R(k,k).
        col_sum = Rp.sum(axis=0)                      # (n,)

        # A_new[i, k] = col_sum[k] - Rp[i, k]
        #   For i ≠ k: we want min(0, result)
        #   For i = k: no clamping
        A_new = col_sum[None, :] - Rp                 # (n, n)
        diag_vals = A_new.flat[:: n + 1].copy()       # save diagonal before clamping
        np.minimum(A_new, 0.0, out=A_new)            # clamp all to ≤ 0
        A_new.flat[:: n + 1] = diag_vals             # restore diagonal (no clamp)

        return damping * A + (1.0 - damping) * A_new

    # ------------------------------------------------------------------
    # Exemplar extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_exemplars(R: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Return the exemplar index for each sample: argmax_k [R(i,k)+A(i,k)]."""
        return np.argmax(R + A, axis=1)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "AffinityPropagationDTW":
        """Run Affinity Propagation on time series array ``X``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_timesteps[, n_features])

        Returns
        -------
        self
        """
        X = np.asarray(X)
        n = len(X)

        # Step 1 — DTW distance matrix
        D = self._compute_distance_matrix(X)           # (n, n)

        # Step 2 — Similarity matrix (Gaussian kernel + preference diagonal)
        S = self._build_similarity(D)

        # Step 3 — Message passing
        R = np.zeros((n, n), dtype=np.float64)
        A = np.zeros((n, n), dtype=np.float64)

        exemplars_history = np.full((self.convits, n), -1, dtype=int)
        converged = False

        for it in range(self.max_iter):
            R = self._update_responsibilities(S, A, R, self.damping)
            A = self._update_availabilities(R, A, self.damping)

            # Convergence check: stable exemplars for `convits` iterations.
            e = self._extract_exemplars(R, A)
            exemplars_history[it % self.convits] = e

            if it >= self.convits - 1:
                # All rows in the history window identical → converged.
                if (exemplars_history == exemplars_history[0]).all():
                    converged = True
                    self.n_iter_ = it + 1
                    break

        if not converged:
            self.n_iter_ = self.max_iter

        # Step 4 — Final exemplar extraction
        e = self._extract_exemplars(R, A)

        # Exemplars are points that choose themselves: e[k] == k
        unique_exemplars = np.unique(e[e == np.arange(n)])

        if len(unique_exemplars) == 0:
            # Degenerate: no self-exemplars found; fall back to the point
            # with the highest cumulative criterion value.
            best = int(np.argmax(np.diag(R) + np.diag(A)))
            unique_exemplars = np.array([best], dtype=int)

        # Assign each sample to the nearest exemplar.
        D_ex = D[:, unique_exemplars]                  # (n, n_ex)
        cluster_assignments = np.argmin(D_ex, axis=1)  # (n,) — index into unique_exemplars
        labels = cluster_assignments.astype(int)

        self.labels_ = labels
        self.cluster_centers_indices_ = unique_exemplars
        self.n_clusters_ = len(unique_exemplars)
        self.n_clusters = self.n_clusters_             # keep consistent with base class
        self.affinity_matrix_ = S
        self._X_fit = X
        self._D_fit = D
        return self

    # ------------------------------------------------------------------
    # Cluster centres and predict
    # ------------------------------------------------------------------

    def _cluster_centers(self) -> np.ndarray:
        """Return the exemplar time series, shape (n_clusters, T[, d])."""
        self._check_is_fitted()
        return self._X_fit[self.cluster_centers_indices_]   # type: ignore[index]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new samples to the nearest exemplar.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_timesteps[, n_features])

        Returns
        -------
        labels : ndarray of shape (n_samples,), dtype int
        """
        self._check_is_fitted()
        centers = self._cluster_centers()
        D = self._compute_distance_matrix(X, Y=centers)     # (m, n_clusters)
        return np.argmin(D, axis=1).astype(int)
