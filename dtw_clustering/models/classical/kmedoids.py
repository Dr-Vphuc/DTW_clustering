"""K-Medoids clustering with DTW distance (PAM algorithm).

K-Medoids selects *actual samples* as cluster representatives (medoids),
making it more robust to outliers than K-Means.  The entire algorithm
operates on the precomputed pairwise DTW distance matrix, so it works
with any metric that does not admit a meaningful Euclidean centroid.

Algorithm
---------
Two initialisation strategies are available:

* ``"random"`` — draw *k* distinct samples uniformly at random.
* ``"kmedoids++"`` — a greedy distance-based initialisation analogous to
  k-means++, which tends to produce better initial medoids and faster
  convergence.

The main loop follows the **PAM-SWAP** algorithm (Kaufman & Rousseeuw,
1990), fully vectorised: for every (medoid, non-medoid) pair the change
in total within-cluster cost is evaluated simultaneously via NumPy
broadcasting; the single best improving swap is applied per iteration.

References
----------
Kaufman, L. & Rousseeuw, P.J. (1990). *Finding Groups in Data: An
Introduction to Cluster Analysis.* Wiley.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from dtw_clustering.distances.base import BaseDistance
from dtw_clustering.models.base import BaseClusterer


class KMedoidsDTW(BaseClusterer):
    """K-Medoids clustering using precomputed DTW distances (PAM).

    Parameters
    ----------
    n_clusters : int
        Number of clusters *k*.
    metric : str or BaseDistance
        Distance metric.  Defaults to ``"dtw"``.
    init : {"random", "kmedoids++"}
        Medoid initialisation strategy.
    max_iter : int
        Maximum number of PAM-SWAP iterations.
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
    medoid_indices_ : ndarray of shape (n_clusters,)
        Indices into the training data of the chosen medoids.
    inertia_ : float
        Sum of DTW distances from each sample to its assigned medoid.

    Examples
    --------
    >>> from dtw_clustering.models.classical import KMedoidsDTW
    >>> model = KMedoidsDTW(n_clusters=3, constraint="sakoe_chiba", window=10)
    >>> labels = model.fit_predict(X)
    """

    def __init__(
        self,
        n_clusters: int = 2,
        metric: Union[str, BaseDistance] = "dtw",
        init: str = "kmedoids++",
        max_iter: int = 300,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        **metric_kwargs,
    ) -> None:
        super().__init__(
            n_clusters=n_clusters,
            metric=metric,
            n_jobs=n_jobs,
            random_state=random_state,
            **metric_kwargs,
        )
        if init not in ("random", "kmedoids++"):
            raise ValueError("init must be 'random' or 'kmedoids++'")
        self.init = init
        self.max_iter = max_iter

        self.medoid_indices_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self._X_fit: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_random(self, n: int, rng: np.random.RandomState) -> np.ndarray:
        return rng.choice(n, size=self.n_clusters, replace=False)

    def _init_pp(self, D: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """K-Medoids++ — proportional-to-distance² greedy initialisation."""
        n = D.shape[0]
        first = rng.randint(n)
        medoids = [first]
        for _ in range(1, self.n_clusters):
            d_min = D[:, medoids].min(axis=1)        # (n,)
            probs = d_min ** 2
            total = probs.sum()
            probs = probs / total if total > 0 else np.ones(n) / n
            medoids.append(int(rng.choice(n, p=probs)))
        return np.array(medoids, dtype=int)

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    @staticmethod
    def _assign(D: np.ndarray, medoids: np.ndarray) -> tuple[np.ndarray, float]:
        """Assign each sample to its nearest medoid; return (labels, cost)."""
        D_med = D[:, medoids]                            # (n, k)
        labels = np.argmin(D_med, axis=1)               # position within medoids
        cost = float(D_med[np.arange(len(D)), labels].sum())
        return labels, cost

    # ------------------------------------------------------------------
    # PAM-SWAP (vectorised)
    # ------------------------------------------------------------------

    def _swap_step(
        self, D: np.ndarray, medoids: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        """Evaluate all (medoid, non-medoid) swaps and apply the best one.

        For each medoid position *pos* and each non-medoid candidate *o*,
        the change in total within-cluster cost ΔTC(pos, o) is:

        * For sample *i* assigned to medoid *pos* (in-cluster):
            Δᵢ = min(d(i,o), d₂(i)) − d(i, medoid[pos])
          where d₂(i) is the distance to the *second*-nearest medoid.

        * For sample *i* assigned to a different medoid (out-cluster):
            Δᵢ = max(d(i,o) − d(i, nearest_medoid), 0)

        Parameters
        ----------
        D : ndarray of shape (n, n) — full pairwise distance matrix.
        medoids : ndarray of shape (k,) — current medoid indices.

        Returns
        -------
        medoids : updated (or unchanged) medoid array.
        swapped : bool — whether a swap was performed.
        """
        n = D.shape[0]
        k = len(medoids)
        non_medoids = np.setdiff1d(np.arange(n), medoids)

        if len(non_medoids) == 0:
            return medoids, False

        D_mk = D[:, medoids]                             # (n, k)
        # Sort columns to find nearest / second-nearest medoid per sample.
        order = np.argsort(D_mk, axis=1)                 # (n, k)
        labels = order[:, 0]                             # nearest-medoid position
        d_nearest = D_mk[np.arange(n), labels]          # (n,)
        d_second = (
            D_mk[np.arange(n), order[:, 1]] if k > 1 else np.full(n, np.inf)
        )

        D_no = D[:, non_medoids]                         # (n, n_nm)

        best_delta = 0.0
        best_pos: int = -1
        best_o_idx: int = -1

        for pos in range(k):
            in_mask = labels == pos                      # (n,)
            out_mask = ~in_mask

            # In-cluster contribution per (sample, candidate) pair
            if in_mask.any():
                delta_in = (
                    np.minimum(D_no[in_mask], d_second[in_mask, None])
                    - d_nearest[in_mask, None]
                )                                        # (n_in, n_nm)
            else:
                delta_in = np.zeros((0, len(non_medoids)))

            # Out-cluster contribution
            if out_mask.any():
                delta_out = np.maximum(
                    D_no[out_mask] - d_nearest[out_mask, None], 0.0
                )                                        # (n_out, n_nm)
            else:
                delta_out = np.zeros((0, len(non_medoids)))

            total_delta = delta_in.sum(axis=0) + delta_out.sum(axis=0)  # (n_nm,)
            min_idx = int(np.argmin(total_delta))

            if total_delta[min_idx] < best_delta:
                best_delta = total_delta[min_idx]
                best_pos = pos
                best_o_idx = min_idx

        if best_pos >= 0:
            medoids = medoids.copy()
            medoids[best_pos] = non_medoids[best_o_idx]
            return medoids, True

        return medoids, False

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "KMedoidsDTW":
        """Compute K-Medoids clustering.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_timesteps[, n_features])

        Returns
        -------
        self
        """
        if isinstance(X, list):
            X_arr = np.empty(len(X), dtype=object)
            for i, track in enumerate(X):
                X_arr[i] = np.asarray(track)
            X = X_arr
        else:
            X = np.asarray(X)
        n = len(X)
        if n < self.n_clusters:
            raise ValueError(
                f"n_samples={n} is smaller than n_clusters={self.n_clusters}"
            )

        rng = self._random_state()
        D = self._compute_distance_matrix(X)             # symmetric (n, n)

        medoids = (
            self._init_pp(D, rng)
            if self.init == "kmedoids++"
            else self._init_random(n, rng)
        )

        for _ in range(self.max_iter):
            medoids, improved = self._swap_step(D, medoids)
            if not improved:
                break

        labels, cost = self._assign(D, medoids)

        self.labels_ = labels.astype(int)
        self.medoid_indices_ = medoids
        self.inertia_ = cost
        self._X_fit = X
        return self

    def _cluster_centers(self) -> np.ndarray:
        """Return medoid time series, shape (n_clusters, T[, d])."""
        self._check_is_fitted()
        return self._X_fit[self.medoid_indices_]           # type: ignore[index]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new samples to the nearest medoid.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_timesteps[, n_features])

        Returns
        -------
        labels : ndarray of shape (n_samples,), dtype int
        """
        self._check_is_fitted()
        centers = self._cluster_centers()
        D = self._compute_distance_matrix(X, Y=centers)
        return np.argmin(D, axis=1).astype(int)
