"""DTW K-Means: K-Means with DBA centroid updates.

Standard K-Means is not directly applicable to DTW because the arithmetic
mean of time series is not a meaningful DTW centroid.  This module replaces
the Euclidean centroid update with **DTW Barycenter Averaging (DBA)**
(Petitjean et al., 2011), which produces a centroid that minimises the
sum of DTW distances to the cluster members.

Algorithm
---------
1. **Initialise** centroids (``"random"`` or ``"dtw_k-means++"``)
2. **Assign** each series to the nearest centroid (by DTW distance).
3. **Update** each centroid using DBA over the assigned members.
4. Repeat steps 2–3 until labels stabilise or ``max_iter`` is reached.

Unlike K-Medoids, centroids are *synthetic* time series (not actual
samples), which allows finer-grained cluster summaries.  The distance
matrix is recomputed at each iteration, so this method is more expensive
than K-Medoids for large datasets.

References
----------
Petitjean, F., Ketterlin, A. & Gançarski, P. (2011). "A global averaging
    method for dynamic time warping, with applications to clustering."
    *Pattern Recognition*, 44(3), 678–693.
    https://doi.org/10.1016/j.patcog.2010.09.013
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from dtw_clustering.distances.base import BaseDistance
from dtw_clustering.distances.constraints import BaseConstraint
from dtw_clustering.distances.barycenter import dtw_barycenter_averaging
from dtw_clustering.models.base import BaseClusterer


class KMeansDTW(BaseClusterer):
    """DTW K-Means with DBA centroid updates.

    Parameters
    ----------
    n_clusters : int
        Number of clusters *k*.
    metric : str or BaseDistance
        Distance metric used for assignment.  Defaults to ``"dtw"``.
    init : {"random", "dtw_kmeans++"}
        Centroid initialisation strategy.

        * ``"random"`` — draw *k* samples uniformly at random.
        * ``"dtw_kmeans++"`` — distance-proportional greedy seeding
          (DTW analogue of k-means++).
    max_iter : int
        Maximum number of assignment–update iterations.
    tol : float
        Convergence tolerance: stop when the fraction of re-assigned
        samples falls below this value.
    dba_max_iter : int
        Maximum DBA iterations per centroid update.
    n_jobs : int
        Parallel workers for distance matrix computation.
    random_state : int or None
    **metric_kwargs
        Forwarded to the metric constructor.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster index of each sample.
    cluster_centers_ : ndarray of shape (n_clusters, n_timesteps[, n_features])
        DBA centroids.
    inertia_ : float
        Sum of DTW distances from each sample to its assigned centroid.
    n_iter_ : int
        Number of iterations run.

    Examples
    --------
    >>> from dtw_clustering.models.classical import KMeansDTW
    >>> model = KMeansDTW(n_clusters=4, constraint="sakoe_chiba", window=10)
    >>> labels = model.fit_predict(X)
    """

    def __init__(
        self,
        n_clusters: int = 2,
        metric: Union[str, BaseDistance] = "dtw",
        init: str = "dtw_kmeans++",
        max_iter: int = 100,
        tol: float = 0.0,
        dba_max_iter: int = 30,
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
        if init not in ("random", "dtw_kmeans++"):
            raise ValueError("init must be 'random' or 'dtw_kmeans++'")
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.dba_max_iter = dba_max_iter

        # Extract constraint for DBA (DBA also uses DTW internally).
        # If the metric is a DTWDistance, reuse its constraint; otherwise
        # use no constraint for DBA.
        from dtw_clustering.distances.dtw import DTWDistance
        if isinstance(self.metric, DTWDistance):
            self._dba_constraint: Optional[BaseConstraint] = self.metric.constraint
        else:
            self._dba_constraint = None

        self.cluster_centers_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.distance_matrix_: Optional[np.ndarray] = None
        self.n_iter_: int = 0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_random(
        self, X: np.ndarray, rng: np.random.RandomState
    ) -> np.ndarray:
        idx = rng.choice(len(X), size=self.n_clusters, replace=False)
        return X[idx].copy()

    def _init_pp(
        self, X: np.ndarray, D: np.ndarray, rng: np.random.RandomState
    ) -> np.ndarray:
        """DTW K-Means++ — distance-proportional greedy centroid seeding.

        Uses the precomputed pairwise matrix so that no extra DTW calls are
        made during initialisation.
        """
        n = len(X)
        first = rng.randint(n)
        chosen_idx = [first]

        for _ in range(1, self.n_clusters):
            d_min = D[:, chosen_idx].min(axis=1)     # (n,)
            probs = d_min ** 2
            total = probs.sum()
            probs = probs / total if total > 0 else np.ones(n) / n
            chosen_idx.append(int(rng.choice(n, p=probs)))

        return X[np.array(chosen_idx)].copy()

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    def _assign(
        self, X: np.ndarray, centers: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Assign each sample to the nearest centroid; return (labels, inertia)."""
        D = self._compute_distance_matrix(X, Y=centers)  # (n, k)
        labels = np.argmin(D, axis=1)
        inertia = float(D[np.arange(len(X)), labels].sum())
        return labels.astype(int), inertia

    # ------------------------------------------------------------------
    # Centroid update (DBA)
    # ------------------------------------------------------------------

    def _update_centers(
        self, X: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Recompute each centroid via DBA over its assigned members."""
        centers = []
        for k in range(self.n_clusters):
            members = X[labels == k]
            if len(members) == 0:
                # Empty cluster: reinitialise to a random sample.
                rng = self._random_state()
                centers.append(X[rng.randint(len(X))].copy())
            elif len(members) == 1:
                centers.append(members[0].copy())
            else:
                centers.append(
                    dtw_barycenter_averaging(
                        members,
                        constraint=self._dba_constraint,
                        max_iter=self.dba_max_iter,
                    )
                )
        if any(c.shape != centers[0].shape for c in centers):
            arr = np.empty(len(centers), dtype=object)
            for i, c in enumerate(centers):
                arr[i] = c
            return arr
        return np.stack(centers, axis=0)

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        precomputed_matrix: Optional[np.ndarray] = None,
    ) -> "KMeansDTW":
        """Compute DTW K-Means clustering.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_timesteps[, n_features])
        precomputed_matrix : ndarray of shape (n_samples, n_samples), optional
            Pre-computed pairwise distance matrix used for ``dtw_kmeans++``
            initialisation.  When provided, the expensive DTW computation is
            skipped.  Stored as ``self.distance_matrix_`` after fitting.

        Returns
        -------
        self
        """
        if isinstance(X, list):
            X_arr = np.empty(len(X), dtype=object)
            for i, t in enumerate(X):
                X_arr[i] = np.asarray(t)
            X = X_arr
        else:
            X = np.asarray(X, dtype=np.float64)
        n = len(X)
        if n < self.n_clusters:
            raise ValueError(
                f"n_samples={n} is smaller than n_clusters={self.n_clusters}"
            )

        rng = self._random_state()

        # Precompute full distance matrix once for initialisation.
        D_full = (
            precomputed_matrix
            if precomputed_matrix is not None
            else self._compute_distance_matrix(X)
        )
        self.distance_matrix_ = D_full

        if self.init == "dtw_kmeans++":
            centers = self._init_pp(X, D_full, rng)
        else:
            centers = self._init_random(X, rng)

        labels = np.full(n, -1, dtype=int)

        for it in range(self.max_iter):
            new_labels, inertia = self._assign(X, centers)

            n_changed = int((new_labels != labels).sum())
            labels = new_labels
            centers = self._update_centers(X, labels)

            self.n_iter_ = it + 1
            if n_changed / n <= self.tol:
                break

        # Final assignment against updated centres
        labels, inertia = self._assign(X, centers)

        self.labels_ = labels
        self.cluster_centers_ = centers
        self.inertia_ = inertia
        return self

    def _cluster_centers(self) -> np.ndarray:
        """Return DBA centroids, shape (n_clusters, T[, d])."""
        self._check_is_fitted()
        return self.cluster_centers_                      # type: ignore[return-value]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new samples to the nearest DBA centroid.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_timesteps[, n_features])

        Returns
        -------
        labels : ndarray of shape (n_samples,), dtype int
        """
        self._check_is_fitted()
        labels, _ = self._assign(X, self.cluster_centers_)
        return labels
