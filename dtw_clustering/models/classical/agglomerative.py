"""Agglomerative (hierarchical) clustering with DTW distance.

The algorithm builds a dendrogram bottom-up by successively merging the two
clusters with the smallest inter-cluster distance.  Because it operates on a
precomputed pairwise distance matrix it is fully compatible with DTW and any
other non-Euclidean metric.

Linkage criteria
----------------
All four standard linkage strategies are supported:

* ``"single"``   — minimum pairwise distance (chaining-prone).
* ``"complete"`` — maximum pairwise distance (compact, balanced clusters).
* ``"average"``  — UPGMA: mean of all pairwise distances.
* ``"ward"``     — minimise the increase in total within-cluster variance.
  Ward's method is derived from Euclidean distances, so the result may
  differ from a Ward linkage applied directly to the raw time series.
  It is included for completeness but ``"complete"`` or ``"average"``
  are typically preferred with DTW.

The dendrogram is computed with :func:`scipy.cluster.hierarchy.linkage`
and cut at ``n_clusters`` with :func:`~scipy.cluster.hierarchy.fcluster`.

References
----------
Müllner, D. (2011). "Modern hierarchical, agglomerative clustering
    algorithms." https://arxiv.org/abs/1109.2378
"""
from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from dtw_clustering.distances.base import BaseDistance
from dtw_clustering.models.base import BaseClusterer


_LINKAGE_METHODS = ("single", "complete", "average", "ward")


class AgglomerativeDTW(BaseClusterer):
    """Agglomerative clustering using a precomputed DTW distance matrix.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to form (dendrogram cut height).
    metric : str or BaseDistance
        Distance metric.  Defaults to ``"dtw"``.
    linkage : {"single", "complete", "average", "ward"}
        Linkage criterion.  ``"ward"`` is derived from Euclidean geometry;
        prefer ``"complete"`` or ``"average"`` when using DTW.
    n_jobs : int
        Parallel workers for distance matrix computation.
    random_state : int or None
        Not used; kept for API consistency with other clusterers.
    **metric_kwargs
        Forwarded to the metric constructor.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster index of each sample (set after :meth:`fit`).
    linkage_matrix_ : ndarray of shape (n_samples - 1, 4)
        The full hierarchical linkage matrix returned by
        :func:`scipy.cluster.hierarchy.linkage`.  Use it to draw
        dendrograms or cut the tree at different heights.

    Examples
    --------
    >>> from dtw_clustering.models.classical import AgglomerativeDTW
    >>> model = AgglomerativeDTW(n_clusters=4, linkage="complete")
    >>> labels = model.fit_predict(X)

    >>> # Draw the dendrogram
    >>> from scipy.cluster.hierarchy import dendrogram
    >>> import matplotlib.pyplot as plt
    >>> dendrogram(model.linkage_matrix_)
    >>> plt.show()
    """

    def __init__(
        self,
        n_clusters: int = 2,
        metric: Union[str, BaseDistance] = "dtw",
        linkage_method: Literal["single", "complete", "average", "ward"] = "complete",
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
        if linkage_method not in _LINKAGE_METHODS:
            raise ValueError(
                f"linkage_method must be one of {_LINKAGE_METHODS}, "
                f"got '{linkage_method}'"
            )
        self.linkage_method = linkage_method

        self.linkage_matrix_: Optional[np.ndarray] = None
        self._X_fit: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "AgglomerativeDTW":
        """Compute hierarchical clustering and cut at ``n_clusters``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_timesteps[, n_features])

        Returns
        -------
        self
        """
        X = np.asarray(X)
        n = len(X)
        if n < self.n_clusters:
            raise ValueError(
                f"n_samples={n} is smaller than n_clusters={self.n_clusters}"
            )

        D = self._compute_distance_matrix(X)             # (n, n), symmetric

        # scipy.linkage expects a condensed (upper-triangle) distance vector.
        condensed = squareform(D, checks=False)

        self.linkage_matrix_ = linkage(condensed, method=self.linkage_method)

        # fcluster returns 1-based labels; shift to 0-based.
        raw_labels = fcluster(
            self.linkage_matrix_, t=self.n_clusters, criterion="maxclust"
        )
        self.labels_ = (raw_labels - 1).astype(int)
        self._X_fit = X
        return self

    # ------------------------------------------------------------------
    # Cluster centres and predict
    # ------------------------------------------------------------------

    def _cluster_centers(self) -> np.ndarray:
        """Return the per-cluster arithmetic mean (medoid-like representative).

        For agglomerative clustering the "centre" is not a medoid but the
        mean of the cluster members.  This is used only by the base-class
        ``predict()`` method for assigning *new* samples to the nearest
        cluster.
        """
        self._check_is_fitted()
        X = self._X_fit
        centers = np.stack(
            [X[self.labels_ == k].mean(axis=0) for k in range(self.n_clusters)],
            axis=0,
        )
        return centers

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new samples to the nearest cluster centre.

        .. note::
            Agglomerative clustering is inherently transductive.  This method
            is provided for convenience; for true inductive use consider
            :class:`KMedoidsDTW`.

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
