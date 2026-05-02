"""Abstract base class for all DTW-based clustering models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np

from dtw_clustering.distances.base import BaseDistance
from dtw_clustering.distances._registry import get_distance


class BaseClusterer(ABC):
    """Common interface for all DTW-based clustering algorithms.

    All concrete clusterers should inherit from this class.  It provides:

    * A unified constructor that wires up the distance metric.
    * :meth:`_compute_distance_matrix` — computes the pairwise DTW distance
      matrix that all classical algorithms operate on.
    * A sklearn-compatible ``fit`` / ``fit_predict`` / ``predict`` API.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    metric : str or BaseDistance, optional
        Distance metric.  Accepts the string names registered in
        :mod:`dtw_clustering.distances._registry` (``"dtw"``,
        ``"soft_dtw"``) or a :class:`~dtw_clustering.distances.BaseDistance`
        instance.  Defaults to ``"dtw"``.
    n_jobs : int
        Number of parallel workers for distance matrix computation.
        ``-1`` uses all available CPUs.
    random_state : int or None
        Seed for any internal random number generation.
    **metric_kwargs
        Forwarded to the metric constructor when ``metric`` is a string
        (e.g. ``constraint="sakoe_chiba"``, ``window=10``).
    """

    def __init__(
        self,
        n_clusters: int,
        metric: Union[str, BaseDistance] = "dtw",
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        **metric_kwargs,
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer")
        self.n_clusters = n_clusters
        self.n_jobs = n_jobs
        self.random_state = random_state

        if isinstance(metric, BaseDistance):
            self.metric: BaseDistance = metric
        else:
            self.metric = get_distance(metric, **metric_kwargs)

        # Set after fit()
        self.labels_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Distance matrix
    # ------------------------------------------------------------------

    def _compute_distance_matrix(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the pairwise distance matrix used by classical algorithms.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_timesteps[, n_features])
        Y : ndarray of shape (m_samples, n_timesteps[, n_features]), optional
            If ``None``, the symmetric ``X``-vs-``X`` matrix is returned.

        Returns
        -------
        D : ndarray of shape (n_samples, n_samples) or (n_samples, m_samples)
            Pairwise distance matrix with ``D[i, j] >= 0`` and
            ``D[i, i] == 0`` (when ``Y`` is ``None``).
        """
        return self.metric.pairwise(X, Y=Y, n_jobs=self.n_jobs)

    # ------------------------------------------------------------------
    # sklearn-compatible API
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseClusterer":
        """Compute clustering from time series array ``X``.

        Must set ``self.labels_`` (shape ``(n_samples,)``) and return
        ``self``.
        """

    def fit_predict(
        self,
        X: np.ndarray,
        precomputed_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fit and return cluster labels for each sample.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_timesteps[, n_features])
        precomputed_matrix : ndarray of shape (n_samples, n_samples), optional
            Pre-computed pairwise distance matrix.  Forwarded to :meth:`fit`.

        Returns
        -------
        labels : ndarray of shape (n_samples,), dtype int
        """
        return self.fit(X, precomputed_matrix=precomputed_matrix).labels_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new samples to the nearest cluster.

        The default implementation computes DTW distances from each sample in
        ``X`` to each cluster centre and returns the index of the nearest
        centre.  Subclasses that have a different notion of "centre" (e.g.
        medoids) should override :meth:`_cluster_centers` instead.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_timesteps[, n_features])

        Returns
        -------
        labels : ndarray of shape (n_samples,), dtype int

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        self._check_is_fitted()
        centers = self._cluster_centers()
        D = self._compute_distance_matrix(X, Y=centers)
        return np.argmin(D, axis=1).astype(int)

    @abstractmethod
    def _cluster_centers(self) -> np.ndarray:
        """Return the cluster centre time series, shape (n_clusters, T[, d])."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        if self.labels_ is None:
            raise RuntimeError(
                f"{type(self).__name__} has not been fitted. Call fit() first."
            )

    def _random_state(self) -> np.random.RandomState:
        if isinstance(self.random_state, np.random.RandomState):
            return self.random_state
        return np.random.RandomState(self.random_state)

    def __repr__(self) -> str:
        metric_name = getattr(self.metric, "__class__", type(self.metric)).__name__
        return (
            f"{type(self).__name__}("
            f"n_clusters={self.n_clusters}, "
            f"metric={metric_name})"
        )
