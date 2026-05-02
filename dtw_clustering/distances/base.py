"""Abstract base class for pairwise distance metrics."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from joblib import Parallel, delayed


class BaseDistance(ABC):
    """Common interface for all pairwise distance functions.

    Subclasses must implement :meth:`__call__`.  A default :meth:`pairwise`
    implementation is provided using ``joblib`` parallelism; subclasses may
    override it with a more efficient vectorised version.
    """

    @abstractmethod
    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute the distance between two time series.

        Parameters
        ----------
        x, y : ndarray of shape (n_timesteps,) or (n_timesteps, n_features)

        Returns
        -------
        float
        """

    def pairwise(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        n_jobs: int = 1,
    ) -> np.ndarray:
        """Compute a pairwise distance matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_timesteps[, n_features])
        Y : ndarray of shape (m_samples, n_timesteps[, n_features]), optional
            If ``None``, the symmetric ``X``-vs-``X`` matrix is computed and
            only the upper triangle is evaluated (O(n²/2) calls).
        n_jobs : int
            Number of parallel jobs forwarded to ``joblib.Parallel``.

        Returns
        -------
        D : ndarray of shape (n_samples, n_samples) or (n_samples, m_samples)
        """
        symmetric = Y is None
        if symmetric:
            Y = X
        n, m = len(X), len(Y)

        if symmetric:
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            vals = Parallel(n_jobs=n_jobs)(
                delayed(self)(X[i], Y[j]) for i, j in pairs
            )
            D = np.zeros((n, n), dtype=np.float64)
            for (i, j), v in zip(pairs, vals):
                D[i, j] = v
                D[j, i] = v
        else:
            pairs = [(i, j) for i in range(n) for j in range(m)]
            vals = Parallel(n_jobs=n_jobs)(
                delayed(self)(X[i], Y[j]) for i, j in pairs
            )
            D = np.array(vals, dtype=np.float64).reshape(n, m)

        return D
