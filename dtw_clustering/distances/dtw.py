"""Dynamic Time Warping (DTW) distance and alignment path."""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
from joblib import Parallel, delayed

from .base import BaseDistance
from .constraints import BaseConstraint, NoConstraint, get_constraint


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _local_cost(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Squared-Euclidean local cost matrix C[i,j] = ||x[i] - y[j]||².

    Supports both univariate (shape ``(n,)``) and multivariate
    (shape ``(n, d)``) time series.
    """
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    diff = x[:, None, :] - y[None, :, :]   # (n, m, d)
    return np.sum(diff ** 2, axis=-1)        # (n, m)


def _accumulate(
    C: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Fill the DTW accumulated cost matrix in-place.

    Parameters
    ----------
    C : ndarray of shape (n, m) — local cost matrix
    mask : ndarray of shape (n, m), dtype bool — valid cell mask

    Returns
    -------
    D : ndarray of shape (n, m) — accumulated costs (inf where unreachable)
    """
    n, m = C.shape
    INF = np.inf
    D = np.full((n, m), INF)

    if mask[0, 0]:
        D[0, 0] = C[0, 0]
    for j in range(1, m):
        if mask[0, j]:
            D[0, j] = C[0, j] + D[0, j - 1]
    for i in range(1, n):
        if mask[i, 0]:
            D[i, 0] = C[i, 0] + D[i - 1, 0]

    for i in range(1, n):
        Di_prev = D[i - 1]
        Di = D[i]
        Ci = C[i]
        maski = mask[i]
        for j in range(1, m):
            if not maski[j]:
                continue
            prev = Di_prev[j]
            left = Di[j - 1]
            diag = Di_prev[j - 1]
            best = prev if prev < left else left
            if diag < best:
                best = diag
            Di[j] = Ci[j] + best

    return D


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dtw_path(
    x: np.ndarray,
    y: np.ndarray,
    constraint: Optional[BaseConstraint] = None,
) -> Tuple[List[Tuple[int, int]], float]:
    """Compute the DTW optimal alignment path and distance.

    Parameters
    ----------
    x, y : array-like of shape (n_timesteps,) or (n_timesteps, n_features)
    constraint : BaseConstraint, optional
        Warping constraint.  Defaults to :class:`NoConstraint`.

    Returns
    -------
    path : list of (i, j) index pairs from (0, 0) to (n-1, m-1)
    distance : float
        DTW distance (square root of the accumulated cost at the endpoint).

    Raises
    ------
    ValueError
        If the constraint leaves no valid path between the endpoints.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if constraint is None:
        constraint = NoConstraint()

    n, m = x.shape[0], y.shape[0]
    C = _local_cost(x, y)
    D = _accumulate(C, constraint.valid_cells(n, m))

    if not np.isfinite(D[n - 1, m - 1]):
        raise ValueError(
            "No valid DTW path found — the constraint is too restrictive "
            f"for sequences of length {n} and {m}."
        )

    # --- Backtrack ---
    path: List[Tuple[int, int]] = []
    i, j = n - 1, m - 1
    while i > 0 or j > 0:
        path.append((i, j))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            step = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
            if step == 0:
                i -= 1
                j -= 1
            elif step == 1:
                i -= 1
            else:
                j -= 1
    path.append((0, 0))
    path.reverse()

    return path, float(np.sqrt(D[n - 1, m - 1]))


def dtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    constraint: Optional[BaseConstraint] = None,
) -> float:
    """Compute the DTW distance between two time series.

    Equivalent to ``dtw_path(x, y, constraint)[1]`` but skips backtracking
    so it is slightly faster when the path is not needed.

    Parameters
    ----------
    x, y : array-like of shape (n_timesteps,) or (n_timesteps, n_features)
    constraint : BaseConstraint, optional

    Returns
    -------
    float
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if constraint is None:
        constraint = NoConstraint()

    n, m = x.shape[0], y.shape[0]
    C = _local_cost(x, y)
    D = _accumulate(C, constraint.valid_cells(n, m))

    val = D[n - 1, m - 1]
    if not np.isfinite(val):
        raise ValueError(
            "No valid DTW path found — the constraint is too restrictive "
            f"for sequences of length {n} and {m}."
        )
    return float(np.sqrt(val))


def dtw_distance_matrix(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    constraint: Optional[BaseConstraint] = None,
    n_jobs: int = 1,
) -> np.ndarray:
    """Compute a pairwise DTW distance matrix.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_timesteps[, n_features])
    Y : ndarray of shape (m_samples, n_timesteps[, n_features]), optional
        If ``None``, computes the symmetric ``X``-vs-``X`` matrix and only
        evaluates the upper triangle (O(n²/2) distance calls).
    constraint : BaseConstraint, optional
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
        return dtw_distance(X[i], Y[j], constraint=constraint)

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


class DTWDistance(BaseDistance):
    """DTW distance as a :class:`~dtw_clustering.distances.base.BaseDistance`.

    Parameters
    ----------
    constraint : str or BaseConstraint, optional
        ``"none"`` (default), ``"sakoe_chiba"``, or ``"itakura"``,
        or a :class:`BaseConstraint` instance.
    **constraint_kwargs
        Forwarded to the constraint constructor when ``constraint`` is a string
        (e.g. ``window=10`` for Sakoe-Chiba).

    Examples
    --------
    >>> dtw = DTWDistance(constraint="sakoe_chiba", window=10)
    >>> d = dtw(x, y)
    >>> D = dtw.pairwise(X, n_jobs=-1)
    """

    def __init__(
        self,
        constraint: Union[str, BaseConstraint, None] = None,
        **constraint_kwargs: object,
    ) -> None:
        if constraint is None or isinstance(constraint, str):
            self._constraint = get_constraint(constraint or "none", **constraint_kwargs)
        else:
            self._constraint = constraint

    @property
    def constraint(self) -> BaseConstraint:
        return self._constraint

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        return dtw_distance(x, y, constraint=self._constraint)

    def path(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], float]:
        """Return the optimal alignment path and the DTW distance."""
        return dtw_path(x, y, constraint=self._constraint)

    def pairwise(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        n_jobs: int = 1,
    ) -> np.ndarray:
        return dtw_distance_matrix(
            X, Y=Y, constraint=self._constraint, n_jobs=n_jobs
        )
