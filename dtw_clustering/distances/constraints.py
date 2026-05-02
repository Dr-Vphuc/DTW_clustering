"""Warping path constraints for DTW alignment."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseConstraint(ABC):
    """Base class for DTW warping path constraints."""

    @abstractmethod
    def valid_cells(self, n: int, m: int) -> np.ndarray:
        """Return a boolean mask of shape (n, m); True means the cell is reachable."""


class NoConstraint(BaseConstraint):
    """No constraint — full warping path space."""

    def valid_cells(self, n: int, m: int) -> np.ndarray:
        return np.ones((n, m), dtype=bool)


class SakoeChiba(BaseConstraint):
    """Sakoe-Chiba band constraint.

    Restricts the warping path to a symmetric band of width ``window`` around
    the main diagonal.  For sequences of unequal length the band is widened by
    ``|n - m|`` so that the endpoints are always reachable.

    Parameters
    ----------
    window : int
        Half-width of the band in samples (``window=0`` enforces the strict
        diagonal path; very large values are equivalent to no constraint).

    References
    ----------
    Sakoe & Chiba (1978), IEEE Trans. Acoust. Speech Signal Process.
    """

    def __init__(self, window: int) -> None:
        if window < 0:
            raise ValueError("window must be a non-negative integer")
        self.window = int(window)

    def valid_cells(self, n: int, m: int) -> np.ndarray:
        i_idx = np.arange(n)[:, None]
        j_idx = np.arange(m)[None, :]
        # Widen band so that both endpoints are reachable when n != m.
        w = max(self.window, abs(n - m))
        return np.abs(i_idx - j_idx) <= w


class Itakura(BaseConstraint):
    """Itakura parallelogram constraint.

    Limits the local slope of the warping path to ``[1/max_slope, max_slope]``.

    Parameters
    ----------
    max_slope : float
        Maximum allowed local slope (must be >= 1).

    References
    ----------
    Itakura (1975), IEEE Trans. Acoust. Speech Signal Process.
    """

    def __init__(self, max_slope: float = 2.0) -> None:
        if max_slope < 1.0:
            raise ValueError("max_slope must be >= 1.0")
        self.max_slope = float(max_slope)

    def valid_cells(self, n: int, m: int) -> np.ndarray:
        # Normalise indices to [0, 1] before computing slopes so that the
        # constraint is independent of sequence length.
        i_norm = np.arange(n, dtype=float)[:, None] / max(n - 1, 1)
        j_norm = np.arange(m, dtype=float)[None, :] / max(m - 1, 1)
        diff = i_norm - j_norm
        bound = (self.max_slope - 1.0) / self.max_slope
        return (diff >= -bound) & (diff <= bound)


_CONSTRAINTS: dict[str, type[BaseConstraint]] = {
    "none": NoConstraint,
    "sakoe_chiba": SakoeChiba,
    "itakura": Itakura,
}


def get_constraint(name: str, **kwargs: object) -> BaseConstraint:
    """Instantiate a constraint by name.

    Parameters
    ----------
    name : str
        One of ``"none"``, ``"sakoe_chiba"``, ``"itakura"``.
    **kwargs
        Forwarded to the constraint constructor (e.g. ``window=10``).

    Returns
    -------
    BaseConstraint
    """
    key = name.lower()
    if key not in _CONSTRAINTS:
        raise ValueError(
            f"Unknown constraint '{name}'. Available: {sorted(_CONSTRAINTS)}"
        )
    return _CONSTRAINTS[key](**kwargs)
