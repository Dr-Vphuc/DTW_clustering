"""Distance registry: maps string names to distance classes.

Users and other modules can retrieve distance objects by name rather than
importing them directly, which makes configuration-driven workflows (e.g.
YAML experiment configs) straightforward.

Usage
-----
>>> from dtw_clustering.distances import get_distance
>>> dist = get_distance("dtw", constraint="sakoe_chiba", window=10)
>>> dist = get_distance("soft_dtw", gamma=0.5)
"""
from __future__ import annotations

from typing import Any

from .base import BaseDistance
from .dtw import DTWDistance
from .soft_dtw import SoftDTWDistance

# Maps canonical string keys to (class, accepted_kwargs) pairs.
# The set of accepted kwargs is used only for documentation; all **kwargs
# are forwarded to the constructor — unknown kwargs will raise there.
_REGISTRY: dict[str, type[BaseDistance]] = {
    "dtw": DTWDistance,
    "soft_dtw": SoftDTWDistance,
}


def get_distance(name: str, **kwargs: Any) -> BaseDistance:
    """Instantiate a distance object by name.

    Parameters
    ----------
    name : str
        ``"dtw"`` or ``"soft_dtw"``.
    **kwargs
        Forwarded to the distance constructor.

        - For ``"dtw"``: ``constraint`` (str or :class:`BaseConstraint`),
          plus any constraint-specific kwargs (e.g. ``window=10``).
        - For ``"soft_dtw"``: ``gamma`` (float).

    Returns
    -------
    BaseDistance

    Raises
    ------
    ValueError
        If ``name`` is not in the registry.

    Examples
    --------
    >>> dtw = get_distance("dtw")
    >>> dtw_sc = get_distance("dtw", constraint="sakoe_chiba", window=10)
    >>> sdtw = get_distance("soft_dtw", gamma=0.1)
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown distance '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key](**kwargs)


def list_distances() -> list[str]:
    """Return the names of all registered distance functions."""
    return sorted(_REGISTRY)
