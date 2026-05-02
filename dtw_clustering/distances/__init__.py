"""Public API for ``dtw_clustering.distances``.

Core distance functions and classes are re-exported here so that users can
write ``from dtw_clustering.distances import DTWDistance`` without needing to
know which submodule each symbol lives in.
"""
from .base import BaseDistance
from .constraints import (
    BaseConstraint,
    Itakura,
    NoConstraint,
    SakoeChiba,
    get_constraint,
)
from .dtw import DTWDistance, dtw_distance, dtw_distance_matrix, dtw_path
from .soft_dtw import (
    SoftDTW,
    SoftDTWDistance,
    soft_dtw_distance_matrix,
    soft_dtw_grad,
    soft_dtw_value,
)
from .barycenter import dtw_barycenter_averaging, soft_dtw_barycenter
from ._registry import get_distance, list_distances

__all__ = [
    # Base
    "BaseDistance",
    # Constraints
    "BaseConstraint",
    "NoConstraint",
    "SakoeChiba",
    "Itakura",
    "get_constraint",
    # DTW
    "DTWDistance",
    "dtw_distance",
    "dtw_distance_matrix",
    "dtw_path",
    # Soft-DTW
    "SoftDTW",
    "SoftDTWDistance",
    "soft_dtw_value",
    "soft_dtw_grad",
    "soft_dtw_distance_matrix",
    # Barycenter
    "dtw_barycenter_averaging",
    "soft_dtw_barycenter",
    # Registry
    "get_distance",
    "list_distances",
]
