"""DTW clustering models."""
from .base import BaseClusterer
from .classical import (
    AffinityPropagationDTW,
    AgglomerativeDTW,
    KMeansDTW,
    KMedoidsDTW,
    SpectralClusteringDTW,
)
from .deep import (
    DAMIC,
    DEC,
    DECS,
    DSC,
    IDEC,
    DeepClusterer,
    SoftDTWReconstructionLoss,
    TimeSeriesAutoencoder,
)

__all__ = [
    # Classical
    "BaseClusterer",
    "KMedoidsDTW",
    "AgglomerativeDTW",
    "KMeansDTW",
    "SpectralClusteringDTW",
    "AffinityPropagationDTW",
    # Deep
    "DeepClusterer",
    "TimeSeriesAutoencoder",
    "SoftDTWReconstructionLoss",
    "DEC",
    "IDEC",
    "DECS",
    "DSC",
    "DAMIC",
]
