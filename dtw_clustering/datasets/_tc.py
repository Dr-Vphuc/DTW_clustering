from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

_DEFAULT_CSV = Path(__file__).parents[2] / ".data" / "historical_TC.csv"
_REQUIRED_COLS = ["StormID", "Time", "Latitude", "Longitude"]


@dataclass
class TCDataset:
    """Loaded tropical-cyclone dataset.

    Attributes
    ----------
    tracks_raw : list of ndarray, each shape (T_i, 2)
        Variable-length tracks [Latitude, Longitude].  Use with classical models.
    track_info : DataFrame
        One row per track: StormID, StormName, n_points.
    X : ndarray (N, 2, T) or None
        Channel-first resampled array ready for deep models.
        None when ``resample_len`` was not supplied.
    scaler : StandardScaler or None
        Fitted scaler used to standardise X.  None when standardisation was
        skipped or X was not produced.
    """

    tracks_raw: List[np.ndarray]
    track_info: pd.DataFrame
    X: Optional[np.ndarray] = field(default=None)
    scaler: Optional[object] = field(default=None)


def _resample_linear(track: np.ndarray, L: int) -> np.ndarray:
    """Linearly resample a (T, 2) track to exactly L points."""
    T = track.shape[0]
    if T == L:
        return track.astype(np.float32, copy=True)
    if T == 1:
        return np.repeat(track.astype(np.float32), L, axis=0)
    x_old = np.linspace(0.0, 1.0, T, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, L, dtype=np.float32)
    lat = np.interp(x_new, x_old, track[:, 0]).astype(np.float32)
    lon = np.interp(x_new, x_old, track[:, 1]).astype(np.float32)
    return np.stack([lat, lon], axis=1)  # (L, 2)


def load_historical_tc(
    csv_path=None,
    min_points: int = 5,
    filter_lon_0_180: bool = True,
    dayfirst: bool = True,
    resample_len: Optional[int] = None,
    standardize: bool = True,
) -> TCDataset:
    """Load and preprocess the historical tropical-cyclone dataset.

    Parameters
    ----------
    csv_path : str or Path or None
        Path to the CSV file.  Defaults to ``<project_root>/.data/historical_TC.csv``.
    min_points : int
        Minimum number of time steps a track must have to be kept.
    filter_lon_0_180 : bool
        If True, drop rows where Longitude is outside [0, 180].
    dayfirst : bool
        Passed to ``pd.to_datetime``; True for DD/MM/YYYY format.
    resample_len : int or None
        If provided, resample every track to exactly this many steps and
        return ``X`` of shape ``(N, 2, resample_len)``.
    standardize : bool
        If True and ``resample_len`` is set, standardise X with
        ``StandardScaler`` fitted on the full dataset.

    Returns
    -------
    TCDataset
    """
    path = Path(csv_path) if csv_path is not None else _DEFAULT_CSV

    df = pd.read_csv(path)
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Found: {df.columns.tolist()}"
        )

    if "StormName" not in df.columns:
        df["StormName"] = ""

    df["Time"] = pd.to_datetime(df["Time"], dayfirst=dayfirst, errors="coerce")
    df = df.dropna(subset=_REQUIRED_COLS).copy()
    df["StormID"] = df["StormID"].astype(str)
    df["StormName"] = df["StormName"].fillna("").astype(str)

    if filter_lon_0_180:
        df = df[(df["Longitude"] >= 0) & (df["Longitude"] <= 180)]

    tracks_raw: List[np.ndarray] = []
    meta: List[tuple] = []

    for (sid, sname), g in df.groupby(["StormID", "StormName"], sort=False):
        g2 = g.sort_values("Time")
        xy = g2[["Latitude", "Longitude"]].to_numpy(dtype=np.float32)
        if len(xy) >= min_points:
            tracks_raw.append(xy)
            meta.append((sid, sname))

    lengths = np.array([len(t) for t in tracks_raw], dtype=int)
    track_info = pd.DataFrame(
        {
            "StormID": [sid for sid, _ in meta],
            "StormName": [sname for _, sname in meta],
            "n_points": lengths,
        }
    )

    if resample_len is None:
        return TCDataset(tracks_raw=tracks_raw, track_info=track_info)

    # Deep-model path: resample → (N, L, 2) → channel-first (N, 2, L)
    tracks_rs = np.stack(
        [_resample_linear(t, resample_len) for t in tracks_raw], axis=0
    )  # (N, L, 2)

    scaler = None
    if standardize:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        flat = tracks_rs.reshape(-1, 2)
        flat_z = scaler.fit_transform(flat)
        tracks_rs = flat_z.reshape(tracks_rs.shape).astype(np.float32)

    X = tracks_rs.transpose(0, 2, 1)  # (N, 2, L)
    return TCDataset(tracks_raw=tracks_raw, track_info=track_info, X=X, scaler=scaler)
