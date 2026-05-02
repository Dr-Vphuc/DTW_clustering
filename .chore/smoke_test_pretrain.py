"""Smoke test: joint pretrain (metric + reconstruction) on all 5 deep models."""
import sys
import time

import numpy as np

sys.path.insert(0, r"f:\NCKH\Framework")

from dtw_clustering.distances import soft_dtw_distance_matrix  # noqa: E402
from dtw_clustering.models.deep import (  # noqa: E402
    DAMIC,
    DEC,
    DECS,
    DSC,
    IDEC,
    TimeSeriesAutoencoder,
)


def _check_distance_matrix():
    print("=" * 60)
    print("[1] soft_dtw_distance_matrix sanity check")
    print("=" * 60)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 3, 16)).astype(np.float32)
    D = soft_dtw_distance_matrix(X, gamma=0.05, n_jobs=1)
    assert D.shape == (6, 6), D.shape
    assert np.allclose(D, D.T), "matrix not symmetric"
    assert np.allclose(np.diag(D), 0.0), "diagonal not zero"
    print(f"  shape {D.shape}  symmetric=True  diag=0  range=[{D.min():.3f},{D.max():.3f}]")
    print("  PASS\n")


def _make_data(n=64, d=2, t=24):
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, d, t)).astype(np.float32)


def _smoke_one(model_cls, name, ctor_kwargs=None, fit_kwargs=None):
    print("=" * 60)
    print(f"[*] {name}")
    print("=" * 60)
    ctor_kwargs = ctor_kwargs or {}
    fit_kwargs = fit_kwargs or {}
    X = _make_data()
    ae = TimeSeriesAutoencoder(input_channels=2, seq_len=24, latent_dim=8)
    model = model_cls(n_clusters=3, autoencoder=ae, random_state=0, **ctor_kwargs)
    t0 = time.time()
    common = dict(
        pretrain_epochs=2,
        pretrain_lr=1e-3,
        sdtw_gamma=0.1,
        verbose=True,
    )
    common.update(fit_kwargs)
    model.fit(X, **common)
    elapsed = time.time() - t0
    assert hasattr(model, "labels_") and model.labels_ is not None
    assert model.labels_.shape == (len(X),)
    print(f"  labels shape {model.labels_.shape}  unique={np.unique(model.labels_).tolist()}  time={elapsed:.1f}s")
    print("  PASS\n")


def main():
    _check_distance_matrix()

    _smoke_one(DEC, "DEC",
               ctor_kwargs={"update_interval": 2},
               fit_kwargs={"epochs": 2, "batch_size": 32})
    _smoke_one(IDEC, "IDEC",
               ctor_kwargs={"update_interval": 2},
               fit_kwargs={"epochs": 2, "batch_size": 32})
    _smoke_one(DECS, "DECS",
               ctor_kwargs={"update_interval": 2},
               fit_kwargs={"epochs": 2, "batch_size": 32})
    _smoke_one(DSC, "DSC",
               fit_kwargs={"epochs": 2})
    _smoke_one(DAMIC, "DAMIC",
               ctor_kwargs={"ce_warmup_epochs": 1},
               fit_kwargs={"epochs": 2, "batch_size": 32,
                           "cluster_ae_epochs": 1})

    print("=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
