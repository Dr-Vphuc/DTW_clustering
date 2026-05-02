"""Verify joint pretrain achieves the goal: ||z_i - z_j||_2 ≈ DTW(x_i, x_j)."""
import sys
sys.path.insert(0, r"f:\NCKH\Framework")

import numpy as np
import torch

from dtw_clustering.distances import soft_dtw_distance_matrix
from dtw_clustering.models.deep import DEC, TimeSeriesAutoencoder


def correlation(model, X, n_pairs=200):
    rng = np.random.default_rng(0)
    n = len(X)
    pi = rng.integers(0, n, size=n_pairs)
    pj = rng.integers(0, n, size=n_pairs)
    same = pi == pj
    pj[same] = (pj[same] + 1) % n

    X_t = torch.from_numpy(X).to(model.device)
    model.autoencoder.eval()
    with torch.no_grad():
        z = model.autoencoder.encode(X_t).cpu().numpy()

    D_dtw = soft_dtw_distance_matrix(X, gamma=0.05, n_jobs=-1)
    d_pred = np.linalg.norm(z[pi] - z[pj], axis=-1)
    d_target = D_dtw[pi, pj]
    return float(np.corrcoef(d_pred, d_target)[0, 1])


def main():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 2, 20)).astype(np.float32)

    # Joint pretrain (default)
    ae = TimeSeriesAutoencoder(input_channels=2, seq_len=20, latent_dim=8)
    model = DEC(n_clusters=3, autoencoder=ae, random_state=0, update_interval=2)
    model.pretrain(X, epochs=20, batch_size=64, n_jobs=-1, verbose=False)
    rho_joint = correlation(model, X)
    print(f"After 20 epochs of joint pretrain  Pearson(||z||, DTW) = {rho_joint:.3f}")

    # Reconstruction-only baseline (set lambda_metric=0)
    ae2 = TimeSeriesAutoencoder(input_channels=2, seq_len=20, latent_dim=8)
    model2 = DEC(n_clusters=3, autoencoder=ae2, random_state=0, update_interval=2)
    model2.pretrain(X, epochs=20, batch_size=64,
                    lambda_metric=0.0, lambda_recon=1.0, n_jobs=-1, verbose=False)
    rho_recon = correlation(model2, X)
    print(f"After 20 epochs of recon-only       Pearson(||z||, DTW) = {rho_recon:.3f}")

    print()
    print(f"Improvement: +{(rho_joint - rho_recon):.3f}")
    if rho_joint > rho_recon + 0.05:
        print("PASS: joint pretrain achieves stronger DTW-Euclidean alignment")
    else:
        print("WARNING: joint pretrain not clearly better than recon-only")


if __name__ == "__main__":
    main()
