import numpy as np

from ...distances.soft_dtw import soft_dtw_distance_matrix
from .autoencoder import SoftDTWReconstructionLoss
from .base import _get_torch
from .dec import DEC, _soft_assign


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _augment(x, jitter_std, scale_std):
    """
    Time-series augmentation: additive Gaussian jitter + per-channel magnitude scaling.
    x: (B, d, T) tensor on any device.
    """
    torch = _get_torch()
    noise = torch.randn_like(x) * jitter_std
    scale = 1.0 + torch.randn(
        x.size(0), x.size(1), 1, device=x.device, dtype=x.dtype
    ) * scale_std
    return (x + noise) * scale


def _otsu_threshold(values):
    """
    Otsu's method on a flattened numpy array of assignment probabilities.
    Returns the threshold that maximises between-class variance.
    """
    v = np.asarray(values).ravel()
    if v.size == 0:
        return 0.5
    vmin, vmax = float(v.min()), float(v.max())
    if vmax - vmin < 1e-12:
        return float(vmin)

    hist, edges = np.histogram(v, bins=256, range=(vmin, vmax))
    centers = 0.5 * (edges[:-1] + edges[1:])

    total = hist.sum()
    sum_total = float((hist * centers).sum())

    w_b = 0
    sum_b = 0.0
    best_var = -1.0
    best_t = float(centers[0])
    for i in range(len(hist)):
        w_b += int(hist[i])
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += float(hist[i]) * float(centers[i])
        mu_b = sum_b / w_b
        mu_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (mu_b - mu_f) ** 2
        if var_between > best_var:
            best_var = var_between
            best_t = float(centers[i])
    return best_t


# ---------------------------------------------------------------------------
# DECS
# ---------------------------------------------------------------------------

class DECS(DEC):
    """
    Deep Embedded Clustering driven by Sample Stability (DECS) for time-series.

    Differences from DEC
    --------------------
    Pretraining    : Soft-DTW reconstruction on **augmented** inputs
                     (time-series jitter + magnitude scaling).
    Fine-tuning    : Decoder discarded (like DEC), but the clustering loss
                     is Sample Stability instead of KL divergence:

        q_ij  : Student's t soft assignment (α=1)
        t     : Otsu threshold over all q values (refreshed periodically)
        fq_ij : (q_ij - t)² / t²     if q_ij < t
                (q_ij - t)² / (1-t)² if q_ij ≥ t
        sq_i  : mean_j(fq_ij) − λ · var_j(fq_ij)        (biased variance)
        L_c   : 1 − mean_i(sq_i)                        (instability)

    Parameters
    ----------
    n_clusters      : int
    autoencoder     : TimeSeriesAutoencoder (nn.Module)
    lam             : float  Weight of the variance term in sq
    jitter_std      : float  Std of additive Gaussian noise during pretraining
    scale_std       : float  Std of per-channel magnitude scaling
    update_interval : int    Mini-batch iterations between Otsu/convergence refresh
    tol             : float  Convergence threshold on label-change fraction
    device          : str or None
    random_state    : int or None
    """

    def __init__(
        self,
        n_clusters,
        autoencoder,
        lam=1.0,
        jitter_std=0.03,
        scale_std=0.05,
        update_interval=140,
        tol=0.001,
        device=None,
        random_state=None,
    ):
        super().__init__(
            n_clusters=n_clusters,
            autoencoder=autoencoder,
            update_interval=update_interval,
            tol=tol,
            device=device,
            random_state=random_state,
        )
        self.lam = lam
        self.jitter_std = jitter_std
        self.scale_std = scale_std
        self.threshold_ = None

    # ------------------------------------------------------------------
    # pretrain (override to add augmentation)
    # ------------------------------------------------------------------

    def pretrain(
        self,
        X,
        epochs=100,
        lr=1e-3,
        batch_size=256,
        sdtw_gamma=0.1,
        metric_target_gamma=0.05,
        lambda_metric=1.0,
        lambda_recon=0.5,
        pairs_per_step=None,
        target_matrix=None,
        n_jobs=1,
        verbose=True,
    ):
        """
        Joint pretrain for DECS: siamese metric loss on **clean** inputs +
        Soft-DTW reconstruction loss on **augmented** inputs (denoising AE).

        See ``DeepClusterer.pretrain`` for the meaning of the joint-loss
        arguments. Augmentation parameters (``self.jitter_std``,
        ``self.scale_std``) come from the constructor.
        """
        torch = _get_torch()
        X_t = self._to_tensor(X)
        N = len(X_t)
        if pairs_per_step is None:
            pairs_per_step = 4 * batch_size

        # ---- Precompute target matrix (CPU, numpy) on CLEAN inputs ----
        if target_matrix is None:
            if verbose:
                print(
                    f"[DECS-Pretrain] Computing N×N Soft-DTW target matrix "
                    f"(N={N}, gamma={metric_target_gamma}, n_jobs={n_jobs})..."
                )
            X_np = X if isinstance(X, np.ndarray) else X_t.detach().cpu().numpy()
            target_matrix = soft_dtw_distance_matrix(
                X_np, gamma=metric_target_gamma, n_jobs=n_jobs
            )
        D = np.asarray(target_matrix, dtype=np.float32)
        if D.shape != (N, N):
            raise ValueError(
                f"target_matrix shape {D.shape} does not match (N, N)=({N}, {N})"
            )

        recon_loss_fn = SoftDTWReconstructionLoss(gamma=sdtw_gamma, normalize=True)
        recon_loss_fn.to(self.device)
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)

        self.autoencoder.train()
        rng = np.random.default_rng(self.random_state)
        log_every = max(1, epochs // 10)

        for epoch in range(epochs):
            idx = rng.permutation(N)
            ep_metric, ep_recon, ep_total, nb = 0.0, 0.0, 0.0, 0
            for start in range(0, N, batch_size):
                b_global = idx[start: start + batch_size]
                B = len(b_global)
                if B < 2:
                    continue

                # ---- sample pairs (local indices) ----
                p_i = rng.integers(0, B, size=pairs_per_step)
                p_j = rng.integers(0, B, size=pairs_per_step)
                same = p_i == p_j
                if same.any():
                    p_j[same] = (p_j[same] + 1) % B

                t = D[b_global[p_i], b_global[p_j]]
                t_mean, t_std = float(t.mean()), float(t.std() + 1e-6)
                t = (t - t_mean) / t_std
                t_min = float(t.min())
                if t_min < 0:
                    t = t + abs(t_min)
                t = torch.from_numpy(t.astype(np.float32)).to(self.device)

                batch_clean = X_t[b_global]
                batch_aug = _augment(batch_clean, self.jitter_std, self.scale_std)

                optimizer.zero_grad()

                # Metric loss on CLEAN inputs (preserves DTW geometry)
                z_clean = self.autoencoder.encode(batch_clean)
                d_pred = torch.norm(z_clean[p_i] - z_clean[p_j], dim=-1)
                L_metric = torch.nn.functional.mse_loss(d_pred, t)

                # Reconstruction loss on AUGMENTED inputs (denoising AE)
                _, x_hat_aug = self.autoencoder(batch_aug)
                L_recon = recon_loss_fn(batch_aug, x_hat_aug)

                loss = lambda_metric * L_metric + lambda_recon * L_recon
                loss.backward()
                optimizer.step()

                ep_metric += L_metric.item()
                ep_recon += L_recon.item()
                ep_total += loss.item()
                nb += 1

            if verbose and (epoch + 1) % log_every == 0:
                print(
                    f"[DECS-Pretrain] epoch {epoch + 1}/{epochs}  "
                    f"total={ep_total / max(1, nb):.4f}  "
                    f"metric={ep_metric / max(1, nb):.4f}  "
                    f"recon={ep_recon / max(1, nb):.4f}"
                )

        self._pretrained = True

    # ------------------------------------------------------------------
    # fit (override: sample-stability loss, Otsu threshold)
    # ------------------------------------------------------------------

    def fit(
        self,
        X,
        epochs=200,
        lr=1e-4,
        batch_size=256,
        pretrain_epochs=100,
        pretrain_lr=1e-3,
        sdtw_gamma=0.1,
        verbose=True,
    ):
        """
        Parameters
        ----------
        X               : ndarray (N, d, T)
        epochs          : int
        lr              : float
        batch_size      : int
        pretrain_epochs : int   Used only if not yet pretrained
        pretrain_lr     : float
        sdtw_gamma      : float Soft-DTW γ for pretraining
        verbose         : bool
        """
        torch = _get_torch()
        import torch.nn as nn
        from sklearn.cluster import KMeans

        if not self._pretrained:
            self.pretrain(
                X, epochs=pretrain_epochs, lr=pretrain_lr,
                sdtw_gamma=sdtw_gamma, verbose=verbose,
            )

        X_t = self._to_tensor(X)
        N = len(X_t)
        latent_dim = self.autoencoder.latent_dim

        # ---- Cluster-centre layer ----
        class _CentersModule(nn.Module):
            def __init__(self, k, d):
                super().__init__()
                self.centers = nn.Parameter(torch.zeros(k, d))

        self._clustering_layer = _CentersModule(self.n_clusters, latent_dim).to(self.device)

        # ---- K-Means initialisation ----
        Z_cpu = self._get_embeddings(X_t, batch_size).numpy()
        km = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=self.random_state)
        km.fit(Z_cpu)
        self._clustering_layer.centers.data.copy_(
            torch.tensor(km.cluster_centers_, dtype=torch.float32)
        )
        prev_labels = km.labels_.copy()
        if verbose:
            print(f"[DECS] K-Means init done. Fine-tuning for up to {epochs} epochs.")

        # ---- Optimiser: encoder (no decoder) + cluster centres ----
        enc_params = (
            list(self.autoencoder.encoder_conv.parameters())
            + list(self.autoencoder.encoder_fc.parameters())
        )
        optimizer = torch.optim.Adam(
            enc_params + list(self._clustering_layer.parameters()), lr=lr
        )

        iteration = 0
        t_thr = 1.0 / self.n_clusters     # uniform baseline until first Otsu refresh
        rng = self._random_state()
        converged = False

        for _ in range(epochs):
            perm = rng.permutation(N)

            for start in range(0, N, batch_size):
                # -- Refresh Otsu threshold + convergence check --
                if iteration % self.update_interval == 0:
                    Q_cpu = self._q_full(X_t, batch_size)           # (N, k)
                    t_thr = _otsu_threshold(Q_cpu.numpy())

                    new_labels = Q_cpu.argmax(1).numpy()
                    delta = float((new_labels != prev_labels).sum()) / N
                    if verbose and iteration > 0:
                        print(
                            f"[DECS] iter {iteration:5d}  t={t_thr:.4f}  "
                            f"label-change={delta:.4f}"
                        )
                    if iteration > 0 and delta < self.tol:
                        if verbose:
                            print("[DECS] Converged.")
                        converged = True
                        break
                    prev_labels = new_labels.copy()

                # -- Mini-batch update --
                idx = perm[start: start + batch_size]
                x_b = X_t[idx]
                z_b = self.autoencoder.encode(x_b)                          # (B, d_z)
                q_b = _soft_assign(z_b, self._clustering_layer.centers)     # (B, k)

                # Determinacy fq: piecewise (q − t)² / {t², (1−t)²}
                diff = q_b - t_thr
                denom = torch.where(
                    q_b < t_thr,
                    torch.full_like(q_b, t_thr),
                    torch.full_like(q_b, 1.0 - t_thr),
                ).clamp(min=1e-8)
                fq = (diff * diff) / (denom * denom)                         # (B, k)

                # Sample stability: mean − λ · var (biased variance, 1/k)
                mean_fq = fq.mean(dim=1)                                     # (B,)
                var_fq = fq.var(dim=1, unbiased=False)                       # (B,)
                sq = mean_fq - self.lam * var_fq                             # (B,)

                loss = 1.0 - sq.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iteration += 1

            if converged:
                break

        # ---- Final assignment ----
        Q_final = self._q_full(X_t, batch_size)
        self.labels_ = Q_final.argmax(1).numpy().astype(int)
        self.cluster_centers_ = self._clustering_layer.centers.detach().cpu().numpy()
        self.threshold_ = float(t_thr)
