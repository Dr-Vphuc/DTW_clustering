from .base import DeepClusterer, _get_torch


# ---------------------------------------------------------------------------
# Standalone helpers (called at runtime, not at import)
# ---------------------------------------------------------------------------

def _soft_assign(z, centers):
    """
    Student's t soft assignment with α=1.
    z       : (B, d_z) tensor
    centers : (k, d_z) tensor
    returns   (B, k)
    """
    diff = z.unsqueeze(1) - centers.unsqueeze(0)    # (B, k, d_z)
    dist_sq = (diff ** 2).sum(2)                    # (B, k)
    num = (1.0 + dist_sq) ** -1.0
    return num / num.sum(1, keepdim=True)


def _target_dist(Q):
    """
    Auxiliary target distribution P from soft assignment Q.
    Q : (N, k) — full-dataset soft assignments (CPU or GPU)
    returns (N, k) same device
    """
    f = Q.sum(0, keepdim=True)       # (1, k) soft cluster frequencies
    P = Q ** 2 / f
    return P / P.sum(1, keepdim=True)


# ---------------------------------------------------------------------------
# DEC
# ---------------------------------------------------------------------------

class DEC(DeepClusterer):
    """
    Deep Embedded Clustering (Xie et al., 2016) for time-series.

    Training phases
    ---------------
    Phase 1 — Pretraining:
        autoencoder trained with Soft-DTW reconstruction loss.
        Call pretrain(X) explicitly, or load_weights(path), or let fit() do it.

    Phase 2 — Clustering:
        Decoder discarded. Encoder + cluster centres μ_j optimised jointly
        by minimising KL(P ∥ Q) where Q is Student's t soft assignment and
        P is the auxiliary target distribution recomputed every update_interval
        mini-batch iterations.

    Parameters
    ----------
    n_clusters      : int
    autoencoder     : TimeSeriesAutoencoder (nn.Module)
    update_interval : int   Mini-batch iterations between P refreshes
    tol             : float Converge when label-change fraction < tol
    device          : str or None   Auto-detects CUDA if None
    random_state    : int or None
    """

    def __init__(
        self,
        n_clusters,
        autoencoder,
        update_interval=140,
        tol=0.001,
        device=None,
        random_state=None,
    ):
        super().__init__(n_clusters, autoencoder, device, random_state)
        self.update_interval = update_interval
        self.tol = tol
        self._clustering_layer = None   # built lazily in fit()
        self.cluster_centers_ = None

    # ------------------------------------------------------------------
    # Soft assignment helpers (use self.device)
    # ------------------------------------------------------------------

    def _q_full(self, X_t, batch_size):
        """Compute Q = soft_assign for all samples. Returns CPU tensor (N, k)."""
        torch = _get_torch()
        centers_cpu = self._clustering_layer.centers.detach().cpu()
        parts = []
        self.autoencoder.eval()
        with torch.no_grad():
            for s in range(0, len(X_t), batch_size):
                z = self.autoencoder.encode(X_t[s: s + batch_size]).cpu()
                q = _soft_assign(z, centers_cpu)
                parts.append(q)
        self.autoencoder.train()
        return torch.cat(parts, dim=0)   # (N, k) CPU

    # ------------------------------------------------------------------
    # fit
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
        epochs          : int   Clustering fine-tuning epochs
        lr              : float Fine-tuning learning rate (Adam)
        batch_size      : int
        pretrain_epochs : int   Used only if not yet pretrained
        pretrain_lr     : float
        sdtw_gamma      : float Soft-DTW gamma used during pretraining
        verbose         : bool
        """
        torch = _get_torch()
        import torch.nn as nn
        import torch.nn.functional as F
        from sklearn.cluster import KMeans

        # ---- Phase 1: pretrain if needed ----
        if not self._pretrained:
            self.pretrain(
                X, epochs=pretrain_epochs, lr=pretrain_lr,
                sdtw_gamma=sdtw_gamma, verbose=verbose,
            )

        X_t = self._to_tensor(X)
        N = len(X_t)
        latent_dim = self.autoencoder.latent_dim

        # ---- Build learnable cluster-centre layer ----
        class _CentersModule(nn.Module):
            def __init__(self, k, d):
                super().__init__()
                self.centers = nn.Parameter(torch.zeros(k, d))

        self._clustering_layer = _CentersModule(self.n_clusters, latent_dim).to(self.device)

        # ---- K-Means initialisation ----
        Z_cpu = self._get_embeddings(X_t, batch_size).numpy()
        km = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=self.random_state)
        km.fit(Z_cpu)
        centers_init = torch.tensor(km.cluster_centers_, dtype=torch.float32)
        self._clustering_layer.centers.data.copy_(centers_init)
        prev_labels = km.labels_.copy()
        if verbose:
            print(f"[DEC] K-Means init done. Fine-tuning for up to {epochs} epochs.")

        # ---- Optimiser: encoder only (no decoder) + cluster centres ----
        enc_params = (
            list(self.autoencoder.encoder_conv.parameters())
            + list(self.autoencoder.encoder_fc.parameters())
        )
        optimizer = torch.optim.Adam(
            enc_params + list(self._clustering_layer.parameters()), lr=lr
        )

        # ---- Training loop ----
        iteration = 0
        P_device = None     # (N, k) on self.device, refreshed every update_interval iters
        rng = self._random_state()
        converged = False

        for _ in range(epochs):
            perm = rng.permutation(N)

            for start in range(0, N, batch_size):
                # -- Refresh P on full dataset --
                if iteration % self.update_interval == 0:
                    Q_cpu = self._q_full(X_t, batch_size)           # (N, k) CPU
                    P_device = _target_dist(Q_cpu).to(self.device)  # (N, k) GPU/CPU

                    new_labels = Q_cpu.argmax(1).numpy()
                    delta = float((new_labels != prev_labels).sum()) / N
                    if verbose and iteration > 0:
                        print(f"[DEC] iter {iteration:5d}  label-change={delta:.4f}")
                    if iteration > 0 and delta < self.tol:
                        if verbose:
                            print("[DEC] Converged.")
                        converged = True
                        break
                    prev_labels = new_labels.copy()

                # -- Mini-batch KL update --
                idx = perm[start: start + batch_size]
                x_b = X_t[idx]
                z_b = self.autoencoder.encode(x_b)                  # (B, d_z)
                Q_b = _soft_assign(z_b, self._clustering_layer.centers)  # (B, k)
                P_b = P_device[idx]                                  # (B, k)

                # KL(P ∥ Q): F.kl_div expects log-probabilities as first arg
                loss = F.kl_div(Q_b.log(), P_b, reduction="batchmean")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iteration += 1

            if converged:
                break

        # ---- Final assignment ----
        Q_final = self._q_full(X_t, batch_size)     # (N, k) CPU
        self.labels_ = Q_final.argmax(1).numpy().astype(int)
        self.cluster_centers_ = self._clustering_layer.centers.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, X):
        """Assign each sample to nearest cluster centre (argmax of Q)."""
        if self._clustering_layer is None:
            raise RuntimeError("Call fit() before predict().")
        X_t = self._to_tensor(X)
        Q = self._q_full(X_t, batch_size=256)
        return Q.argmax(1).numpy().astype(int)
