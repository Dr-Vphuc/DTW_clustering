from .autoencoder import SoftDTWReconstructionLoss
from .base import _get_torch
from .dec import DEC, _soft_assign, _target_dist


class IDEC(DEC):
    """
    Improved Deep Embedded Clustering (Guo et al., 2017) for time-series.

    Differs from DEC in one key way: the decoder is **kept** during fine-tuning.
    The joint loss is:

        L = L_r + γ · L_c

    where L_r is Soft-DTW reconstruction loss and L_c = KL(P ∥ Q).
    Keeping the decoder prevents the encoder from distorting the embedding space
    while the clustering loss pushes points toward cluster centres.

    Everything else (pretraining, K-Means init, P refresh, convergence) is
    identical to DEC, so IDEC inherits and overrides only fit().

    Parameters
    ----------
    n_clusters      : int
    autoencoder     : TimeSeriesAutoencoder (nn.Module)
    gamma           : float  Weight of clustering loss (paper suggests < 1)
    sdtw_gamma      : float  Soft-DTW γ used in reconstruction loss during fine-tuning
    update_interval : int
    tol             : float
    device          : str or None
    random_state    : int or None
    """

    def __init__(
        self,
        n_clusters,
        autoencoder,
        gamma=0.1,
        sdtw_gamma_finetune=0.1,
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
        self.gamma = gamma
        self.sdtw_gamma_finetune = sdtw_gamma_finetune

    # ------------------------------------------------------------------
    # fit  (overrides DEC.fit)
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
        sdtw_gamma      : float Soft-DTW γ for pretraining loss
        verbose         : bool
        """
        torch = _get_torch()
        import torch.nn as nn
        import torch.nn.functional as F
        from sklearn.cluster import KMeans

        # ---- Phase 1: pretrain ----
        if not self._pretrained:
            self.pretrain(
                X, epochs=pretrain_epochs, lr=pretrain_lr,
                sdtw_gamma=sdtw_gamma, verbose=verbose,
            )

        X_t = self._to_tensor(X)
        N = len(X_t)
        latent_dim = self.autoencoder.latent_dim

        # ---- Build clustering layer (same as DEC) ----
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
            print(f"[IDEC] K-Means init done. Fine-tuning for up to {epochs} epochs.")

        # ---- Reconstruction loss (used throughout fine-tuning) ----
        recon_loss_fn = SoftDTWReconstructionLoss(
            gamma=self.sdtw_gamma_finetune, normalize=True
        )
        recon_loss_fn.to(self.device)

        # ---- Optimiser: ALL autoencoder params (encoder + decoder) + centres ----
        optimizer = torch.optim.Adam(
            list(self.autoencoder.parameters())
            + list(self._clustering_layer.parameters()),
            lr=lr,
        )

        # ---- Training loop ----
        iteration = 0
        P_device = None
        rng = self._random_state()
        converged = False

        for _ in range(epochs):
            perm = rng.permutation(N)

            for start in range(0, N, batch_size):
                # -- Refresh P on full dataset --
                if iteration % self.update_interval == 0:
                    Q_cpu = self._q_full(X_t, batch_size)
                    P_device = _target_dist(Q_cpu).to(self.device)

                    new_labels = Q_cpu.argmax(1).numpy()
                    delta = float((new_labels != prev_labels).sum()) / N
                    if verbose and iteration > 0:
                        print(f"[IDEC] iter {iteration:5d}  label-change={delta:.4f}")
                    if iteration > 0 and delta < self.tol:
                        if verbose:
                            print("[IDEC] Converged.")
                        converged = True
                        break
                    prev_labels = new_labels.copy()

                # -- Mini-batch joint update --
                idx = perm[start: start + batch_size]
                x_b = X_t[idx]

                # Full forward pass (encoder + decoder kept)
                z_b, x_hat_b = self.autoencoder(x_b)

                # Clustering loss
                Q_b = _soft_assign(z_b, self._clustering_layer.centers)
                P_b = P_device[idx]
                L_c = F.kl_div(Q_b.log(), P_b, reduction="batchmean")

                # Reconstruction loss
                L_r = recon_loss_fn(x_b, x_hat_b)

                loss = L_r + self.gamma * L_c

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
