from copy import deepcopy

import numpy as np

from .autoencoder import SoftDTWReconstructionLoss
from .base import DeepClusterer, _get_torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _per_sample_sdtw(x, x_hat, loss_module, sdtw_xx=None):
    """
    Per-sample Soft-DTW reconstruction value, shape (B,).

    Exposes the per-sample values that SoftDTWReconstructionLoss hides behind
    its .mean() reduction. If ``sdtw_xx`` (the self-distance of x) is given,
    it is reused instead of being recomputed.
    """
    D = loss_module._pairwise_sq_dist(x, x_hat)
    val = loss_module._soft_dtw_value(D)
    if loss_module.normalize:
        if sdtw_xx is None:
            sdtw_xx = loss_module._soft_dtw_value(loss_module._pairwise_sq_dist(x, x))
        D_hh = loss_module._pairwise_sq_dist(x_hat, x_hat)
        sdtw_hh = loss_module._soft_dtw_value(D_hh)
        val = val - 0.5 * (sdtw_xx + sdtw_hh)
    return val


def _build_damic_submodules():
    torch = _get_torch()
    import torch.nn as nn
    import torch.nn.functional as F

    class _MultiDecoder(nn.Module):
        """
        k independent decoders cloned from a shared TimeSeriesAutoencoder decoder.

        Each decoder owns its own decoder_fc + decoder_conv weights but the
        architecture is identical, so every decoder maps latent_dim → (d, T).
        """

        def __init__(self, n_clusters, source_ae):
            super().__init__()
            self.n_clusters = n_clusters
            self.decoder_fcs = nn.ModuleList(
                [deepcopy(source_ae.decoder_fc) for _ in range(n_clusters)]
            )
            self.decoder_convs = nn.ModuleList(
                [deepcopy(source_ae.decoder_conv) for _ in range(n_clusters)]
            )
            self.conv_channels_last = source_ae.conv_channels[-1]
            self.t_enc = source_ae._t_enc
            self.seq_len = source_ae.seq_len

        def decode(self, z, k):
            h = self.decoder_fcs[k](z)
            h = h.view(h.size(0), self.conv_channels_last, self.t_enc)
            x_hat = self.decoder_convs[k](h)
            if x_hat.size(2) != self.seq_len:
                x_hat = x_hat[:, :, : self.seq_len]
            return x_hat

    class _ClusterNet(nn.Module):
        """Routing MLP: latent → softmax over k clusters."""

        def __init__(self, latent_dim, n_clusters, hidden=(128, 64)):
            super().__init__()
            layers = []
            prev = latent_dim
            for h in hidden:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU(inplace=True))
                prev = h
            layers.append(nn.Linear(prev, n_clusters))
            self.net = nn.Sequential(*layers)

        def forward(self, z):
            logits = self.net(z)
            probs = F.softmax(logits, dim=1)
            return logits, probs

    return _MultiDecoder, _ClusterNet


# ---------------------------------------------------------------------------
# DAMIC
# ---------------------------------------------------------------------------

class DAMIC(DeepClusterer):
    """
    Deep Autoencoding Mixture Model for Clustering (DAMIC) for time-series,
    with a shared-encoder + k-decoders backbone.

    Architecture
    ------------
        encoder  : shared (from TimeSeriesAutoencoder)
        decoders : k copies (one per cluster), cloned from the pretrained decoder
        router   : ClusterNet MLP  z → p(c | z)

    The mixture-of-experts likelihood is

        L = − Σᵢ log( Σₖ p_ik · exp(−sdtw(xᵢ, dec_k(enc(xᵢ)))) )

    computed in log-sum-exp form for numerical stability.

    Training pipeline
    -----------------
    Phase 1 — Pretrain shared autoencoder (Soft-DTW reconstruction).
    Phase 2 — K-Means on embeddings → initial labels ``labels_init``.
    Phase 3 — Clone decoder k times; specialise each decoder on its cluster's
              data (encoder frozen, reconstruction-only loss).
    Phase 4a — CE warm-up of ClusterNet using ``labels_init`` as soft targets.
    Phase 4b — Joint end-to-end training: encoder + k decoders + ClusterNet,
               optimised with the mixture loss above.

    Inference
    ---------
        label(x) = argmaxₖ [ log p_k(x) − sdtw(x, dec_k(enc(x))) ]

    Parameters
    ----------
    n_clusters          : int
    autoencoder         : TimeSeriesAutoencoder (nn.Module)
    cluster_net_hidden  : tuple of int  Hidden layer sizes of ClusterNet
    ce_warmup_epochs    : int           Epochs for Phase 4a (0 disables it)
    device              : str or None
    random_state        : int or None
    """

    def __init__(
        self,
        n_clusters,
        autoencoder,
        cluster_net_hidden=(128, 64),
        ce_warmup_epochs=5,
        device=None,
        random_state=None,
    ):
        super().__init__(n_clusters, autoencoder, device, random_state)
        self.cluster_net_hidden = cluster_net_hidden
        self.ce_warmup_epochs = ce_warmup_epochs

        self._multi_decoder = None
        self._cluster_net = None
        self._recon_loss_fn = None    # kept for reuse in predict()
        self.cluster_centers_ = None  # DAMIC has no explicit centres in Z

    # ------------------------------------------------------------------
    # Encoder freeze/unfreeze helpers
    # ------------------------------------------------------------------

    def _set_encoder_grad(self, requires_grad):
        for p in self.autoencoder.encoder_conv.parameters():
            p.requires_grad = requires_grad
        for p in self.autoencoder.encoder_fc.parameters():
            p.requires_grad = requires_grad

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X,
        epochs=100,
        lr=1e-3,
        batch_size=256,
        pretrain_epochs=100,
        pretrain_lr=1e-3,
        cluster_ae_epochs=50,
        cluster_ae_lr=1e-3,
        ce_warmup_lr=1e-3,
        sdtw_gamma=0.1,
        sdtw_gamma_finetune=0.1,
        verbose=True,
    ):
        """
        Parameters
        ----------
        X                    : ndarray (N, d, T)
        epochs               : int   Joint-training epochs (Phase 4b)
        lr                   : float Joint-training learning rate
        batch_size           : int
        pretrain_epochs      : int   Used only if not yet pretrained
        pretrain_lr          : float
        cluster_ae_epochs    : int   Epochs per cluster decoder (Phase 3)
        cluster_ae_lr        : float Learning rate in Phase 3
        ce_warmup_lr         : float Learning rate in Phase 4a
        sdtw_gamma           : float γ for pretraining reconstruction
        sdtw_gamma_finetune  : float γ for reconstruction in Phases 3 and 4b
        verbose              : bool
        """
        torch = _get_torch()
        import torch.nn.functional as F
        from sklearn.cluster import KMeans

        _MultiDecoder, _ClusterNet = _build_damic_submodules()

        # ========== Phase 1: pretrain shared AE ==========
        if not self._pretrained:
            self.pretrain(
                X, epochs=pretrain_epochs, lr=pretrain_lr,
                sdtw_gamma=sdtw_gamma, verbose=verbose,
            )

        X_t = self._to_tensor(X)
        N = len(X_t)
        latent_dim = self.autoencoder.latent_dim

        # Reconstruction loss used throughout Phases 3 and 4b
        recon_loss_fn = SoftDTWReconstructionLoss(
            gamma=sdtw_gamma_finetune, normalize=True
        )
        recon_loss_fn.to(self.device)
        self._recon_loss_fn = recon_loss_fn

        # ========== Phase 2: K-Means init ==========
        if verbose:
            print(f"[DAMIC] Phase 2: K-Means on embedding space (N={N}).")
        Z_cpu = self._get_embeddings(X_t, batch_size).numpy()
        km = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=self.random_state)
        km.fit(Z_cpu)
        labels_init = km.labels_.astype(np.int64)
        if verbose:
            sizes = {int(k): int((labels_init == k).sum()) for k in range(self.n_clusters)}
            print(f"[DAMIC] Cluster sizes from K-Means: {sizes}")

        # ========== Phase 3: clone + specialise k decoders ==========
        self._multi_decoder = _MultiDecoder(self.n_clusters, self.autoencoder).to(self.device)
        self._set_encoder_grad(False)      # encoder frozen
        self.autoencoder.eval()
        self._multi_decoder.train()

        rng = self._random_state()

        if verbose:
            print(
                f"[DAMIC] Phase 3: specialising {self.n_clusters} decoders "
                f"({cluster_ae_epochs} epochs each)."
            )

        for k in range(self.n_clusters):
            idx_k = np.where(labels_init == k)[0]
            if len(idx_k) < 2:
                if verbose:
                    print(f"[DAMIC]   cluster {k}: only {len(idx_k)} sample(s) — skipping.")
                continue

            X_k = X_t[idx_k]
            decoder_params = (
                list(self._multi_decoder.decoder_fcs[k].parameters())
                + list(self._multi_decoder.decoder_convs[k].parameters())
            )
            opt_k = torch.optim.Adam(decoder_params, lr=cluster_ae_lr)

            for epoch in range(cluster_ae_epochs):
                perm_k = rng.permutation(len(idx_k))
                ep_loss, nb = 0.0, 0
                for start in range(0, len(idx_k), batch_size):
                    x_b = X_k[perm_k[start: start + batch_size]]
                    with torch.no_grad():
                        z = self.autoencoder.encode(x_b)
                    x_hat = self._multi_decoder.decode(z, k)
                    loss = recon_loss_fn(x_b, x_hat)

                    opt_k.zero_grad()
                    loss.backward()
                    opt_k.step()
                    ep_loss += loss.item()
                    nb += 1

                if verbose and (epoch + 1) % max(1, cluster_ae_epochs // 5) == 0:
                    print(
                        f"[DAMIC]   cluster {k} epoch {epoch + 1}/{cluster_ae_epochs}  "
                        f"loss={ep_loss / max(1, nb):.4f}"
                    )

        # ========== Phase 4a: CE warm-up of ClusterNet ==========
        self._cluster_net = _ClusterNet(
            latent_dim, self.n_clusters, self.cluster_net_hidden
        ).to(self.device)

        if self.ce_warmup_epochs > 0:
            if verbose:
                print(f"[DAMIC] Phase 4a: ClusterNet CE warm-up ({self.ce_warmup_epochs} epochs).")
            y = torch.from_numpy(labels_init).to(self.device)
            opt_cn = torch.optim.Adam(self._cluster_net.parameters(), lr=ce_warmup_lr)
            self.autoencoder.eval()
            self._cluster_net.train()

            for epoch in range(self.ce_warmup_epochs):
                perm = rng.permutation(N)
                ep_loss, nb = 0.0, 0
                for start in range(0, N, batch_size):
                    idx = perm[start: start + batch_size]
                    with torch.no_grad():
                        z = self.autoencoder.encode(X_t[idx])
                    logits, _ = self._cluster_net(z)
                    loss = F.cross_entropy(logits, y[idx])

                    opt_cn.zero_grad()
                    loss.backward()
                    opt_cn.step()
                    ep_loss += loss.item()
                    nb += 1
                if verbose:
                    print(
                        f"[DAMIC]   CE warm-up epoch {epoch + 1}/{self.ce_warmup_epochs}  "
                        f"loss={ep_loss / max(1, nb):.4f}"
                    )

        # ========== Phase 4b: joint mixture training ==========
        self._set_encoder_grad(True)     # unfreeze encoder
        self.autoencoder.train()
        self._multi_decoder.train()
        self._cluster_net.train()

        joint_params = (
            list(self.autoencoder.encoder_conv.parameters())
            + list(self.autoencoder.encoder_fc.parameters())
            + list(self._multi_decoder.parameters())
            + list(self._cluster_net.parameters())
        )
        opt_joint = torch.optim.Adam(joint_params, lr=lr)

        if verbose:
            print(f"[DAMIC] Phase 4b: joint mixture training ({epochs} epochs).")

        log_every = max(1, epochs // 10)
        for epoch in range(epochs):
            perm = rng.permutation(N)
            ep_loss, nb = 0.0, 0
            for start in range(0, N, batch_size):
                x_b = X_t[perm[start: start + batch_size]]

                z = self.autoencoder.encode(x_b)                   # (B, d_z)
                _, probs = self._cluster_net(z)                    # (B, k)
                log_probs = torch.log(probs + 1e-10)

                # sdtw(x, x) is identical across all k decoders — compute once
                sdtw_xx = recon_loss_fn._soft_dtw_value(
                    recon_loss_fn._pairwise_sq_dist(x_b, x_b)
                )                                                   # (B,)

                neg_sdtws = []
                for k in range(self.n_clusters):
                    x_hat_k = self._multi_decoder.decode(z, k)
                    sdtw_k = _per_sample_sdtw(x_b, x_hat_k, recon_loss_fn, sdtw_xx=sdtw_xx)
                    neg_sdtws.append(-sdtw_k)
                neg_sdtws = torch.stack(neg_sdtws, dim=1)          # (B, k)

                # Numerically stable mixture NLL
                loss = -torch.logsumexp(neg_sdtws + log_probs, dim=1).mean()

                opt_joint.zero_grad()
                loss.backward()
                opt_joint.step()
                ep_loss += loss.item()
                nb += 1

            if verbose and (epoch + 1) % log_every == 0:
                print(
                    f"[DAMIC]   joint epoch {epoch + 1}/{epochs}  "
                    f"loss={ep_loss / max(1, nb):.4f}"
                )

        # ========== Final inference ==========
        self.labels_ = self._infer_labels(X_t, batch_size)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer_labels(self, X_t, batch_size):
        """Argmax of log p_k − sdtw_k (equivalently p_k · exp(−sdtw_k))."""
        torch = _get_torch()
        self.autoencoder.eval()
        self._multi_decoder.eval()
        self._cluster_net.eval()

        loss_fn = self._recon_loss_fn
        parts = []
        with torch.no_grad():
            for start in range(0, len(X_t), batch_size):
                x_b = X_t[start: start + batch_size]
                z = self.autoencoder.encode(x_b)
                _, probs = self._cluster_net(z)
                log_probs = torch.log(probs + 1e-10)

                sdtw_xx = loss_fn._soft_dtw_value(loss_fn._pairwise_sq_dist(x_b, x_b))
                neg_sdtws = []
                for k in range(self.n_clusters):
                    x_hat_k = self._multi_decoder.decode(z, k)
                    sdtw_k = _per_sample_sdtw(x_b, x_hat_k, loss_fn, sdtw_xx=sdtw_xx)
                    neg_sdtws.append(-sdtw_k)
                neg_sdtws = torch.stack(neg_sdtws, dim=1)

                scores = neg_sdtws + log_probs                      # (B, k)
                parts.append(scores.argmax(dim=1).cpu().numpy())

        return np.concatenate(parts).astype(int)

    def predict(self, X):
        """Assign new samples to their most likely decoder-cluster."""
        if self._multi_decoder is None or self._cluster_net is None:
            raise RuntimeError("Call fit() before predict().")
        X_t = self._to_tensor(X)
        return self._infer_labels(X_t, batch_size=256)
