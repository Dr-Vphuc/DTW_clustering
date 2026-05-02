import warnings

import numpy as np

from .autoencoder import SoftDTWReconstructionLoss
from .base import DeepClusterer, _get_torch


class DSC(DeepClusterer):
    """
    Deep Subspace Clustering Networks (Ji et al., 2017) for time-series.

    Architecture
    ------------
        x ─► encoder ─► Z ─► [self-expressive layer: C] ─► CZ ─► decoder ─► x̂

    The self-expressive layer is a fully-connected linear layer whose weight
    is the N×N matrix C.  Under the subspace-clustering assumption,
    ``z_i ≈ Σ_j c_ij · z_j`` for samples in the same subspace, and nonzero
    entries of C indicate subspace membership.

    Loss
    ----
        L = L_r + λ₁ · ‖C‖_p  +  (λ₂ / 2) · ‖Z − CZ‖_F²

        L_r : Soft-DTW reconstruction between x and x̂
        ‖C‖_p : regularisation on C  (p ∈ {1, 2})
        constraint : diag(C) = 0, enforced after every step

    Full-batch training
    -------------------
    DSC must process the entire dataset in a single batch because C couples
    all samples.  For large N, this becomes expensive (C is N×N and stored
    as an nn.Parameter).  A warning is emitted when ``N > warn_large_n``.

    Post-processing
    ---------------
    After training, the symmetrised affinity ``A = (|C| + |Cᵀ|) / 2`` is fed
    to spectral clustering (sklearn) to produce the final hard labels.

    This model is **transductive**: ``predict`` on unseen samples is not
    supported because C is tied to the training set.

    Parameters
    ----------
    n_clusters       : int
    autoencoder      : TimeSeriesAutoencoder (nn.Module)
    lam1, lam2       : float  Weights of the two C-related loss terms
    reg_type         : {"l1", "l2"}  Norm used for ‖C‖_p
    warn_large_n     : int    Emit a warning if N exceeds this
    spectral_n_init  : int    n_init for the final spectral-clustering step
    device           : str or None
    random_state     : int or None
    """

    def __init__(
        self,
        n_clusters,
        autoencoder,
        lam1=1.0,
        lam2=1.0,
        reg_type="l2",
        warn_large_n=5000,
        spectral_n_init=20,
        device=None,
        random_state=None,
    ):
        super().__init__(n_clusters, autoencoder, device, random_state)
        if reg_type not in ("l1", "l2"):
            raise ValueError(f"reg_type must be 'l1' or 'l2', got {reg_type!r}")
        self.lam1 = lam1
        self.lam2 = lam2
        self.reg_type = reg_type
        self.warn_large_n = warn_large_n
        self.spectral_n_init = spectral_n_init

        self._self_expr = None        # built in fit()
        self.C_ = None                # final self-expressive matrix (numpy)
        self.affinity_matrix_ = None  # post-processed affinity used for spectral

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X,
        epochs=1000,
        lr=1e-3,
        pretrain_epochs=100,
        pretrain_lr=1e-3,
        sdtw_gamma=0.1,
        sdtw_gamma_finetune=0.1,
        verbose=True,
    ):
        """
        Parameters
        ----------
        X                    : ndarray (N, d, T)
        epochs               : int   Fine-tuning epochs (full-batch)
        lr                   : float Adam learning rate (paper uses 1e-3)
        pretrain_epochs      : int   Used only if not yet pretrained
        pretrain_lr          : float
        sdtw_gamma           : float γ for pretraining reconstruction
        sdtw_gamma_finetune  : float γ for reconstruction during DSC fine-tuning
        verbose              : bool
        """
        torch = _get_torch()
        import torch.nn as nn
        from sklearn.cluster import SpectralClustering

        # ---- Phase 1: pretrain (standard AE, no self-expressive layer yet) ----
        if not self._pretrained:
            self.pretrain(
                X, epochs=pretrain_epochs, lr=pretrain_lr,
                sdtw_gamma=sdtw_gamma, verbose=verbose,
            )

        X_t = self._to_tensor(X)
        N = len(X_t)

        if N > self.warn_large_n:
            warnings.warn(
                f"DSC uses a {N}×{N} self-expressive matrix "
                f"(~{N * N * 4 / 1e6:.1f} MB at fp32); training may be slow "
                f"or memory-intensive.",
                RuntimeWarning,
            )

        # ---- Build self-expressive layer C ∈ R^{N×N} ----
        class _SelfExpressive(nn.Module):
            def __init__(self, n, init_std):
                super().__init__()
                self.C = nn.Parameter(init_std * torch.randn(n, n))

            def forward(self, Z):
                return self.C @ Z  # (N, N) @ (N, d_z)

        self._self_expr = _SelfExpressive(N, init_std=1e-4).to(self.device)

        # ---- Optimiser: encoder + decoder + C ----
        optimizer = torch.optim.Adam(
            list(self.autoencoder.parameters())
            + list(self._self_expr.parameters()),
            lr=lr,
        )

        # ---- Reconstruction loss ----
        recon_loss_fn = SoftDTWReconstructionLoss(gamma=sdtw_gamma_finetune, normalize=True)
        recon_loss_fn.to(self.device)

        self.autoencoder.train()
        log_every = max(1, epochs // 20)

        for epoch in range(epochs):
            # Enforce diag(C) = 0 BEFORE forward so gradients don't see it
            with torch.no_grad():
                self._self_expr.C.fill_diagonal_(0.0)

            optimizer.zero_grad()

            # ---- Forward ----
            z = self.autoencoder.encode(X_t)           # (N, d_z)
            z_expr = self._self_expr(z)                 # (N, d_z) = C @ z
            x_hat = self.autoencoder.decode(z_expr)     # (N, d, T)

            # ---- Losses ----
            L_r = recon_loss_fn(X_t, x_hat)
            L_se = 0.5 * ((z - z_expr) ** 2).sum()
            if self.reg_type == "l1":
                L_c = self._self_expr.C.abs().sum()
            else:
                L_c = (self._self_expr.C ** 2).sum()

            loss = L_r + self.lam1 * L_c + self.lam2 * L_se

            loss.backward()
            optimizer.step()

            if verbose and (epoch + 1) % log_every == 0:
                print(
                    f"[DSC] epoch {epoch + 1}/{epochs}  "
                    f"L_r={L_r.item():.4f}  "
                    f"L_c={L_c.item():.4f}  "
                    f"L_se={L_se.item():.4f}"
                )

        # ---- Post-processing: affinity + spectral clustering ----
        with torch.no_grad():
            self._self_expr.C.fill_diagonal_(0.0)
            C_np = self._self_expr.C.detach().cpu().numpy()
        self.C_ = C_np
        self.affinity_matrix_ = 0.5 * (np.abs(C_np) + np.abs(C_np.T))

        spectral = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity="precomputed",
            n_init=self.spectral_n_init,
            random_state=self.random_state,
            assign_labels="kmeans",
        )
        self.labels_ = spectral.fit_predict(self.affinity_matrix_).astype(int)

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, X):  # noqa: ARG002  (transductive — X intentionally unused)
        """
        DSC is transductive — the self-expressive matrix C is tied to the
        training set and cannot be queried for new samples.
        """
        del X
        raise NotImplementedError(
            "DSC is transductive: C is fitted to the training set. "
            "Use fit_predict(X_all) to cluster all samples jointly."
        )
