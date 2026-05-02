import numpy as np

from ...distances.soft_dtw import soft_dtw_distance_matrix
from .autoencoder import SoftDTWReconstructionLoss, TimeSeriesAutoencoder  # noqa: F401


def _get_torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for deep clustering models. "
            "Install with: pip install torch"
        ) from e


class DeepClusterer:
    """
    Base class for all deep clustering models.

    Handles autoencoder pretraining, weight persistence,
    data/device utilities, and defines the fit/predict interface.

    Parameters
    ----------
    n_clusters : int
    autoencoder : TimeSeriesAutoencoder (nn.Module)
        Shared Conv1D autoencoder backbone.
    device : str or None
        "cuda", "cpu", or None (auto-detect CUDA).
    random_state : int or None
    """

    def __init__(self, n_clusters, autoencoder, device=None, random_state=None):
        torch = _get_torch()
        self.n_clusters = n_clusters
        self.autoencoder = autoencoder
        self.random_state = random_state
        self.labels_ = None
        self._pretrained = False

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.autoencoder.to(self.device)

    # ------------------------------------------------------------------
    # Pretraining
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
        Joint pretrain: siamese metric loss + Soft-DTW reconstruction loss.

        The encoder is pushed so that ||z_i - z_j||_2 in the latent space
        approximates the Soft-DTW distance between the corresponding inputs,
        while the decoder is kept usable via the reconstruction term.

        Parameters
        ----------
        X                   : ndarray (N, d, T)
        epochs              : int
        lr                  : float
        batch_size          : int
        sdtw_gamma          : float   γ for the reconstruction Soft-DTW loss
        metric_target_gamma : float   γ used to compute the N×N target matrix
        lambda_metric       : float   weight of the metric loss
        lambda_recon        : float   weight of the reconstruction loss
        pairs_per_step      : int or None
            Number of pairs sampled per mini-batch. Defaults to 4 * batch_size.
        target_matrix       : ndarray (N, N) or None
            Precomputed Soft-DTW distance matrix. If None, it is computed
            internally via ``soft_dtw_distance_matrix``.
        n_jobs              : int     Workers for the target matrix computation.
        verbose             : bool
        """
        torch = _get_torch()
        X_t = self._to_tensor(X)
        N = len(X_t)
        if pairs_per_step is None:
            pairs_per_step = 4 * batch_size

        # ---- Precompute target matrix (CPU, numpy) ----
        if target_matrix is None:
            if verbose:
                print(
                    f"[Pretrain] Computing N×N Soft-DTW target matrix "
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

        # ---- Optimiser & losses ----
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

                # ---- sample pairs (local indices into b_global) ----
                p_i = rng.integers(0, B, size=pairs_per_step)
                p_j = rng.integers(0, B, size=pairs_per_step)
                same = p_i == p_j
                if same.any():
                    p_j[same] = (p_j[same] + 1) % B

                t = D[b_global[p_i], b_global[p_j]]
                # Per-batch normalisation (matches user's snippet)
                t_mean, t_std = float(t.mean()), float(t.std() + 1e-6)
                t = (t - t_mean) / t_std
                t_min = float(t.min())
                if t_min < 0:
                    t = t + abs(t_min)
                t = torch.from_numpy(t.astype(np.float32)).to(self.device)

                # ---- forward (separate passes to avoid autograd version
                # conflicts between Soft-DTW inplace ops and the metric path) ----
                batch = X_t[b_global]
                optimizer.zero_grad()

                # Metric path
                z = self.autoencoder.encode(batch)
                d_pred = torch.norm(z[p_i] - z[p_j], dim=-1)
                L_metric = torch.nn.functional.mse_loss(d_pred, t)

                # Reconstruction path (independent forward)
                _, x_hat = self.autoencoder(batch)
                L_recon = recon_loss_fn(batch, x_hat)

                loss = lambda_metric * L_metric + lambda_recon * L_recon
                loss.backward()
                optimizer.step()

                ep_metric += L_metric.item()
                ep_recon += L_recon.item()
                ep_total += loss.item()
                nb += 1

            if verbose and (epoch + 1) % log_every == 0:
                print(
                    f"[Pretrain] epoch {epoch + 1}/{epochs}  "
                    f"total={ep_total / max(1, nb):.4f}  "
                    f"metric={ep_metric / max(1, nb):.4f}  "
                    f"recon={ep_recon / max(1, nb):.4f}"
                )

        self._pretrained = True

    # ------------------------------------------------------------------
    # Weight persistence
    # ------------------------------------------------------------------

    def save_weights(self, path):
        """Save autoencoder state_dict to path."""
        torch = _get_torch()
        torch.save(self.autoencoder.state_dict(), path)

    def load_weights(self, path):
        """Load autoencoder state_dict from path (marks model as pretrained)."""
        torch = _get_torch()
        state = torch.load(path, map_location=self.device)
        self.autoencoder.load_state_dict(state)
        self._pretrained = True

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _to_tensor(self, X):
        """numpy (N, d, T) or torch Tensor → float32 tensor on self.device."""
        torch = _get_torch()
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32))
        return X.to(self.device)

    def _get_embeddings(self, X, batch_size=256):
        """
        Pass X through encoder, return embeddings as torch Tensor (N, latent_dim) on CPU.
        Sets autoencoder to eval mode; restores train mode afterwards.
        """
        torch = _get_torch()
        X_t = self._to_tensor(X)
        self.autoencoder.eval()
        parts = []
        with torch.no_grad():
            for start in range(0, len(X_t), batch_size):
                z = self.autoencoder.encode(X_t[start: start + batch_size])
                parts.append(z.cpu())
        self.autoencoder.train()
        return torch.cat(parts, dim=0)

    def _random_state(self):
        return np.random.default_rng(self.random_state)

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def fit(self, X, **kwargs):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def fit_predict(self, X, **kwargs):
        self.fit(X, **kwargs)
        return self.labels_
