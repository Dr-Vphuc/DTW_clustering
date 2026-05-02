_TORCH_MODULE = None


def _get_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for deep clustering models. "
            "Install with: pip install torch"
        ) from e


def _build_autoencoder_class():
    torch, nn = _get_torch()

    class _TimeSeriesAutoencoder(nn.Module):
        """
        Conv1D autoencoder backbone shared by all deep clustering models.

        Input shape : (B, d, T)  — d = variables, T = sequence length
        Latent shape: (B, latent_dim)

        Encoder: Conv1D(s=1) → Conv1D(s=2) → Conv1D(s=2) → Linear
        Decoder: Linear → ConvTranspose1D(s=2) → ConvTranspose1D(s=2) → ConvTranspose1D(s=1)
        Each Conv/ConvTranspose is followed by BatchNorm + ReLU except the final decoder output.
        """

        def __init__(
            self,
            input_channels,
            seq_len,
            latent_dim=32,
            conv_channels=None,
            kernel_size=3,
            dropout_rate=0.2,
        ):
            super().__init__()
            if conv_channels is None:
                conv_channels = [32, 64, 128]

            self.input_channels = input_channels
            self.seq_len = seq_len
            self.latent_dim = latent_dim
            self.conv_channels = conv_channels
            self.kernel_size = kernel_size
            self.dropout_rate = dropout_rate

            # Sequence length after stride-2 layers (one per non-first channel)
            t_enc = seq_len
            for _ in range(len(conv_channels) - 1):
                t_enc = (t_enc + 1) // 2
            self._t_enc = t_enc
            self._flat_dim = conv_channels[-1] * t_enc

            # ---- Encoder ----
            enc_layers = []
            in_ch = input_channels
            for idx, out_ch in enumerate(conv_channels):
                stride = 1 if idx == 0 else 2
                enc_layers += [
                    nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                ]
                in_ch = out_ch
            enc_layers.append(nn.Dropout(dropout_rate))
            self.encoder_conv = nn.Sequential(*enc_layers)
            self.encoder_fc = nn.Linear(self._flat_dim, latent_dim)

            # ---- Decoder ----
            self.decoder_fc = nn.Linear(latent_dim, self._flat_dim)
            dec_layers = []
            rev = list(reversed(conv_channels))
            for idx in range(len(rev) - 1):
                dec_layers += [
                    nn.ConvTranspose1d(
                        rev[idx], rev[idx + 1], kernel_size,
                        stride=2, padding=kernel_size // 2, output_padding=1,
                    ),
                    nn.BatchNorm1d(rev[idx + 1]),
                    nn.ReLU(inplace=True),
                ]
            dec_layers.append(
                nn.ConvTranspose1d(
                    rev[-1], input_channels, kernel_size,
                    stride=1, padding=kernel_size // 2,
                )
            )
            self.decoder_conv = nn.Sequential(*dec_layers)

        def encode(self, x):
            """(B, d, T) → (B, latent_dim)"""
            h = self.encoder_conv(x)
            h = h.view(h.size(0), -1)
            return self.encoder_fc(h)

        def decode(self, z):
            """(B, latent_dim) → (B, d, T)"""
            h = self.decoder_fc(z)
            h = h.view(h.size(0), self.conv_channels[-1], self._t_enc)
            x_hat = self.decoder_conv(h)
            # Trim to original seq_len if stride arithmetic adds an extra step
            if x_hat.size(2) != self.seq_len:
                x_hat = x_hat[:, :, : self.seq_len]
            return x_hat

        def forward(self, x):
            """Returns (z, x_hat)."""
            z = self.encode(x)
            return z, self.decode(z)

    # ---- Soft-DTW reconstruction loss ----------------------------------------

    class _SoftDTWLoss(nn.Module):
        """
        Soft-DTW loss between two batches of sequences.

        Inputs: x, x_hat — both (B, d, T)
        Returns: scalar mean loss over the batch.

        gamma    : smoothness (small → approaches hard DTW)
        normalize: subtract 0.5*(sdtw(x,x) + sdtw(x_hat,x_hat)) to make loss ≥ 0
        """

        def __init__(self, gamma=0.1, normalize=True):
            super().__init__()
            self.gamma = gamma
            self.normalize = normalize

        @staticmethod
        def _pairwise_sq_dist(x, y):
            """x: (B,d,T1), y: (B,d,T2) → (B,T1,T2) squared Euclidean local cost."""
            xx = (x * x).sum(1)                         # (B, T1)
            yy = (y * y).sum(1)                         # (B, T2)
            xy = torch.bmm(x.permute(0, 2, 1), y)       # (B, T1, T2)
            return (xx.unsqueeze(2) + yy.unsqueeze(1) - 2 * xy).clamp(min=0.0)

        def _soft_dtw_value(self, D):
            """D: (B, T1, T2) → per-sample Soft-DTW values (B,).

            Uses a 2-D Python list of tensors (no inplace tensor modification)
            so that autograd's view/version tracking remains valid even when
            the result is combined with other autograd graphs (e.g. a siamese
            metric loss sharing the same encoder weights).
            """
            B, T1, T2 = D.shape
            gamma = self.gamma
            inf_row = torch.full((B,), float("inf"), dtype=D.dtype, device=D.device)
            zero_row = torch.zeros((B,), dtype=D.dtype, device=D.device)

            R = [[None] * (T2 + 1) for _ in range(T1 + 1)]
            R[0][0] = zero_row
            for j in range(1, T2 + 1):
                R[0][j] = inf_row
            for i in range(1, T1 + 1):
                R[i][0] = inf_row

            for i in range(1, T1 + 1):
                for j in range(1, T2 + 1):
                    r0 = R[i - 1][j]
                    r1 = R[i][j - 1]
                    r2 = R[i - 1][j - 1]
                    rmin = torch.minimum(torch.minimum(r0, r1), r2)
                    softmin = rmin - gamma * torch.log(
                        torch.exp(-(r0 - rmin) / gamma)
                        + torch.exp(-(r1 - rmin) / gamma)
                        + torch.exp(-(r2 - rmin) / gamma)
                    )
                    R[i][j] = D[:, i - 1, j - 1] + softmin
            return R[T1][T2]

        def forward(self, x, x_hat):
            loss = self._soft_dtw_value(self._pairwise_sq_dist(x, x_hat))
            if self.normalize:
                loss = loss - 0.5 * (
                    self._soft_dtw_value(self._pairwise_sq_dist(x, x))
                    + self._soft_dtw_value(self._pairwise_sq_dist(x_hat, x_hat))
                )
            return loss.mean()

    return _TimeSeriesAutoencoder, _SoftDTWLoss


def _ensure_built():
    global _TORCH_MODULE
    if _TORCH_MODULE is None:
        _TORCH_MODULE = _build_autoencoder_class()
    return _TORCH_MODULE


class TimeSeriesAutoencoder:
    """
    Lazy proxy — returns a PyTorch nn.Module instance on construction.

    Usage:
        ae = TimeSeriesAutoencoder(input_channels=3, seq_len=64, latent_dim=32)
        z, x_hat = ae(x)   # x: (B, 3, 64)
    """

    def __new__(cls, *args, **kwargs):
        AEClass, _ = _ensure_built()
        return AEClass(*args, **kwargs)


class SoftDTWReconstructionLoss:
    """
    Lazy proxy — returns a PyTorch nn.Module instance on construction.

    Usage:
        loss_fn = SoftDTWReconstructionLoss(gamma=0.1, normalize=True)
        loss = loss_fn(x, x_hat)
    """

    def __new__(cls, *args, **kwargs):
        _, LossClass = _ensure_built()
        return LossClass(*args, **kwargs)
