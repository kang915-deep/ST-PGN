"""First runnable ST-PGN implementation.

The implementation is intentionally self-contained.  The downloaded UniTS,
PatchTST, FreDF and MTGNN repositories are kept as references; this module
does not modify or import their training code.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class STPGNConfig:
    in_channels: int
    seq_len: int = 30
    patch_len: int = 10
    stride: int = 5
    d_model: int = 64
    graph_dim: int = 16
    graph_layers: int = 2
    prior_dim: int = 64
    prior_hidden_dim: int = 128
    dft_keep_ratio: float = 0.5
    padding: str = "none"
    dropout: float = 0.1
    head_hidden: int = 128


class PatchEmbedding(nn.Module):
    """Channel-independent temporal patch embedding.

    Input:  [B, T, C]
    Output: [B, C, N, D]
    """

    def __init__(self, patch_len: int, stride: int, d_model: int,
                 padding: str = "none", dropout: float = 0.0):
        super().__init__()
        if patch_len <= 0 or stride <= 0:
            raise ValueError("patch_len and stride must be positive")
        if padding not in {"none", "replicate", "reflect"}:
            raise ValueError("padding must be none, replicate, or reflect")
        self.patch_len = patch_len
        self.stride = stride
        self.padding = padding
        self.proj = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def _pad(self, x: Tensor) -> Tensor:
        # x: [B, C, T]
        remainder = (x.shape[-1] - self.patch_len) % self.stride
        if remainder == 0:
            return x
        pad_right = self.stride - remainder
        if self.padding == "none":
            return x
        if self.padding == "replicate":
            return F.pad(x, (0, pad_right), mode="replicate")
        # reflect requires at least one existing value and a pad smaller than
        # the current length; fall back to replication for very short inputs.
        if pad_right < x.shape[-1]:
            return F.pad(x, (0, pad_right), mode="reflect")
        return F.pad(x, (0, pad_right), mode="replicate")

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [B,T,C], got {tuple(x.shape)}")
        x = x.transpose(1, 2).contiguous()  # [B,C,T]
        x = self._pad(x)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # [B,C,N,P] -> project P to D
        return self.dropout(self.proj(patches))


class DFTFilter(nn.Module):
    """Differentiable low-pass DFT filter over the patch-time axis.

    Input/output: [B, C, N, D].  The initial version uses a fixed mask so
    that the first experiments isolate the architecture from extra learned
    frequency parameters.
    """

    def __init__(self, keep_ratio: float = 0.5):
        super().__init__()
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")
        self.keep_ratio = keep_ratio

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected [B,C,N,D], got {tuple(x.shape)}")
        n = x.shape[2]
        spectrum = torch.fft.rfft(x, dim=2)
        keep = max(1, int((spectrum.shape[2]) * self.keep_ratio))
        mask = torch.zeros(spectrum.shape[2], device=x.device, dtype=x.dtype)
        mask[:keep] = 1.0
        spectrum = spectrum * mask.view(1, 1, -1, 1)
        return torch.fft.irfft(spectrum, n=n, dim=2)


class AdaptiveGraph(nn.Module):
    """Learnable directed graph interaction over sensor channels."""

    def __init__(self, channels: int, feature_dim: int, graph_dim: int,
                 layers: int, dropout: float):
        super().__init__()
        self.channels = channels
        self.graph_dim = graph_dim
        self.source = nn.Parameter(torch.randn(channels, graph_dim) * 0.02)
        self.target = nn.Parameter(torch.randn(graph_dim, channels) * 0.02)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ) for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(layers)])

    def adjacency(self) -> Tensor:
        logits = self.source @ self.target / (self.graph_dim ** 0.5)
        logits = F.relu(logits)
        return torch.softmax(logits, dim=-1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # x: [B,C,F], A[i,j] aggregates node j into node i.
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(f"expected [B,{self.channels},F], got {tuple(x.shape)}")
        a = self.adjacency()
        for block, norm in zip(self.blocks, self.norms):
            aggregated = torch.einsum("ij,bjf->bif", a, x)
            x = norm(x + block(aggregated))
        return x, a


class PriorProjector(nn.Module):
    """Pool and project UniTS intermediate features.

    Accepts [B,C,L,D_u] or [B,C,D_u].
    Returns [B,D_prior].
    """

    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, prior: Tensor) -> Tensor:
        if prior.ndim == 4:
            prior = prior.mean(dim=2)
        if prior.ndim != 3:
            raise ValueError("prior must have shape [B,C,D] or [B,C,L,D]")
        return self.proj(prior).mean(dim=1)


class STPGN(nn.Module):
    """ST-PGN v0: local DFT branch + adaptive graph + optional UniTS prior."""

    def __init__(self, config: STPGNConfig):
        super().__init__()
        self.config = config
        self.patch = PatchEmbedding(
            config.patch_len, config.stride, config.d_model,
            padding=config.padding, dropout=config.dropout)
        self.dft = DFTFilter(config.dft_keep_ratio)
        n_patches = (config.seq_len - config.patch_len) // config.stride + 1
        if config.padding != "none" and (config.seq_len - config.patch_len) % config.stride:
            n_patches += 1
        if n_patches <= 0:
            raise ValueError("seq_len must be >= patch_len")
        local_dim = n_patches * config.d_model
        self.graph = AdaptiveGraph(
            config.in_channels, local_dim, config.graph_dim,
            config.graph_layers, config.dropout)
        self.prior = PriorProjector(config.prior_hidden_dim, config.prior_dim)
        fused_dim = local_dim + config.prior_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, config.head_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden, 1),
        )

    def forward(self, x: Tensor, prior_hidden: Optional[Tensor] = None,
                return_aux: bool = False):
        patches = self.patch(x)
        temporal = self.dft(patches)
        b, c, n, d = temporal.shape
        local = temporal.reshape(b, c, n * d)
        spatial, adjacency = self.graph(local)
        local_repr = spatial.mean(dim=1)

        if prior_hidden is None:
            prior_repr = local_repr.new_zeros((b, self.config.prior_dim))
        else:
            prior_repr = self.prior(prior_hidden)
        prediction = self.head(torch.cat([local_repr, prior_repr], dim=-1))
        if return_aux:
            return prediction, {
                "patches": patches,
                "temporal": temporal,
                "local": spatial,
                "prior": prior_repr,
                "adjacency": adjacency,
            }
        return prediction
