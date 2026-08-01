"""ST-PGN v2: Spatial-Temporal Prior Graph Network.

Improvements over v1:
  A. CrossAttentionFusion — replaces naive flatten+concat with multi-head
     cross-attention between foundation-model prior and local graph features.
  B. LearnableFreqFilter — adaptive frequency-domain weights replace the
     hard DFT cutoff.
  C. Topology-sparse AdaptiveGraph — Top-K masking + learnable self-loops.
  D. Monotonicity head — per-patch RUL estimates for physics-informed loss.

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
    # --- New v2 parameters ---
    learnable_filter: bool = True
    use_cross_attention: bool = True
    cross_attn_heads: int = 4
    graph_topk: int = 5
    graph_self_loop: bool = True


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

    Input/output: [B, C, N, D].  Uses a fixed mask so that v1 experiments
    are fully reproducible.
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


class LearnableFreqFilter(nn.Module):
    """Learnable frequency domain filter over the patch-time axis.

    Instead of a hard binary cutoff, each frequency bin is weighted by a
    learnable parameter.  Initialization approximates the old hard cutoff
    with a smooth sigmoid curve so that early training behaves similarly to
    the fixed filter.

    Input/output: [B, C, N, D].
    """

    def __init__(self, n_patches: int, keep_ratio: float = 0.5):
        super().__init__()
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")

        freq_len = n_patches // 2 + 1
        self.w_freq = nn.Parameter(torch.ones(freq_len))

        # Smooth sigmoid initialization that approximates the hard cutoff
        keep = max(1, int(freq_len * keep_ratio))
        with torch.no_grad():
            idx = torch.arange(freq_len, dtype=torch.float32)
            self.w_freq.copy_(torch.sigmoid((keep - 0.5 - idx) * 10.0))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected [B,C,N,D], got {tuple(x.shape)}")
        n = x.shape[2]
        spectrum = torch.fft.rfft(x, dim=2)
        spectrum = spectrum * self.w_freq.view(1, 1, -1, 1)
        return torch.fft.irfft(spectrum, n=n, dim=2)


class AdaptiveGraph(nn.Module):
    """Learnable directed graph interaction over sensor channels.

    v2 enhancements:
      - Scaled dot-product (1/sqrt(d)) — already present in v1.
      - Top-K sparsity mask that retains only the strongest connections.
      - Learnable self-loop weight so each node preserves its own history.
    """

    def __init__(self, channels: int, feature_dim: int, graph_dim: int,
                 layers: int, dropout: float, topk: int = 0,
                 self_loop: bool = False):
        super().__init__()
        self.channels = channels
        self.graph_dim = graph_dim
        self.topk = topk
        self.self_loop = self_loop
        self.source = nn.Parameter(torch.randn(channels, graph_dim) * 0.02)
        self.target = nn.Parameter(torch.randn(graph_dim, channels) * 0.02)

        if self.self_loop:
            self.self_weight = nn.Parameter(torch.zeros(channels))

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

        if self.topk > 0 and self.topk < self.channels:
            _topk_vals, topk_indices = torch.topk(logits, self.topk, dim=-1)
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(-1, topk_indices, True)
            logits = logits.masked_fill(~mask, float('-inf'))

        a = torch.softmax(logits, dim=-1)

        if self.self_loop:
            eye = torch.eye(self.channels, device=a.device)
            a = a + eye * self.self_weight.unsqueeze(1)

        return a

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


class CrossAttentionFusion(nn.Module):
    """Cross-Attention Multi-Scale Fusion.

    Treats the foundation-model prior as a Query and the local spatio-temporal
    graph features as Key/Value, allowing the macroscopic degradation trend to
    adaptively attend to the most critical local sensor interactions.

    This replaces naive flatten+concatenation and simultaneously solves the
    parameter explosion issue while providing interpretable attention maps.
    """

    def __init__(self, prior_dim: int, local_dim: int, heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=prior_dim, num_heads=heads,
            dropout=dropout, batch_first=True)
        self.k_proj = nn.Linear(local_dim, prior_dim)
        self.v_proj = nn.Linear(local_dim, prior_dim)

        self.norm1 = nn.LayerNorm(prior_dim)
        self.norm2 = nn.LayerNorm(prior_dim)
        self.ffn = nn.Sequential(
            nn.Linear(prior_dim, prior_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(prior_dim * 2, prior_dim),
        )

    def forward(self, prior: Tensor, local: Tensor) -> Tensor:
        """Fuse prior [B, D_p] with local features [B, D_l] via cross-attn."""
        q = prior.unsqueeze(1)                  # [B, 1, D_p]
        k = self.k_proj(local).unsqueeze(1)     # [B, 1, D_p]
        v = self.v_proj(local).unsqueeze(1)     # [B, 1, D_p]

        attn_out, _ = self.mha(q, k, v)
        x = self.norm1(q + attn_out)

        ffn_out = self.ffn(x)
        out = self.norm2(x + ffn_out)

        return out.squeeze(1)                   # [B, D_p]


class STPGN(nn.Module):
    """ST-PGN v2: local DFT branch + adaptive graph + optional UniTS prior
    with cross-attention fusion and physics-informed monotonicity head."""

    def __init__(self, config: STPGNConfig):
        super().__init__()
        self.config = config

        # --- Patch embedding ---
        self.patch = PatchEmbedding(
            config.patch_len, config.stride, config.d_model,
            padding=config.padding, dropout=config.dropout)

        n_patches = (config.seq_len - config.patch_len) // config.stride + 1
        if config.padding != "none" and (config.seq_len - config.patch_len) % config.stride:
            n_patches += 1
        if n_patches <= 0:
            raise ValueError("seq_len must be >= patch_len")
        self._n_patches = n_patches

        # --- Frequency filter (v2: learnable or v1: fixed) ---
        if getattr(config, 'learnable_filter', False):
            self.dft = LearnableFreqFilter(n_patches, config.dft_keep_ratio)
        else:
            self.dft = DFTFilter(config.dft_keep_ratio)

        # --- Adaptive graph (v2: sparse + self-loop) ---
        local_dim = n_patches * config.d_model
        self.graph = AdaptiveGraph(
            config.in_channels, local_dim, config.graph_dim,
            config.graph_layers, config.dropout,
            topk=getattr(config, 'graph_topk', 0),
            self_loop=getattr(config, 'graph_self_loop', False))

        # --- Prior projector ---
        self.prior = PriorProjector(config.prior_hidden_dim, config.prior_dim)

        # --- Fusion (v2: cross-attention or v1: concatenation) ---
        self.use_cross_attention = getattr(config, 'use_cross_attention', False)
        if self.use_cross_attention:
            self.cross_attn = CrossAttentionFusion(
                config.prior_dim, local_dim,
                heads=getattr(config, 'cross_attn_heads', 4),
                dropout=config.dropout)

        # --- Prediction head ---
        fused_dim = local_dim + config.prior_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, config.head_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden, 1),
        )

        # --- Monotonicity head: per-patch RUL estimates for physics loss ---
        self.mono_head = nn.Linear(config.d_model, 1)

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

        if self.use_cross_attention:
            fused_prior = self.cross_attn(prior_repr, local_repr)
            fused_repr = torch.cat([local_repr, fused_prior], dim=-1)
        else:
            fused_repr = torch.cat([local_repr, prior_repr], dim=-1)

        prediction = self.head(fused_repr)

        if return_aux:
            # Per-patch RUL estimates for monotonicity regularization.
            # spatial: [B, C, n*d] -> reshape to [B, C, N, D], avg over C
            spatial_4d = spatial.view(b, c, n, d)
            patch_feats = spatial_4d.mean(dim=1)       # [B, N, D]
            rul_sequence = self.mono_head(patch_feats).squeeze(-1)  # [B, N]
            return prediction, {
                "patches": patches,
                "temporal": temporal,
                "local": spatial,
                "prior": prior_repr,
                "adjacency": adjacency,
                "rul_sequence": rul_sequence,
            }
        return prediction
