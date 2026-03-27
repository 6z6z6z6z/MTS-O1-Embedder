"""
TC-Former: Multi-Granularity Patch Encoder for Multivariate Time Series

Replaces the CNN-based TimeSeriesEncoder with a Transformer that:
1. Uses linear patch projection (no lossy CNN compression)
2. Applies RevIN for cross-domain normalization (preserves relative amplitudes)
3. Uses alternating Temporal-Channel Attention (TC-Former) to jointly model
   within-channel temporal structure AND cross-channel correlations

Architecture:
  [B, C, T]
    -> RevIN (per-channel instance norm, reversible)
    -> PatchEmbedding: unfold patches + Linear(p -> d) + LayerNorm
       + Learnable position embeddings (per-patch-position)
       + Learnable channel embeddings (per-channel)
    -> TC-Former: L layers of alternating
         (a) Temporal MHA across patches within each channel  [B*C, P, d]
         (b) Channel MHA across channels at each patch position [B*P, C, d]
         (c) FFN
    -> Output: [B, C*P, d_model]   (same interface as TimeSeriesEncoder)

Reference:
  - MOMENT (ICML 2024): masked patch reconstruction pre-training
  - TimesFM (ICML 2024): decoder-only patching for time series
  - iTransformer (ICLR 2024): channel-first attention for MTS
  - RevIN (ICLR 2022): reversible instance normalization
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RevIN
# ---------------------------------------------------------------------------

class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., ICLR 2022).

    Normalises each channel of each sample independently:
      x_norm = (x - mean) / (std + eps)
    and optionally applies learnable affine parameters.

    The normalization statistics are stored per forward call so that
    the inverse transform can be applied at decode time.

    Key difference from PatchTokenizer's instance norm:
      - PatchTokenizer discards mean/std permanently (amplitude destroyed)
      - RevIN stores them and can restore original scale (reversible)
      - Learnable affine (gamma, beta) lets the model tune the scale
    """

    def __init__(self, num_channels: int, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.affine = affine
        self.eps = eps
        if affine:
            # Shared across the channel dimension — model learns how to
            # re-scale normalized features, not per-dataset statistics
            self.gamma = nn.Parameter(torch.ones(1, num_channels, 1))
            self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))

        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input and cache statistics.

        Args:
            x: [B, C, T]

        Returns:
            x_norm: [B, C, T]  normalized, with affine transform applied
        """
        # Handle dynamic channel count (multi-dataset training with variable C)
        C = x.shape[1]
        self._mean = x.mean(dim=-1, keepdim=True).detach()   # [B, C, 1]
        self._std = (x.std(dim=-1, keepdim=True) + self.eps).detach()  # [B, C, 1]

        x = (x - self._mean) / self._std

        if self.affine:
            # Slice affine params if num_channels > self.num_channels at runtime
            gamma = self.gamma[:, :C, :] if C <= self.num_channels else \
                    F.pad(self.gamma, (0, 0, 0, C - self.num_channels, 0, 0), value=1.0)
            beta = self.beta[:, :C, :] if C <= self.num_channels else \
                   F.pad(self.beta, (0, 0, 0, C - self.num_channels, 0, 0), value=0.0)
            x = x * gamma + beta

        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse the normalization (used in forecasting/reconstruction tasks).

        Args:
            x: [B, C, T]  normalized tensor

        Returns:
            x_orig: [B, C, T]  in original scale
        """
        assert self._mean is not None and self._std is not None, \
            "RevIN.forward() must be called before .inverse()"
        C = x.shape[1]
        if self.affine:
            gamma = self.gamma[:, :C, :]
            beta = self.beta[:, :C, :]
            x = (x - beta) / (gamma + self.eps)
        return x * self._std + self._mean


# ---------------------------------------------------------------------------
# PatchEmbedding
# ---------------------------------------------------------------------------

def get_adaptive_patch_size(seq_len: int) -> int:
    """Return a patch size appropriate for the given sequence length."""
    if seq_len <= 32:
        return 2
    elif seq_len <= 64:
        return 4
    elif seq_len <= 256:
        return 8
    elif seq_len <= 1024:
        return 16
    else:
        return 32


class PatchEmbedding(nn.Module):
    """Linear patch tokenization with position and channel embeddings.

    Unlike CNN-based encoders, this module applies NO strided convolution —
    every time-point is included in exactly one patch, preserving all
    temporal information.

    Channel and position embeddings are added after projection so the
    Transformer knows WHICH channel and WHICH time position each token
    represents, enabling meaningful cross-channel and temporal attention.

    Input:  [B, C, T]   (already RevIN-normalized)
    Output: [B, C*P, d_model]  where P = ceil(T / patch_size)
    """

    def __init__(
        self,
        patch_size: int,
        d_model: int,
        max_channels: int = 64,
        max_patches: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model

        # Linear projection: raw patch values -> d_model
        self.proj = nn.Linear(patch_size, d_model, bias=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Learnable absolute position embedding (per patch index)
        self.pos_embed = nn.Embedding(max_patches, d_model)
        # Learnable channel embedding (per channel index)
        # Distinguishes Left-Hand channel from Right-Hand channel, etc.
        self.chan_embed = nn.Embedding(max_channels, d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        nn.init.trunc_normal_(self.pos_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.chan_embed.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Args:
            x: [B, C, T]

        Returns:
            tokens:     [B, C*P, d_model]
            num_patches: P  (needed by TC-Former for reshape)
        """
        B, C, T = x.shape

        # Pad T to be divisible by patch_size
        pad_len = (self.patch_size - T % self.patch_size) % self.patch_size
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))   # pad at the end of time axis

        T_pad = x.shape[-1]
        P = T_pad // self.patch_size

        # Reshape to patches: [B, C, T_pad] -> [B, C, P, patch_size]
        x = x.reshape(B, C, P, self.patch_size)

        # Linear project all patches at once: [B*C*P, patch_size] -> [B*C*P, d_model]
        x_flat = x.reshape(B * C * P, self.patch_size)
        tokens = self.proj(x_flat)          # [B*C*P, d_model]
        tokens = self.norm(tokens)
        tokens = self.dropout(tokens)

        # Reshape to [B, C, P, d_model]
        tokens = tokens.reshape(B, C, P, self.d_model)

        # Add position embedding (shared across channels)
        pos_ids = torch.arange(P, device=x.device)                  # [P]
        tokens = tokens + self.pos_embed(pos_ids).unsqueeze(0).unsqueeze(0)  # [1,1,P,d]

        # Add channel embedding (shared across patch positions)
        chan_ids = torch.arange(C, device=x.device)                  # [C]
        tokens = tokens + self.chan_embed(chan_ids).unsqueeze(0).unsqueeze(2)  # [1,C,1,d]

        # Flatten to [B, C*P, d_model]
        tokens = tokens.reshape(B, C * P, self.d_model)
        return tokens, P


# ---------------------------------------------------------------------------
# TC-Former Layer
# ---------------------------------------------------------------------------

class TCFormerLayer(nn.Module):
    """One TC-Former layer: Temporal Attention -> Channel Attention -> FFN.

    Temporal attention captures within-channel dependencies:
      "This channel's early pattern predicts its later pattern."

    Channel attention captures cross-channel dependencies:
      "Left-hand movement correlates with right-hand counter-movement."

    The alternating structure keeps O(C*P²) + O(P*C²) complexity instead
    of the O((C*P)²) cost of naive joint attention over all tokens.

    All sub-layers use Pre-LN (norm before attention/FFN), which improves
    gradient flow and training stability vs. Post-LN.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # (a) Temporal self-attention (across patches within each channel)
        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # (b) Channel self-attention (across channels at each patch position)
        self.channel_norm = nn.LayerNorm(d_model)
        self.channel_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # (c) Position-wise FFN
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, C: int, P: int) -> torch.Tensor:
        """
        Args:
            x: [B, C*P, d_model]
            C: number of channels
            P: number of patches per channel

        Returns:
            x: [B, C*P, d_model]
        """
        B = x.shape[0]

        # ---- (a) Temporal attention: attend across patches within each channel ----
        # Reshape: [B, C*P, d] -> [B*C, P, d]  (each channel independently)
        x_t = x.reshape(B * C, P, self.temporal_norm.normalized_shape[0])
        x_t_norm = self.temporal_norm(x_t)
        attn_out, _ = self.temporal_attn(x_t_norm, x_t_norm, x_t_norm)
        x_t = x_t + attn_out                        # residual
        # Reshape back: [B*C, P, d] -> [B, C*P, d]
        x = x_t.reshape(B, C * P, -1)

        # ---- (b) Channel attention: attend across channels at each patch position ----
        # Reshape: [B, C*P, d] -> [B, C, P, d] -> [B*P, C, d]
        x_c = x.reshape(B, C, P, -1).permute(0, 2, 1, 3).reshape(B * P, C, -1)
        x_c_norm = self.channel_norm(x_c)
        attn_out, _ = self.channel_attn(x_c_norm, x_c_norm, x_c_norm)
        x_c = x_c + attn_out                        # residual
        # Reshape back: [B*P, C, d] -> [B, P, C, d] -> [B, C, P, d] -> [B, C*P, d]
        x = x_c.reshape(B, P, C, -1).permute(0, 2, 1, 3).reshape(B, C * P, -1)

        # ---- (c) FFN (token-wise) ----
        x = x + self.ffn(self.ffn_norm(x))

        return x


# ---------------------------------------------------------------------------
# TC-Former Encoder (full module, replaces TimeSeriesEncoder)
# ---------------------------------------------------------------------------

class TCFormer(nn.Module):
    """TC-Former: Temporal-Channel Transformer encoder for multivariate time series.

    Full pipeline:
      [B, C, T]
        -> RevIN normalization
        -> PatchEmbedding
        -> L × TCFormerLayer (alternating temporal + channel attention)
        -> Output norm
        -> [B, C*P, d_model]   (same interface as TimeSeriesEncoder)

    Parameter count (Small config: d=128, h=4, ff=512, L=4, max_C=64, max_P=128):
      - RevIN:         24 * 2 = 48  (affine per channel)
      - PatchEmbedding: 4*128 + 128 + 128*128 + 64*128 + 128*128 ≈ 25K
      - TCFormerLayer × 4: ~4 × 530K ≈ 2.1M
      Total: ~2.1M  vs  CNN encoder ~0.5M + projector ~26M
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        patch_size: Optional[int] = None,      # None = adaptive
        max_channels: int = 64,
        max_patches: int = 256,
        use_revin: bool = True,
        revin_affine: bool = True,
    ):
        super().__init__()
        self.d_model = d_model          # same as hidden_dim in existing interface
        self.hidden_dim = d_model       # alias for compatibility
        self.patch_size_cfg = patch_size
        self.use_revin = use_revin

        if use_revin:
            self.revin = RevIN(num_channels=max_channels, affine=revin_affine)
        else:
            self.revin = None

        self.patch_embed = PatchEmbedding(
            patch_size=patch_size if patch_size is not None else 4,  # default; overridden in forward if adaptive
            d_model=d_model,
            max_channels=max_channels,
            max_patches=max_patches,
            dropout=dropout,
        )

        self.layers = nn.ModuleList([
            TCFormerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]  (channel-first multivariate time series)
               Also accepts [B, T, C] — auto-detected by shape heuristic

        Returns:
            tokens: [B, C*P, d_model]
        """
        # Ensure channel-first [B, C, T]
        if x.ndim == 2:
            x = x.unsqueeze(1)  # [B, T] -> [B, 1, T]
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input [B, C, T], got {x.shape}")
        # Auto-detect layout: if dim1 > dim2, likely [B, T, C]
        if x.shape[1] > x.shape[2]:
            x = x.transpose(1, 2)

        x = x.float()
        B, C, T = x.shape

        # (1) RevIN normalization
        if self.revin is not None:
            x = self.revin(x)

        # (2) Adaptive patch size if not fixed
        patch_size = self.patch_size_cfg
        if patch_size is None:
            patch_size = get_adaptive_patch_size(T)
            # Dynamically update patch_embed's patch_size for this forward pass
            # (we use a fixed patch_size at construction time, but can override for inference)
            self.patch_embed.patch_size = patch_size

        # (3) Patch embedding: [B, C, T] -> [B, C*P, d_model]
        tokens, P = self.patch_embed(x)   # P = ceil(T / patch_size)

        # (4) TC-Former layers
        for layer in self.layers:
            tokens = layer(tokens, C, P)

        # (5) Output norm
        tokens = self.out_norm(tokens)

        return tokens  # [B, C*P, d_model]

    @property
    def num_patches(self) -> int:
        """Estimated number of patches per channel (for NATOPS T=51, patch=4 -> 13)."""
        return math.ceil(51 / (self.patch_size_cfg or 4))
