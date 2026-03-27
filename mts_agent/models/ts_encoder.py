"""
Time Series Encoder Backbone.

Designed to be dataset-agnostic:
- accepts [B, T], [B, C, T], or [B, T, C]
- supports variable channel counts and sequence lengths
- compresses arbitrary time resolution into a fixed number of tokens

Encoders:
  TimeSeriesEncoder  — CNN-based (original, learned compression)
  PatchTokenizer     — Linear patch tokenizer (PatchTST-style, lossless)
"""
from typing import Literal, Optional, Sequence
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


NormType = Literal["group", "batch"]
InputLayoutType = Literal["auto", "channel_first", "channel_last"]


def build_norm(norm_type: NormType, channels: int) -> nn.Module:
    if norm_type == "batch":
        return nn.BatchNorm1d(channels)

    num_groups = min(8, channels)
    while channels % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return nn.GroupNorm(num_groups, channels)


class ResBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.1,
        norm_type: NormType = "group",
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = build_norm(norm_type, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = build_norm(norm_type, out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                build_norm(norm_type, out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.activation(out)
        return out


class TimeSeriesEncoder(nn.Module):
    """
    Channel-Independent (CI) 1D ResNet encoder for multivariate time series.
    Works by treating each channel independently, mapping [B, C, T] -> [B*C, 1, T],
    applying heavy Stem downsampling -> Patching -> [B, C * num_patches, hidden_dim].
    Designed for extreme long sequence and dynamic channel dimensions.
    """

    def __init__(
        self,
        hidden_dim: int,
        base_channels: int = 64,
        stem_strides: Optional[Sequence[int]] = None,
        patch_size: int = 5,
        dropout: float = 0.1,
        norm_type: NormType = "group",
        input_layout: InputLayoutType = "auto",
        max_auto_channel_dim: int = 128,
        token_merge_factor: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_channels = base_channels
        self.patch_size = patch_size
        self.norm_type = norm_type
        self.input_layout = input_layout
        self.max_auto_channel_dim = max_auto_channel_dim
        self.token_merge_factor = max(1, int(token_merge_factor))
        self._layout_warning_emitted = False

        resolved_stem_strides = list(stem_strides) if stem_strides is not None else [5, 5]

        # CI: Input is always 1 channel conceptually ([B*C, 1, T])
        in_channels = 1
        
        # Stem: Heavy downsampling. Example: two strides of 5 (25x compression)
        stem_layers = []
        current_channels = base_channels
        prev_channels = in_channels
        for stride in resolved_stem_strides:
            stem_layers.extend([
                nn.Conv1d(prev_channels, current_channels, kernel_size=stride*2-1, stride=stride, padding=stride-1, bias=False),
                build_norm(norm_type, current_channels),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_channels = current_channels
            current_channels = current_channels * 2

        self.stem = nn.Sequential(*stem_layers)
        
        mid_channels = prev_channels
        
        # A couple of ResBlocks for deep feature extraction after stem downsampling
        self.layer1 = ResBlock1D(mid_channels, mid_channels, stride=1, dropout=dropout, norm_type=norm_type)
        self.layer2 = ResBlock1D(mid_channels, hidden_dim, stride=1, dropout=dropout, norm_type=norm_type)
        
        self.output_norm = build_norm(norm_type, hidden_dim)
        self.output_dropout = nn.Dropout(dropout)

        # Patching layer: Extract non-overlapping patch embeddings. Using kernel=stride=patch_size.
        # Ensure padding in case sequence length is smaller than patch_size or not divisible.
        self.patching = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)

    def _warn_auto_layout(self, message: str) -> None:
        if self._layout_warning_emitted:
            return
        warnings.warn(message, stacklevel=2)
        self._layout_warning_emitted = True

    def _resolve_auto_layout(self, x: torch.Tensor) -> torch.Tensor:
        channel_first_channels = x.shape[1]
        channel_last_channels = x.shape[2]
        channel_first_likely = channel_first_channels <= self.max_auto_channel_dim
        channel_last_likely = channel_last_channels <= self.max_auto_channel_dim

        if channel_first_likely and not channel_last_likely:
            return x
        if channel_last_likely and not channel_first_likely:
            self._warn_auto_layout(
                f"Auto-detected channel-last input layout for shape {tuple(x.shape)}; transposing to [B, C, T]."
            )
            return x.transpose(1, 2)

        if x.shape[1] <= x.shape[2]:
            if channel_last_likely:
                self._warn_auto_layout(
                    f"Ambiguous input layout for shape {tuple(x.shape)}; assuming channel-first because dim1 <= dim2."
                )
            return x

        self._warn_auto_layout(
            f"Ambiguous input layout for shape {tuple(x.shape)}; assuming channel-last because dim2 < dim1."
        )
        return x.transpose(1, 2)

    def _ensure_channel_first(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input to [B, C, T] regardless of input layout."""
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected 2D or 3D time-series tensor, got shape {tuple(x.shape)}")

        if self.input_layout == "channel_first":
            return x
        if self.input_layout == "channel_last":
            return x.transpose(1, 2)
        return self._resolve_auto_layout(x)

    def _merge_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.token_merge_factor <= 1:
            return tokens

        batch_size, token_count, hidden_dim = tokens.shape
        remainder = token_count % self.token_merge_factor
        if remainder != 0:
            pad_tokens = self.token_merge_factor - remainder
            padding = torch.zeros(
                batch_size,
                pad_tokens,
                hidden_dim,
                device=tokens.device,
                dtype=tokens.dtype,
            )
            tokens = torch.cat([tokens, padding], dim=1)

        merged = tokens.view(batch_size, -1, self.token_merge_factor, hidden_dim)
        return merged.mean(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x = self._ensure_channel_first(x).float()
        B, C, T = x.shape
        
        # Channel Independence (CI): Treat every channel as an independent sample
        x = x.view(B * C, 1, T)
        
        # Stem -> Residual blocks
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.output_norm(out)
        out = self.output_dropout(out)
        
        # Ensure sequence length is large enough for patching
        seq_len = out.shape[-1]
        if seq_len < self.patch_size:
            pad_len = self.patch_size - seq_len
            out = F.pad(out, (0, pad_len))
        elif seq_len % self.patch_size != 0:
            pad_len = self.patch_size - (seq_len % self.patch_size)
            out = F.pad(out, (0, pad_len))
        
        # Patching: [B*C, Hidden, L] -> [B*C, Hidden, Num_Patches]
        out = self.patching(out)
        
        # Reshape for Transformer: [B, C * Num_Patches, Hidden]
        # This allows LLM cross-attention to naturally fuse across Channels and Time(Patches)
        _, Hidden, Num_Patches = out.shape
        out = out.transpose(1, 2)  # [B*C, Num_Patches, Hidden]
        out = out.contiguous().view(B, C * Num_Patches, Hidden)

        return self._merge_tokens(out)


class PatchTokenizer(nn.Module):
    """Channel-Independent linear patch tokenizer (PatchTST-style).

    Replaces CNN encoder with direct linear projection of raw patches.
    Key advantages over CNN:
    - No lossy learned compression — all patch information is preserved
    - Shorter gradient path → easier to train with small datasets
    - Proven effective: basis of PatchTST (ICLR 2023, SOTA on many TS tasks)

    Pipeline per channel:
      1. Instance-normalize each channel (focus on temporal pattern, not magnitude)
      2. Slice into non-overlapping patches of size P
      3. Linear(P → hidden_dim) + LayerNorm + Dropout

    Input:  (B, C, T)  or auto-detected layout
    Output: (B, C * num_patches, hidden_dim)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        patch_size: int = 8,
        patch_stride: Optional[int] = None,
        dropout: float = 0.1,
        input_layout: InputLayoutType = "auto",
        max_auto_channel_dim: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.patch_stride = patch_stride if patch_stride is not None else patch_size
        self.input_layout = input_layout
        self.max_auto_channel_dim = max_auto_channel_dim
        self._layout_warning_emitted = False

        # Shared linear projection across all patches and channels (CI)
        self.patch_embed = nn.Linear(patch_size, hidden_dim, bias=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)

    # ── layout helpers (shared with TimeSeriesEncoder) ────────────────────────
    def _warn_auto_layout(self, message: str) -> None:
        if self._layout_warning_emitted:
            return
        warnings.warn(message, stacklevel=2)
        self._layout_warning_emitted = True

    def _resolve_auto_layout(self, x: torch.Tensor) -> torch.Tensor:
        channel_first_channels = x.shape[1]
        channel_last_channels = x.shape[2]
        channel_first_likely = channel_first_channels <= self.max_auto_channel_dim
        channel_last_likely = channel_last_channels <= self.max_auto_channel_dim

        if channel_first_likely and not channel_last_likely:
            return x
        if channel_last_likely and not channel_first_likely:
            self._warn_auto_layout(
                f"Auto-detected channel-last input layout for shape {tuple(x.shape)}; transposing to [B, C, T]."
            )
            return x.transpose(1, 2)
        if x.shape[1] <= x.shape[2]:
            if channel_last_likely:
                self._warn_auto_layout(
                    f"Ambiguous input layout for shape {tuple(x.shape)}; assuming channel-first because dim1 <= dim2."
                )
            return x
        self._warn_auto_layout(
            f"Ambiguous input layout for shape {tuple(x.shape)}; assuming channel-last because dim2 < dim1."
        )
        return x.transpose(1, 2)

    def _ensure_channel_first(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected 2D or 3D time-series tensor, got shape {tuple(x.shape)}")
        if self.input_layout == "channel_first":
            return x
        if self.input_layout == "channel_last":
            return x.transpose(1, 2)
        return self._resolve_auto_layout(x)

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_channel_first(x).float()
        B, C, T = x.shape

        # Per-channel instance normalization: focus on temporal pattern, not magnitude
        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
        x = (x - mean) / std

        # Pad so T is divisible by patch_stride
        pad = (self.patch_stride - (T % self.patch_stride)) % self.patch_stride
        if pad > 0:
            x = F.pad(x, (0, pad))

        # Unfold: (B, C, T_pad) → (B, C, num_patches, patch_size)
        x_patches = x.unfold(-1, self.patch_size, self.patch_stride)
        num_patches = x_patches.shape[2]

        # Flatten for linear: (B*C*num_patches, patch_size)
        x_flat = x_patches.reshape(B * C * num_patches, self.patch_size)

        # Linear project + norm + dropout
        out = self.patch_embed(x_flat)        # (B*C*num_patches, hidden_dim)
        out = self.norm(out)
        out = self.dropout_layer(out)

        # Reshape to (B, C * num_patches, hidden_dim)
        return out.view(B, C * num_patches, self.hidden_dim)


class ChannelMixer(nn.Module):
    """Lightweight cross-channel attention after CI encoder.

    Captures cross-channel correlations (e.g. left-hand / right-hand coordination
    in NATOPS gesture data) that the channel-independent encoder cannot express.

    Pipeline:
      Input  [B, C*P, D]
        → reshape [B*P, C, D]           (group by patch position)
        → multi-head self-attention over C channels
        → residual + LayerNorm
        → reshape back [B, C*P, D]

    Parameters
    ----------
    hidden_dim  : feature dimension D (same as ts_encoder hidden_dim)
    num_heads   : attention heads; must divide hidden_dim
    dropout     : attention dropout
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, num_channels: int) -> torch.Tensor:
        """
        x           : [B, C*P, D]
        num_channels: C (number of sensor channels)
        Returns     : [B, C*P, D]
        """
        B, CP, D = x.shape
        num_patches = CP // num_channels          # P

        # [B, C*P, D] → [B, C, P, D] → [B*P, C, D]
        x_cp = x.view(B, num_channels, num_patches, D)
        x_pc = x_cp.permute(0, 2, 1, 3).contiguous().view(B * num_patches, num_channels, D)

        attn_out, _ = self.attn(x_pc, x_pc, x_pc)   # [B*P, C, D]
        x_pc = self.norm(x_pc + attn_out)             # residual + norm

        # [B*P, C, D] → [B, P, C, D] → [B, C, P, D] → [B, C*P, D]
        x_out = (x_pc.view(B, num_patches, num_channels, D)
                      .permute(0, 2, 1, 3)
                      .contiguous()
                      .view(B, CP, D))
        return x_out
