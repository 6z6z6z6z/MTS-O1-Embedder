"""
Time Series Encoder Backbone.

Designed to be dataset-agnostic:
- accepts [B, T], [B, C, T], or [B, T, C]
- supports variable channel counts and sequence lengths
- compresses arbitrary time resolution into a fixed number of tokens
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_norm(norm_type, channels):
    if norm_type == "batch":
        return nn.BatchNorm1d(channels)

    num_groups = min(8, channels)
    while channels % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return nn.GroupNorm(num_groups, channels)


class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.1, norm_type="group"):
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

    def forward(self, x):
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
    Generic 1D ResNet encoder for multivariate time series.

    Maps input [B, C, T] (or [B, T, C] / [B, T]) → [B, target_tokens, hidden_dim].
    No dataset-specific assumptions are made here.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        base_channels=64,
        target_tokens=16,
        dropout=0.1,
        norm_type="group",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.base_channels = base_channels
        self.target_tokens = target_tokens
        self.norm_type = norm_type

        # Stem: project from input_dim channels to base_channels features
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, base_channels, kernel_size=7, stride=1, padding=3, bias=False),
            build_norm(norm_type, base_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        mid_channels = base_channels * 2

        mid_channels = base_channels * 2

        self.layer1 = ResBlock1D(base_channels, base_channels, stride=1, dropout=dropout, norm_type=norm_type)
        self.layer2 = ResBlock1D(base_channels, mid_channels, stride=2, dropout=dropout, norm_type=norm_type)
        self.layer3 = ResBlock1D(mid_channels, hidden_dim, stride=2, dropout=dropout, norm_type=norm_type)

        self.output_norm = build_norm(norm_type, hidden_dim)
        self.output_dropout = nn.Dropout(dropout)

    def _ensure_channel_first(self, x):
        """Normalize input to [B, C, T] regardless of input layout."""
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim == 3 and x.shape[-1] == self.input_dim and x.shape[1] != self.input_dim:
            x = x.transpose(1, 2)

        if x.ndim != 3:
            raise ValueError(f"Expected 2D or 3D time-series tensor, got shape {tuple(x.shape)}")

        current_dim = x.shape[1]
        if current_dim == self.input_dim:
            return x

        if current_dim > self.input_dim:
            return x[:, :self.input_dim, :]

        pad = self.input_dim - current_dim
        return F.pad(x, (0, 0, 0, pad))

    def forward(self, x):
        # x: [B, C, T] or [B, T, C] or [B, T]
        # returns: [B, target_tokens, hidden_dim]
        x = self._ensure_channel_first(x).float()

        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.output_norm(out)
        out = self.output_dropout(out)

        # Compress time axis to a fixed number of tokens
        pooled_len = min(self.target_tokens, max(1, out.shape[-1]))
        out = F.adaptive_avg_pool1d(out, pooled_len)
        out = out.transpose(1, 2)  # [B, target_tokens, hidden_dim]
        return out

