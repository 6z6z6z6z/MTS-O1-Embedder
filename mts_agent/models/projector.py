"""
Cross-Modal Projector
Projects TS features to LLM embedding space.
"""
from typing import Literal, Optional

import torch
import torch.nn as nn


TokenMergeMode = Literal["mean", "attn"]


class TimeSeriesProjector(nn.Module):
    def __init__(
        self,
        ts_dim: int,
        llm_dim: int,
        dropout: float = 0.1,
        expansion_ratio: float = 2.0,
        use_residual: bool = True,
        token_merge_factor: int = 1,
        token_merge_mode: TokenMergeMode = "mean",
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        # P5: allow explicit hidden_dim override; else use expansion_ratio relative to llm_dim
        if hidden_dim is not None:
            _hidden_dim = hidden_dim
        else:
            _hidden_dim = max(llm_dim, int(llm_dim * expansion_ratio))

        self.ts_dim = ts_dim
        self.llm_dim = llm_dim
        self.token_merge_factor = max(1, int(token_merge_factor))
        self.token_merge_mode = token_merge_mode

        # Pre-norm improves distribution alignment before projection.
        self.pre_norm = nn.LayerNorm(ts_dim)

        # A lightweight MLP projector is usually more stable than a single Linear.
        self.net = nn.Sequential(
            nn.Linear(ts_dim, _hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(_hidden_dim, llm_dim),
            nn.Dropout(dropout)
        )

        self.use_residual = bool(use_residual)
        if self.use_residual:
            if ts_dim == llm_dim:
                self.residual_proj = nn.Identity()
            else:
                self.residual_proj = nn.Linear(ts_dim, llm_dim)

        if self.token_merge_factor > 1 and self.token_merge_mode == "attn":
            self.token_merge_scorer = nn.Linear(llm_dim, 1)
        else:
            self.token_merge_scorer = None

    def _merge_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if self.token_merge_factor <= 1:
            return x

        batch_size, token_count, hidden_dim = x.shape
        remainder = token_count % self.token_merge_factor
        if remainder != 0:
            pad_tokens = self.token_merge_factor - remainder
            padding = torch.zeros(
                batch_size,
                pad_tokens,
                hidden_dim,
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, padding], dim=1)

        x = x.view(batch_size, -1, self.token_merge_factor, hidden_dim)

        if self.token_merge_mode == "mean" or self.token_merge_scorer is None:
            return x.mean(dim=2)

        scores = self.token_merge_scorer(x).squeeze(-1)
        weights = torch.softmax(scores, dim=2).unsqueeze(-1)
        return (x * weights).sum(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Seq_Len, ts_dim]
        x_norm = self.pre_norm(x)
        projected = self.net(x_norm)
        if self.use_residual:
            projected = projected + self.residual_proj(x_norm)  # use normed x for consistent scale
        return self._merge_tokens(projected)

