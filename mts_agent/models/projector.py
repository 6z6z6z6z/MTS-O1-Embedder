"""
Cross-Modal Projector
Projects TS features to LLM embedding space.
"""
import torch.nn as nn

class TimeSeriesProjector(nn.Module):
    def __init__(self, ts_dim, llm_dim, dropout=0.1):
        super().__init__()
        # A simple MLP projector often works better than a single Linear layer
        self.net = nn.Sequential(
            nn.Linear(ts_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [Batch, Seq_Len, ts_dim]
        return self.net(x)

