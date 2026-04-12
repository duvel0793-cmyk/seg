"""Instance-level auxiliary head for pooled polygon representations."""

from __future__ import annotations

import torch
import torch.nn as nn


class InstanceAuxHead(nn.Module):
    """Small MLP that predicts ordinal logits from polygon pooled features."""

    def __init__(self, in_channels: int, out_channels: int, source: str = "ordinal_logits", hidden_channels: int = 128) -> None:
        super().__init__()
        self.source = source
        hidden_channels = max(int(hidden_channels), out_channels)
        self.proj = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, pooled_repr: torch.Tensor) -> torch.Tensor:
        if pooled_repr.ndim != 2:
            raise ValueError(f"Expected pooled_repr to be [N, C], got {tuple(pooled_repr.shape)}")
        return self.proj(pooled_repr)

