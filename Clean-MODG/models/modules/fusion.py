"""Feature fusion layers for multi-scale damage reasoning."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConcatMLPFusion(nn.Module):
    def __init__(self, input_dim: int, fusion_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.output_dim = fusion_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )

    def forward(self, tight_feat: torch.Tensor, context_feat: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([tight_feat, context_feat], dim=1))


class GatedScaleFusion(nn.Module):
    def __init__(self, input_dim: int, fusion_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.output_dim = fusion_dim
        self.tight_proj = nn.Linear(input_dim, fusion_dim)
        self.context_proj = nn.Linear(input_dim, fusion_dim)
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, fusion_dim),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )

    def forward(self, tight_feat: torch.Tensor, context_feat: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([tight_feat, context_feat], dim=1))
        fused = gate * self.tight_proj(tight_feat) + (1.0 - gate) * self.context_proj(context_feat)
        return self.out(fused)
