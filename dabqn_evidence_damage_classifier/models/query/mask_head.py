from __future__ import annotations

import torch
import torch.nn as nn


class QueryMaskHead(nn.Module):
    def __init__(self, query_dim: int, pixel_dim: int) -> None:
        super().__init__()
        self.mask_embed = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, pixel_dim),
        )

    def forward(self, query_features: torch.Tensor, pixel_feature: torch.Tensor) -> dict[str, torch.Tensor]:
        mask_embeddings = self.mask_embed(query_features)
        mask_logits = torch.einsum("bqc,bchw->bqhw", mask_embeddings, pixel_feature)
        return {"mask_embeddings": mask_embeddings, "mask_logits": mask_logits}
