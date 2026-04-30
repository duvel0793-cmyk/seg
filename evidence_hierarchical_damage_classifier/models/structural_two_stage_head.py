from __future__ import annotations

import torch
import torch.nn as nn


class StructuralTwoStageHead(nn.Module):
    def __init__(self, in_features: int, *, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.structural_binary = nn.Linear(hidden_dim, 1)
        self.low_stage = nn.Linear(hidden_dim, 1)
        self.high_stage = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.shared(x)
        structural_binary_logit = self.structural_binary(hidden).squeeze(1)
        low_stage_logit = self.low_stage(hidden).squeeze(1)
        high_stage_logit = self.high_stage(hidden).squeeze(1)

        p_high = torch.sigmoid(structural_binary_logit).unsqueeze(1)
        p_minor_given_low = torch.sigmoid(low_stage_logit).unsqueeze(1)
        p_destroyed_given_high = torch.sigmoid(high_stage_logit).unsqueeze(1)

        class_probabilities = torch.cat(
            [
                (1.0 - p_high) * (1.0 - p_minor_given_low),
                (1.0 - p_high) * p_minor_given_low,
                p_high * (1.0 - p_destroyed_given_high),
                p_high * p_destroyed_given_high,
            ],
            dim=1,
        )
        class_probabilities = class_probabilities / class_probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pred_labels = class_probabilities.argmax(dim=1)
        return {
            "structural_binary_logit": structural_binary_logit,
            "low_stage_logit": low_stage_logit,
            "high_stage_logit": high_stage_logit,
            "structural_class_probabilities": class_probabilities,
            "structural_pred_labels": pred_labels,
        }
