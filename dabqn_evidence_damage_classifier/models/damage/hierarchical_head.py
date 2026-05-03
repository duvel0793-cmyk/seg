from __future__ import annotations

import torch
import torch.nn as nn

from models.damage.corn_head import corn_logits_to_threshold_probabilities, decode_corn_probabilities


def combine_two_stage_probabilities(
    damage_binary_logit: torch.Tensor,
    severity_class_probabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    damaged_probability = torch.sigmoid(damage_binary_logit).unsqueeze(1)
    no_damage = 1.0 - damaged_probability
    damaged_probs = damaged_probability * severity_class_probabilities
    class_probabilities = torch.cat([no_damage, damaged_probs], dim=1)
    class_probabilities = class_probabilities / class_probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return damaged_probability.squeeze(1), class_probabilities


class TwoStageHierarchicalOrdinalHead(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, dropout: float) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.damage_binary = nn.Linear(hidden_features, 1)
        self.severity_corn = nn.Linear(hidden_features, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.shared(x)
        damage_binary_logit = self.damage_binary(hidden).squeeze(1)
        severity_logits = self.severity_corn(hidden)
        threshold_probabilities = corn_logits_to_threshold_probabilities(severity_logits)
        severity_class_probabilities = decode_corn_probabilities(threshold_probabilities)
        damaged_prob, class_probabilities = combine_two_stage_probabilities(
            damage_binary_logit,
            severity_class_probabilities,
        )
        pred_labels = torch.where(
            damaged_prob >= 0.5,
            severity_class_probabilities.argmax(dim=1) + 1,
            torch.zeros_like(damaged_prob, dtype=torch.long),
        )
        return {
            "damage_binary_logit": damage_binary_logit,
            "severity_corn_logits": severity_logits,
            "severity_class_probabilities": severity_class_probabilities,
            "class_probabilities": class_probabilities,
            "pred_labels": pred_labels,
        }
