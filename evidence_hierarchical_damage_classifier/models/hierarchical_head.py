from __future__ import annotations

import torch
import torch.nn as nn


def corn_logits_to_threshold_probabilities(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits.float())


def decode_corn_probabilities(threshold_probabilities: torch.Tensor) -> torch.Tensor:
    batch_size, num_thresholds = threshold_probabilities.shape
    num_classes = num_thresholds + 1
    conditional = threshold_probabilities.float().clamp(1e-6, 1.0 - 1e-6)
    survival = torch.ones(batch_size, num_classes, device=conditional.device, dtype=conditional.dtype)
    survival[:, 1:] = torch.cumprod(conditional, dim=1)
    probabilities = torch.zeros_like(survival)
    probabilities[:, :-1] = survival[:, :-1] - survival[:, 1:]
    probabilities[:, -1] = survival[:, -1]
    return probabilities.clamp_min(1e-8)


def combine_two_stage_probabilities(
    damage_binary_logit: torch.Tensor,
    severity_class_probabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    p_damaged = torch.sigmoid(damage_binary_logit).unsqueeze(1)
    no_damage = 1.0 - p_damaged
    damaged_probs = p_damaged * severity_class_probabilities
    class_probabilities = torch.cat([no_damage, damaged_probs], dim=1)
    class_probabilities = class_probabilities / class_probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return p_damaged.squeeze(1), class_probabilities


def decode_two_stage_predictions(
    damage_binary_logit: torch.Tensor,
    severity_class_probabilities: torch.Tensor,
    damage_decision_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    p_damaged, class_probabilities = combine_two_stage_probabilities(damage_binary_logit, severity_class_probabilities)
    damage_mask = p_damaged >= float(damage_decision_threshold)
    severity_pred = severity_class_probabilities.argmax(dim=1) + 1
    pred_labels = torch.where(damage_mask, severity_pred, torch.zeros_like(severity_pred))
    return class_probabilities, pred_labels


class TwoStageHierarchicalOrdinalHead(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, dropout: float, damage_decision_threshold: float = 0.5) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.damage_binary = nn.Linear(hidden_features, 1)
        self.severity_corn = nn.Linear(hidden_features, 2)
        self.damage_decision_threshold = float(damage_decision_threshold)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.shared(x)
        damage_binary_logit = self.damage_binary(hidden).squeeze(1)
        severity_corn_logits = self.severity_corn(hidden)
        threshold_probabilities = corn_logits_to_threshold_probabilities(severity_corn_logits)
        severity_class_probabilities = decode_corn_probabilities(threshold_probabilities)
        class_probabilities, pred_labels = decode_two_stage_predictions(
            damage_binary_logit,
            severity_class_probabilities,
            damage_decision_threshold=self.damage_decision_threshold,
        )
        return {
            "damage_binary_logit": damage_binary_logit,
            "severity_corn_logits": severity_corn_logits,
            "threshold_probabilities": threshold_probabilities,
            "severity_class_probabilities": severity_class_probabilities,
            "class_probabilities": class_probabilities,
            "pred_labels": pred_labels,
        }
