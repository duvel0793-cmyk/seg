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


def normalize_corn_outputs(corn_logits: torch.Tensor) -> dict[str, torch.Tensor]:
    threshold_probabilities = corn_logits_to_threshold_probabilities(corn_logits)
    class_probabilities = decode_corn_probabilities(threshold_probabilities)
    class_probabilities = class_probabilities / class_probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return {
        "corn_logits": corn_logits,
        "threshold_probabilities": threshold_probabilities,
        "class_probabilities": class_probabilities,
        "pred_labels": class_probabilities.argmax(dim=1),
    }


class FlatCORNOrdinalHead(nn.Module):
    def __init__(self, in_features: int, *, hidden_features: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, num_classes - 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return normalize_corn_outputs(self.head(x))
