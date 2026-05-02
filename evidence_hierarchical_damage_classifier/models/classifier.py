from __future__ import annotations

import torch
import torch.nn as nn

from models.hierarchical_head import corn_logits_to_threshold_probabilities, decode_corn_probabilities


class ResidualFeatureCalibration(nn.Module):
    def __init__(self, in_features: int, *, hidden_features: int | None = None, dropout: float = 0.1, init_alpha: float = 0.1) -> None:
        super().__init__()
        hidden = int(hidden_features or max(in_features * 2, 256))
        self.norm = nn.LayerNorm(in_features)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, in_features),
            nn.Dropout(dropout),
        )
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (self.alpha.to(device=x.device, dtype=x.dtype) * self.mlp(self.norm(x)))


class OrdinalCORNHead(nn.Module):
    def __init__(self, in_features: int, *, hidden_features: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, num_classes - 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


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
        self.corn = OrdinalCORNHead(
            in_features,
            hidden_features=hidden_features,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = normalize_corn_outputs(self.corn(x))
        outputs["pred_label"] = outputs["pred_labels"]
        return outputs
