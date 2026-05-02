from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def corn_logits_to_threshold_probabilities(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits.float())


def decode_corn_probabilities(threshold_probabilities: torch.Tensor) -> torch.Tensor:
    batch_size, num_thresholds = threshold_probabilities.shape
    num_classes = num_thresholds + 1
    conditional = threshold_probabilities.float().clamp(1e-6, 1.0 - 1e-6)
    survival = torch.ones(
        batch_size,
        num_classes,
        device=threshold_probabilities.device,
        dtype=conditional.dtype,
    )
    survival[:, 1:] = torch.cumprod(conditional, dim=1)
    probabilities = torch.zeros_like(survival)
    probabilities[:, :-1] = survival[:, :-1] - survival[:, 1:]
    probabilities[:, -1] = survival[:, -1]
    return probabilities.clamp_min(1e-8)


def decode_corn_logits(logits: torch.Tensor) -> torch.Tensor:
    return decode_corn_probabilities(corn_logits_to_threshold_probabilities(logits))


def get_corn_logits(outputs: dict[str, Any] | torch.Tensor) -> torch.Tensor:
    if isinstance(outputs, dict):
        if "corn_logits" not in outputs:
            raise KeyError("Model outputs dict is missing 'corn_logits'.")
        return outputs["corn_logits"]
    if torch.is_tensor(outputs):
        return outputs
    raise TypeError(f"Unsupported model output type: {type(outputs)!r}")


def normalize_model_outputs(outputs: dict[str, Any] | torch.Tensor) -> dict[str, Any]:
    if isinstance(outputs, dict):
        normalized = dict(outputs)
        corn_logits = get_corn_logits(normalized)
        if "threshold_probabilities" not in normalized:
            normalized["threshold_probabilities"] = corn_logits_to_threshold_probabilities(corn_logits)
        if "class_probabilities" not in normalized:
            normalized["class_probabilities"] = decode_corn_probabilities(normalized["threshold_probabilities"])
        if "pred_labels" not in normalized:
            normalized["pred_labels"] = normalized["class_probabilities"].argmax(dim=1)
        return normalized

    corn_logits = get_corn_logits(outputs)
    threshold_probabilities = corn_logits_to_threshold_probabilities(corn_logits)
    class_probabilities = decode_corn_probabilities(threshold_probabilities)
    return {
        "corn_logits": corn_logits,
        "threshold_probabilities": threshold_probabilities,
        "class_probabilities": class_probabilities,
        "pred_labels": class_probabilities.argmax(dim=1),
    }


class OrdinalCORNHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        *,
        hidden_features: int = 512,
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if int(num_classes) < 2:
            raise ValueError("OrdinalCORNHead expects at least 2 ordered classes.")
        self.num_classes = int(num_classes)
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, self.num_classes - 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class ResidualFeatureCalibration(nn.Module):
    """Light residual feature calibration before the ordinal CORN head."""

    def __init__(
        self,
        in_features: int,
        *,
        hidden_features: int | None = None,
        dropout: float = 0.1,
        init_alpha: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_features or max(in_features * 2, 256))
        self.norm = nn.LayerNorm(in_features)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_features),
            nn.Dropout(dropout),
        )
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.to(device=x.device, dtype=x.dtype)
        return x + (alpha * self.mlp(self.norm(x)))
