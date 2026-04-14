from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from losses.adaptive_tau_safe import (
    AdaptiveTauSafeRegularizer,
    build_soft_class_targets,
    soft_cross_entropy_from_probabilities,
)
from losses.corn_loss import CORNLoss, logits_to_class_probabilities


class CornAdaptiveTauSafeLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        ce_weight: float = 0.3,
        corn_weight: float = 1.0,
        tau_reg_weight: float = 0.05,
        tau_min: float = 0.85,
        tau_max: float = 1.15,
        ce_class_weights: torch.Tensor | None = None,
        soft_target_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.ce_weight = float(ce_weight)
        self.corn_weight = float(corn_weight)
        self.tau_reg_weight = float(tau_reg_weight)
        self.soft_target_weight = float(soft_target_weight)
        if ce_class_weights is not None:
            self.register_buffer("ce_class_weights", ce_class_weights.float())
        else:
            self.ce_class_weights = None
        self.ce_loss = nn.CrossEntropyLoss(weight=self.ce_class_weights)
        self.corn_loss = CORNLoss(num_classes=self.num_classes)
        self.tau_regularizer = AdaptiveTauSafeRegularizer(tau_min=tau_min, tau_max=tau_max, tau_center=1.0)

    def forward(self, outputs: dict[str, torch.Tensor], labels: torch.Tensor) -> dict[str, Any]:
        labels = labels.long().view(-1)
        ce_logits = outputs["ce_logits"]
        tau_adjusted_logits = outputs["tau_adjusted_logits"]
        tau = outputs["tau"]

        loss_ce = self.ce_loss(ce_logits, labels)
        corn_outputs = self.corn_loss(tau_adjusted_logits, labels)
        loss_corn = corn_outputs["loss_corn_main"]
        tau_reg_outputs = self.tau_regularizer(tau)
        loss_tau_reg = tau_reg_outputs["loss_tau_reg"]

        loss_soft_target = tau_adjusted_logits.new_zeros(())
        if self.soft_target_weight > 0:
            soft_targets = build_soft_class_targets(labels, tau.detach(), num_classes=self.num_classes)
            class_probabilities = logits_to_class_probabilities(tau_adjusted_logits)
            loss_soft_target = soft_cross_entropy_from_probabilities(class_probabilities, soft_targets)

        total_loss = (
            (self.ce_weight * loss_ce)
            + (self.corn_weight * loss_corn)
            + (self.tau_reg_weight * loss_tau_reg)
            + (self.soft_target_weight * loss_soft_target)
        )

        return {
            "loss": total_loss,
            "loss_ce": loss_ce.detach(),
            "loss_corn": loss_corn.detach(),
            "loss_tau_reg": loss_tau_reg.detach(),
            "loss_soft_target": loss_soft_target.detach(),
            "tau_mean": tau_reg_outputs["tau_mean"].detach(),
            "tau_std": tau_reg_outputs["tau_std"].detach(),
            "tau_center_penalty": tau_reg_outputs["loss_tau_center"].detach(),
            "tau_spread_penalty": tau_reg_outputs["loss_tau_spread"].detach(),
            "tau_bounds_penalty": tau_reg_outputs["loss_tau_bounds"].detach(),
            "threshold_probabilities": corn_outputs["threshold_probabilities"].detach(),
            "class_probabilities": corn_outputs["class_probabilities"].detach(),
        }


def build_loss(config: dict[str, Any], ce_class_weights: torch.Tensor | None = None) -> CornAdaptiveTauSafeLoss:
    model_cfg = config["model"]
    if str(model_cfg["loss_mode"]) != "corn_adaptive_tau_safe":
        raise ValueError("This project only supports loss_mode=corn_adaptive_tau_safe.")
    return CornAdaptiveTauSafeLoss(
        num_classes=int(model_cfg.get("num_classes", 4)),
        ce_weight=float(model_cfg.get("ce_weight", 0.3)),
        corn_weight=float(model_cfg.get("corn_weight", 1.0)),
        tau_reg_weight=float(model_cfg.get("tau_reg_weight", 0.05)),
        tau_min=float(model_cfg.get("tau_min", 0.85)),
        tau_max=float(model_cfg.get("tau_max", 1.15)),
        ce_class_weights=ce_class_weights,
        soft_target_weight=float(model_cfg.get("soft_target_weight", 0.0)),
    )

