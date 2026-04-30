"""Config-driven composite loss builder."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.corn_loss import corn_loss
from losses.focal_loss import FocalLoss


class CompositeLoss(nn.Module):
    def __init__(self, loss_cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.loss_type = str(loss_cfg.get("type", "corn")).lower()
        self.num_classes = int(loss_cfg.get("num_classes", 4))
        self.lambda_binary = float(loss_cfg.get("lambda_binary", 0.0))
        self.focal = FocalLoss()

    def forward(self, outputs: Dict[str, torch.Tensor | None], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        labels = batch["label"]
        binary_labels = batch["binary_label"]
        components: Dict[str, torch.Tensor] = {}

        if self.loss_type == "corn":
            components["corn_loss"] = corn_loss(outputs["corn_logits"], labels, self.num_classes)
            total = components["corn_loss"]
        elif self.loss_type == "ce":
            logits = outputs.get("logits")
            if logits is None:
                raise ValueError("CE loss requires model outputs['logits'].")
            components["ce_loss"] = F.cross_entropy(logits, labels)
            total = components["ce_loss"]
        elif self.loss_type == "focal":
            logits = outputs.get("logits")
            if logits is None:
                raise ValueError("Focal loss requires model outputs['logits'].")
            components["focal_loss"] = self.focal(logits, labels)
            total = components["focal_loss"]
        elif self.loss_type == "corn_binary":
            components["corn_loss"] = corn_loss(outputs["corn_logits"], labels, self.num_classes)
            binary_logits = outputs.get("binary_logits")
            if binary_logits is None:
                raise ValueError("corn_binary loss requires model outputs['binary_logits'].")
            components["binary_ce_loss"] = F.cross_entropy(binary_logits, binary_labels)
            total = components["corn_loss"] + self.lambda_binary * components["binary_ce_loss"]
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        components["loss"] = total
        return components


def build_loss(config: Dict[str, Any]) -> CompositeLoss:
    return CompositeLoss(config.get("loss", {}))
