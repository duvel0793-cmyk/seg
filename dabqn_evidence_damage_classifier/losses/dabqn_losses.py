from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.box_losses import box_cxcywh_to_xyxy, generalized_box_iou
from losses.mask_losses import dice_loss, sigmoid_bce_loss
from losses.ordinal_losses import corn_loss


class DABQNLoss(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        loss_cfg = config["loss"]
        self.weight_obj = float(loss_cfg.get("obj_weight", 1.0))
        self.weight_box_l1 = float(loss_cfg.get("box_l1_weight", 5.0))
        self.weight_giou = float(loss_cfg.get("giou_weight", 2.0))
        self.weight_mask = float(loss_cfg.get("mask_bce_weight", 2.0))
        self.weight_dice = float(loss_cfg.get("dice_weight", 5.0))
        self.weight_damage = float(loss_cfg.get("damage_weight", 1.0))
        self.eos_coef = float(loss_cfg.get("eos_coef", 0.1))
        self.damage_loss_type = str(config["model"].get("damage_head_type", "corn"))

    @staticmethod
    def _gather_matched(
        tensor: torch.Tensor,
        matches: list[tuple[torch.Tensor, torch.Tensor]],
        *,
        batch_first_query: bool = True,
    ) -> list[torch.Tensor]:
        gathered = []
        for batch_index, (pred_indices, _) in enumerate(matches):
            if pred_indices.numel() == 0:
                gathered.append(tensor.new_zeros((0,) + tensor.shape[2:]))
                continue
            if batch_first_query:
                gathered.append(tensor[batch_index, pred_indices])
            else:
                gathered.append(tensor[pred_indices, batch_index])
        return gathered

    def _loss_objectness(self, pred_logits: torch.Tensor, matches: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        target_classes = torch.zeros(pred_logits.shape[:2], dtype=torch.long, device=pred_logits.device)
        for batch_index, (pred_indices, _) in enumerate(matches):
            if pred_indices.numel() > 0:
                target_classes[batch_index, pred_indices] = 1
        class_weights = torch.tensor([self.eos_coef, 1.0], device=pred_logits.device, dtype=pred_logits.dtype)
        return F.cross_entropy(pred_logits.transpose(1, 2), target_classes, weight=class_weights, reduction="mean")

    def _loss_boxes(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, torch.Tensor]],
        matches: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        l1_losses = []
        giou_losses = []
        for batch_index, (pred_indices, target_indices) in enumerate(matches):
            if pred_indices.numel() == 0:
                continue
            pred_boxes = outputs["pred_boxes"][batch_index, pred_indices]
            target_boxes = targets[batch_index]["boxes_norm"][target_indices]
            l1_losses.append(F.l1_loss(pred_boxes, target_boxes, reduction="mean"))
            giou = generalized_box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(target_boxes))
            giou_losses.append((1.0 - torch.diag(giou)).mean())
        zero = outputs["pred_boxes"].new_tensor(0.0)
        return (torch.stack(l1_losses).mean() if l1_losses else zero, torch.stack(giou_losses).mean() if giou_losses else zero)

    def _loss_masks(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, torch.Tensor]],
        matches: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bce_losses = []
        dice_losses = []
        for batch_index, (pred_indices, target_indices) in enumerate(matches):
            if pred_indices.numel() == 0:
                continue
            pred_masks = outputs["pred_masks"][batch_index, pred_indices]
            target_masks = targets[batch_index]["masks"][target_indices]
            if pred_masks.shape[-2:] != target_masks.shape[-2:]:
                target_masks = F.interpolate(target_masks.unsqueeze(1), size=pred_masks.shape[-2:], mode="nearest").squeeze(1)
            bce_losses.append(sigmoid_bce_loss(pred_masks, target_masks))
            dice_losses.append(dice_loss(pred_masks, target_masks))
        zero = outputs["pred_masks"].new_tensor(0.0)
        return (torch.stack(bce_losses).mean() if bce_losses else zero, torch.stack(dice_losses).mean() if dice_losses else zero)

    def _loss_damage(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, torch.Tensor]],
        matches: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        loss_terms = []
        for batch_index, (pred_indices, target_indices) in enumerate(matches):
            if pred_indices.numel() == 0:
                continue
            labels = targets[batch_index]["labels"][target_indices]
            if self.damage_loss_type == "corn":
                damage_logits = outputs["damage_logits"][batch_index, pred_indices]
                loss_terms.append(corn_loss(damage_logits, labels))
            elif self.damage_loss_type == "hierarchical":
                binary_logits = outputs["damage_binary_logits"][batch_index, pred_indices]
                severity_logits = outputs["damage_severity_logits"][batch_index, pred_indices]
                damaged_target = (labels > 0).float()
                binary_loss = F.binary_cross_entropy_with_logits(binary_logits.float(), damaged_target, reduction="mean")
                damaged_mask = labels > 0
                if damaged_mask.any():
                    severity_loss = corn_loss(severity_logits[damaged_mask], labels[damaged_mask] - 1)
                else:
                    severity_loss = binary_logits.new_tensor(0.0)
                loss_terms.append(binary_loss + severity_loss)
            elif self.damage_loss_type == "ce":
                damage_logits = outputs["damage_logits"][batch_index, pred_indices]
                loss_terms.append(F.cross_entropy(damage_logits.float(), labels, reduction="mean"))
            else:
                raise ValueError(f"Unsupported damage_loss_type='{self.damage_loss_type}'.")
        if not loss_terms:
            if outputs.get("damage_logits") is not None:
                return outputs["damage_logits"].new_tensor(0.0)
            return outputs["pred_logits"].new_tensor(0.0)
        return torch.stack(loss_terms).mean()

    def forward(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, torch.Tensor]],
        matches: list[tuple[torch.Tensor, torch.Tensor]],
        *,
        stage: str,
    ) -> dict[str, torch.Tensor]:
        stage_name = str(stage)
        loss_obj = self._loss_objectness(outputs["pred_logits"], matches)
        loss_box_l1, loss_giou = self._loss_boxes(outputs, targets, matches)
        loss_mask_bce, loss_dice = self._loss_masks(outputs, targets, matches)
        if stage_name == "localization":
            loss_damage = outputs["pred_logits"].new_tensor(0.0)
        else:
            loss_damage = self._loss_damage(outputs, targets, matches)

        total = (
            self.weight_obj * loss_obj
            + self.weight_box_l1 * loss_box_l1
            + self.weight_giou * loss_giou
            + self.weight_mask * loss_mask_bce
            + self.weight_dice * loss_dice
            + self.weight_damage * loss_damage
        )
        return {
            "loss_total": total,
            "loss_objectness": loss_obj,
            "loss_box_l1": loss_box_l1,
            "loss_giou": loss_giou,
            "loss_mask_bce": loss_mask_bce,
            "loss_dice": loss_dice,
            "loss_damage": loss_damage,
        }
