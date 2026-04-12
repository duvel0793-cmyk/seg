"""Pixel-level CORN loss with conservative safe tau regularization."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..models.modules.ordinal_utils import class_prob_at_label, corn_logits_to_probs, one_hot_labels
from .corn_loss import CORNLoss
from .tau_regularizer import TauRegularizer


class PixelCORNSafeLoss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        data_cfg = cfg["data"]
        model_cfg = cfg["model"]

        self.ignore_index = int(data_cfg["ignore_index"])
        self.num_classes = int(data_cfg["num_classes_damage"])
        self.tau_mode = str(model_cfg.get("tau_mode", "pixel_corn_safe_v2"))
        self.tau_target = float(model_cfg.get("tau_target", 0.03))
        self.tau_easy = float(model_cfg.get("tau_easy", 0.0))
        self.tau_hard = float(model_cfg.get("tau_hard", 0.12))
        self.tau_std_floor = float(model_cfg.get("tau_std_floor", 0.05))
        self.corn_soft_start_epoch = int(model_cfg.get("corn_soft_start_epoch", 0))
        self.lambda_corn_soft = float(model_cfg.get("lambda_corn_soft", 0.0))
        self.soft_target_detach = bool(model_cfg.get("soft_target_detach", True))
        self.soft_target_max_mix = float(model_cfg.get("soft_target_max_mix", 0.35))

        lambda_tau_reg_legacy = float(model_cfg.get("lambda_tau_reg", 0.001))
        self.corn = CORNLoss(ignore_index=self.ignore_index)
        self.tau_regularizer = TauRegularizer(
            lambda_tau_mean=float(model_cfg.get("lambda_tau_mean", lambda_tau_reg_legacy)),
            lambda_tau_diff=float(model_cfg.get("lambda_tau_diff", lambda_tau_reg_legacy * 5.0)),
            lambda_tau_rank=float(model_cfg.get("lambda_tau_rank", lambda_tau_reg_legacy)),
            lambda_raw_tau_center=float(model_cfg.get("lambda_raw_tau_center", lambda_tau_reg_legacy)),
            lambda_raw_tau_bound=float(model_cfg.get("lambda_raw_tau_bound", lambda_tau_reg_legacy * 0.5)),
            raw_tau_bound=float(model_cfg.get("raw_tau_bound", 4.0)),
            difficulty_bins=int(model_cfg.get("difficulty_bins", 5)),
        )

    def _difficulty_targets(self, raw_logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        valid_mask = labels != self.ignore_index
        if not valid_mask.any():
            empty = raw_logits.new_zeros((0,))
            return {
                "valid_mask": valid_mask,
                "difficulty": empty,
                "target_tau": empty,
                "raw_probs_valid": raw_logits.new_zeros((0, self.num_classes)),
            }

        raw_probs = corn_logits_to_probs(raw_logits)
        raw_probs_valid = raw_probs.permute(0, 2, 3, 1)[valid_mask]
        labels_valid = labels[valid_mask].long()
        p_gt = raw_probs_valid.gather(1, labels_valid[:, None]).squeeze(1)
        difficulty = (1.0 - p_gt).clamp(0.0, 1.0)

        difficulty_std = torch.clamp(difficulty.std(unbiased=False), min=self.tau_std_floor)
        difficulty_z = (difficulty - difficulty.mean()) / difficulty_std
        difficulty_mix = torch.sigmoid(difficulty_z)
        target_tau = self.tau_easy + (self.tau_hard - self.tau_easy) * difficulty_mix
        return {
            "valid_mask": valid_mask,
            "difficulty": difficulty,
            "target_tau": target_tau,
            "raw_probs_valid": raw_probs_valid,
        }

    def _soft_alignment_loss(
        self,
        ordinal_logits: torch.Tensor,
        raw_probs_valid: torch.Tensor,
        difficulty: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask = labels != self.ignore_index
        if not valid_mask.any():
            return ordinal_logits.new_tensor(0.0)

        pred_probs = corn_logits_to_probs(ordinal_logits).permute(0, 2, 3, 1)[valid_mask]
        teacher_probs = raw_probs_valid.detach() if self.soft_target_detach else raw_probs_valid
        hard_onehot = one_hot_labels(labels, num_classes=self.num_classes, ignore_index=self.ignore_index).to(
            device=pred_probs.device, dtype=pred_probs.dtype
        )
        soft_alpha = difficulty.clamp(0.0, 1.0).unsqueeze(1) * self.soft_target_max_mix
        soft_target = (1.0 - soft_alpha) * hard_onehot + soft_alpha * teacher_probs
        soft_target = soft_target / soft_target.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        return -(soft_target * torch.log(pred_probs.clamp_min(1.0e-6))).sum(dim=1).mean()

    def forward(self, outputs: Dict[str, object], labels: torch.Tensor, epoch: int = 0, is_train: bool = True) -> Dict[str, object]:
        ordinal_logits = outputs["ordinal_logits"]
        raw_logits = outputs["raw_ordinal_logits"]
        adaptive_tau = outputs["adaptive_tau"]
        raw_tau = outputs["raw_tau"]

        loss_corn_main = self.corn(ordinal_logits, labels)
        difficulty_info = self._difficulty_targets(raw_logits=raw_logits, labels=labels)

        valid_mask = difficulty_info["valid_mask"]
        if valid_mask.any():
            adaptive_tau_valid = adaptive_tau.permute(0, 2, 3, 1)[valid_mask]
            raw_tau_valid = raw_tau.permute(0, 2, 3, 1)[valid_mask]
            tau_reg = self.tau_regularizer(
                adaptive_tau=adaptive_tau_valid,
                raw_tau=raw_tau_valid,
                target_tau=difficulty_info["target_tau"],
                difficulty=difficulty_info["difficulty"],
            )
        else:
            zero = ordinal_logits.new_tensor(0.0)
            tau_reg = {
                "total": zero,
                "tau_mean_loss": zero,
                "tau_diff_loss": zero,
                "tau_rank_loss": zero,
                "raw_tau_center_loss": zero,
                "raw_tau_bound_loss": zero,
                "corr_tau_difficulty": 0.0,
                "corr_raw_tau_difficulty": 0.0,
                "tau_by_difficulty_bin": [],
            }

        soft_enabled = bool(outputs.get("corn_soft_enabled", False)) and epoch >= self.corn_soft_start_epoch
        if soft_enabled and self.lambda_corn_soft > 0.0:
            loss_corn_soft = self.lambda_corn_soft * self._soft_alignment_loss(
                ordinal_logits=ordinal_logits,
                raw_probs_valid=difficulty_info["raw_probs_valid"],
                difficulty=difficulty_info["difficulty"],
                labels=labels,
            )
        else:
            loss_corn_soft = ordinal_logits.new_tensor(0.0)

        tau_values = outputs["tau_values"].detach()
        return {
            "loss_corn_main": loss_corn_main,
            "loss_corn_soft": loss_corn_soft,
            "tau_reg": tau_reg["total"],
            "tau_mean": float(tau_values.mean().item()) if tau_values.numel() > 0 else 0.0,
            "tau_std": float(tau_values.std(unbiased=False).item()) if tau_values.numel() > 0 else 0.0,
            "tau_min": float(tau_values.min().item()) if tau_values.numel() > 0 else 0.0,
            "tau_max": float(tau_values.max().item()) if tau_values.numel() > 0 else 0.0,
            "corr_tau_difficulty": tau_reg["corr_tau_difficulty"],
            "corr_raw_tau_difficulty": tau_reg["corr_raw_tau_difficulty"],
            "tau_by_difficulty_bin": tau_reg["tau_by_difficulty_bin"],
            "corn_soft_enabled": soft_enabled,
            "tau_phase": outputs.get("tau_phase", "disabled"),
            "valid_building_pixels": int(valid_mask.sum().item()),
        }

