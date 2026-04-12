"""Total loss assembly."""

from __future__ import annotations

import torch
import torch.nn as nn

from .instance_corn import InstanceCORNLoss
from .localization_loss import LocalizationLoss
from .pixel_corn_safe import PixelCORNSafeLoss


class TotalLoss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        loss_cfg = cfg["loss"]

        self.loc_weight = float(loss_cfg.get("loc_ce_weight", 1.0))
        self.lambda_instance = float(model_cfg.get("lambda_instance", 0.0))
        self.instance_aux_warmup_epochs = int(model_cfg.get("instance_aux_warmup_epochs", 0))
        self.instance_aux_on_val = bool(model_cfg.get("instance_aux_on_val", False))

        self.localization = LocalizationLoss()
        self.pixel_corn = PixelCORNSafeLoss(cfg)
        self.instance_corn = InstanceCORNLoss()

    def forward(self, outputs, batch, epoch: int = 0, is_train: bool = True):
        loc_loss = self.localization(outputs["loc_logits"], batch["loc_target"])
        pixel_loss = self.pixel_corn(outputs, batch["damage_rank_target"], epoch=epoch, is_train=is_train)

        instance_pool = outputs.get("instance_pool", {})
        instance_aux_enabled = (is_train or self.instance_aux_on_val) and epoch >= self.instance_aux_warmup_epochs
        if instance_aux_enabled:
            instance_corn_aux = self.instance_corn(
                outputs.get("instance_logits"),
                instance_pool.get("targets"),
            )
            if not torch.is_tensor(instance_corn_aux):
                instance_corn_aux = outputs["loc_logits"].new_tensor(float(instance_corn_aux))
            else:
                instance_corn_aux = instance_corn_aux.to(
                    device=outputs["loc_logits"].device,
                    dtype=outputs["loc_logits"].dtype,
                )
        else:
            instance_corn_aux = outputs["loc_logits"].new_tensor(0.0)

        total_loss = (
            self.loc_weight * loc_loss
            + pixel_loss["loss_corn_main"]
            + pixel_loss["loss_corn_soft"]
            + pixel_loss["tau_reg"]
            + self.lambda_instance * instance_corn_aux
        )

        pixel_counts = instance_pool.get("instance_pixel_counts", []) or []
        instance_aux_stats = {
            "enabled": bool(instance_aux_enabled),
            "valid_instances": int(instance_pool.get("valid_instances", 0) or 0),
            "pool_source": instance_pool.get("pool_source", "disabled"),
            "label_source_counts": instance_pool.get("label_source_counts", {"polygon_subtype": 0, "pixel_majority": 0}),
            "pixel_count_mean": float(sum(pixel_counts) / max(len(pixel_counts), 1)) if pixel_counts else 0.0,
            "pixel_count_min": int(min(pixel_counts)) if pixel_counts else 0,
            "pixel_count_max": int(max(pixel_counts)) if pixel_counts else 0,
        }

        return {
            "loc_loss": loc_loss,
            "pixel_corn_main": pixel_loss["loss_corn_main"],
            "pixel_corn_soft": pixel_loss["loss_corn_soft"],
            "tau_reg": pixel_loss["tau_reg"],
            "instance_corn_aux": instance_corn_aux,
            "total_loss": total_loss,
            "tau_phase": pixel_loss["tau_phase"],
            "corn_soft_enabled": pixel_loss["corn_soft_enabled"],
            "tau_mean": pixel_loss["tau_mean"],
            "tau_std": pixel_loss["tau_std"],
            "tau_min": pixel_loss["tau_min"],
            "tau_max": pixel_loss["tau_max"],
            "corr_tau_difficulty": pixel_loss["corr_tau_difficulty"],
            "corr_raw_tau_difficulty": pixel_loss["corr_raw_tau_difficulty"],
            "tau_by_difficulty_bin": pixel_loss["tau_by_difficulty_bin"],
            "valid_building_pixels": pixel_loss["valid_building_pixels"],
            "instance_aux_enabled": instance_aux_enabled,
            "instance_aux_stats": instance_aux_stats,
        }

