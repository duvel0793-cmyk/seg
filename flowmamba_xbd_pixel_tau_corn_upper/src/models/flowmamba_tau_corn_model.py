"""Top-level model assembly for xBD pixel-level damage upper-bound tests."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import FlowMambaVMambaWrapper
from .heads import InstanceAuxHead, LocalizationHead, PixelCORNHead, SafeTau
from .modules import PolygonPooling


def _empty_instance_pool(device: torch.device) -> Dict[str, object]:
    return {
        "pooled_representations": None,
        "targets": None,
        "valid_instances": 0,
        "instance_pixel_counts": [],
        "label_source_counts": {"polygon_subtype": 0, "pixel_majority": 0},
        "pool_source": "disabled",
    }


class FlowMambaTauCornUpper(nn.Module):
    """Backbone + localization + pixel CORN + safe tau + instance aux."""

    def __init__(self, cfg: Dict[str, object]) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        data_cfg = cfg["data"]
        loss_cfg = cfg["loss"]

        self.num_damage_classes = int(data_cfg["num_classes_damage"])
        self.ignore_index = int(data_cfg["ignore_index"])
        self.instance_pool_source = str(model_cfg.get("instance_pool_source", "ordinal_logits")).lower()
        if self.instance_pool_source == "logits":
            self.instance_pool_source = "ordinal_logits"
        if self.instance_pool_source == "feature":
            self.instance_pool_source = "fused_feature"

        self.backbone = FlowMambaVMambaWrapper(model_cfg)
        self.localization_head = LocalizationHead(
            feature_channels=self.backbone.out_channels,
            decoder_channels=int(model_cfg["decoder_channels"]),
            out_channels=int(data_cfg["num_classes_loc"]),
        )
        self.pixel_corn_head = PixelCORNHead(
            feature_channels=self.backbone.out_channels,
            decoder_channels=int(model_cfg["head_channels"]),
            num_damage_classes=self.num_damage_classes,
        )
        self.safe_tau = SafeTau(
            num_ordinal_logits=self.num_damage_classes - 1,
            feature_channels=int(self.backbone.out_channels[0]),
            enable_tau=bool(model_cfg.get("enable_tau", True)),
            tau_mode=str(model_cfg.get("tau_mode", "pixel_corn_safe_v2")),
            tau_init=float(model_cfg.get("tau_init", 0.0)),
            tau_min=float(model_cfg.get("tau_min", -0.05)),
            tau_max=float(model_cfg.get("tau_max", 0.20)),
            tau_target=float(model_cfg.get("tau_target", 0.03)),
            tau_warmup_epochs=int(model_cfg.get("tau_warmup_epochs", 0)),
            corn_soft_start_epoch=int(model_cfg.get("corn_soft_start_epoch", 0)),
            per_boundary=bool(model_cfg.get("tau_per_boundary", True)),
            hidden_channels=int(model_cfg.get("tau_hidden_channels", 64)),
            detach_features=bool(model_cfg.get("tau_detach_features", True)),
            detach_logits=bool(model_cfg.get("tau_detach_logits", True)),
        )

        self.instance_feature_channels = int(model_cfg.get("instance_feature_channels", model_cfg.get("head_channels", 128)))
        self.instance_feature_adapter = nn.Sequential(
            nn.Conv2d(int(self.backbone.out_channels[0]), self.instance_feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.instance_feature_channels),
            nn.GELU(),
        )
        self.instance_pooling = PolygonPooling(
            source=self.instance_pool_source,
            min_pixels=int(loss_cfg.get("instance_min_pixels", 16)),
            ignore_index=self.ignore_index,
        )

        if self.instance_pool_source == "ordinal_logits":
            instance_in_channels = self.num_damage_classes - 1
        elif self.instance_pool_source == "fused_feature":
            instance_in_channels = self.instance_feature_channels
        elif self.instance_pool_source == "both":
            instance_in_channels = (self.num_damage_classes - 1) + self.instance_feature_channels
        else:
            raise ValueError(f"Unsupported instance_pool_source: {self.instance_pool_source}")

        self.instance_aux_head = InstanceAuxHead(
            in_channels=instance_in_channels,
            out_channels=self.num_damage_classes - 1,
            source=self.instance_pool_source,
            hidden_channels=int(model_cfg.get("instance_head_channels", self.instance_feature_channels)),
        )

    def get_model_metadata(self) -> Dict[str, object]:
        return {
            "num_damage_classes": self.num_damage_classes,
            "ignore_index": self.ignore_index,
            "instance_pool_source": self.instance_pool_source,
            **self.backbone.get_metadata(),
        }

    def forward(
        self,
        batch: Dict[str, object],
        epoch: int = 0,
        enable_instance_aux: bool = True,
    ) -> Dict[str, object]:
        features = self.backbone(batch["pre_image"], batch["post_image"])
        output_size = batch["post_image"].shape[-2:]

        loc_logits = self.localization_head(features["fused_features"], output_size=output_size)
        raw_ordinal_logits = self.pixel_corn_head(features["fused_features"], output_size=output_size)
        tau_output = self.safe_tau(raw_ordinal_logits, features["fused_features"][0], epoch=epoch)
        ordinal_logits = tau_output["ordinal_logits"]
        damage_rank_pred = self.pixel_corn_head.decode(ordinal_logits)

        instance_feature_map = self.instance_feature_adapter(features["fused_features"][0])
        instance_feature_map = F.interpolate(instance_feature_map, size=output_size, mode="bilinear", align_corners=False)

        if enable_instance_aux:
            instance_pool = self.instance_pooling(
                ordinal_logits=ordinal_logits,
                fused_feature=instance_feature_map,
                polygons=batch.get("polygons", []),
                damage_targets=batch.get("damage_rank_target"),
            )
        else:
            instance_pool = _empty_instance_pool(device=loc_logits.device)

        instance_logits = None
        if instance_pool["pooled_representations"] is not None and enable_instance_aux:
            instance_logits = self.instance_aux_head(instance_pool["pooled_representations"])

        backbone_metadata = self.backbone.get_metadata()
        return {
            "loc_logits": loc_logits,
            "raw_ordinal_logits": raw_ordinal_logits,
            "ordinal_logits": ordinal_logits,
            "damage_rank_pred": damage_rank_pred,
            "tau_values": tau_output["tau_values"],
            "adaptive_tau": tau_output["adaptive_tau"],
            "raw_tau": tau_output["raw_tau"],
            "fixed_tau": tau_output["fixed_tau"],
            "tau_phase": tau_output["tau_phase"],
            "corn_soft_enabled": tau_output["corn_soft_enabled"],
            "tau_stats": tau_output["tau_stats"],
            "instance_pool": instance_pool,
            "instance_logits": instance_logits,
            "instance_feature_map": instance_feature_map,
            "backbone_name": backbone_metadata["backend_name"],
            "backbone_reason": backbone_metadata["backend_reason"],
            "backbone_metadata": backbone_metadata,
            "feature_channels": features["feature_channels"],
            "feature_strides": features["feature_strides"],
        }

