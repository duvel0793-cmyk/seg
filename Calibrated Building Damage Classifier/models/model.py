from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.alignment import ResidualGateAlignment
from models.backbone import ConvNeXtV2Backbone, DEFAULT_CONVNEXTV2_MODEL, resolve_input_mode
from models.change_suppression import DamageAwareChangeBlock
from models.classifier import (
    OrdinalCORNHead,
    ResidualFeatureCalibration,
    corn_logits_to_threshold_probabilities,
    decode_corn_probabilities,
)
from models.fusion import PrePostFusion
from models.token_aggregator import MaskTokenAggregator


class DamageInstanceModel(nn.Module):
    def __init__(
        self,
        *,
        backbone_name: str = DEFAULT_CONVNEXTV2_MODEL,
        pretrained: bool = True,
        input_mode: str = "rgbm",
        feature_dim: int = 256,
        token_count: int = 8,
        token_mixer_layers: int = 1,
        token_mixer_heads: int = 4,
        dropout: float = 0.2,
        use_change_suppression: bool = True,
        change_block_channels: int | None = None,
        enable_pseudo_suppression: bool = True,
        fuse_change_to_tokens: bool = True,
        change_residual_scale: float = 0.2,
        change_gate_init_gamma: float = 0.1,
        gate_temperature: float = 2.0,
        gate_bias_init: float = -2.0,
        enable_damage_aux: bool = True,
        enable_severity_aux: bool = True,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.input_spec = resolve_input_mode(input_mode)
        self.append_mask_channel = bool(self.input_spec["append_mask_channel"])
        self.use_change_suppression = bool(use_change_suppression)
        self.fuse_change_to_tokens = bool(fuse_change_to_tokens)
        self.feature_dim = int(feature_dim)

        self.backbone = ConvNeXtV2Backbone(
            backbone_name=backbone_name,
            in_channels=int(self.input_spec["branch_in_channels"]),
            pretrained=pretrained,
        )
        self.c4_projection = nn.Sequential(
            nn.Conv2d(self.backbone.feature_channels["c4"], feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
        )
        self.c5_projection = nn.Sequential(
            nn.Conv2d(self.backbone.feature_channels["c5"], feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
        )
        self.alignment = ResidualGateAlignment(feature_dim)
        self.fusion = PrePostFusion(feature_dim, feature_dim)
        self.change_block_channels = int(change_block_channels or feature_dim)
        if self.change_block_channels != self.feature_dim:
            self.change_input_projection = nn.Conv2d(self.feature_dim, self.change_block_channels, kernel_size=1, bias=False)
            self.change_output_projection = nn.Conv2d(self.change_block_channels, self.feature_dim, kernel_size=1, bias=False)
        else:
            self.change_input_projection = nn.Identity()
            self.change_output_projection = nn.Identity()
        if self.use_change_suppression:
            self.change_block = DamageAwareChangeBlock(
                self.change_block_channels,
                enable_pseudo_suppression=enable_pseudo_suppression,
                enable_damage_aux=enable_damage_aux,
                enable_severity_aux=enable_severity_aux,
                residual_scale=change_residual_scale,
                gate_temperature=gate_temperature,
                gate_bias_init=gate_bias_init,
            )
            self.gamma_change = nn.Parameter(torch.tensor(float(change_gate_init_gamma), dtype=torch.float32))
        else:
            self.change_block = None
            self.register_parameter("gamma_change", None)
        self.token_aggregator = MaskTokenAggregator(
            feature_dim,
            token_count=token_count,
            mixer_layers=token_mixer_layers,
            mixer_heads=token_mixer_heads,
            dropout=dropout,
        )
        self.feature_calibration = ResidualFeatureCalibration(
            feature_dim,
            hidden_features=max(feature_dim * 2, 256),
            dropout=dropout,
            init_alpha=0.1 if self.use_change_suppression else 0.0,
        )
        self.classifier = OrdinalCORNHead(
            feature_dim,
            hidden_features=max(feature_dim * 2, 256),
            num_classes=self.num_classes,
            dropout=dropout,
        )

    def get_primary_classifier_head_parameters(self) -> list[nn.Parameter]:
        parameters = list(self.feature_calibration.parameters()) + list(self.classifier.parameters())
        return [parameter for parameter in parameters if parameter.requires_grad]

    def _prepare_branch_input(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.append_mask_channel:
            return torch.cat([image, mask], dim=1)
        return image

    def _encode_pair(
        self,
        pre_image: torch.Tensor,
        post_image: torch.Tensor,
        instance_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pre_features = self.backbone(self._prepare_branch_input(pre_image, instance_mask))
        post_features = self.backbone(self._prepare_branch_input(post_image, instance_mask))

        pre_c4 = self.c4_projection(pre_features["c4"])
        post_c4 = self.c4_projection(post_features["c4"])
        pre_c5 = self.c5_projection(pre_features["c5"])
        post_c5 = self.c5_projection(post_features["c5"])

        pre_c4 = F.interpolate(pre_c4, size=pre_c5.shape[-2:], mode="bilinear", align_corners=False)
        post_c4 = F.interpolate(post_c4, size=post_c5.shape[-2:], mode="bilinear", align_corners=False)
        return pre_c5 + pre_c4, post_c5 + post_c4

    @staticmethod
    def _resize_mask(mask: torch.Tensor | None, spatial_size: tuple[int, int], *, dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
        if mask is None:
            return None
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        resized_mask = F.interpolate(mask.float(), size=spatial_size, mode="nearest")
        return resized_mask.to(device=device, dtype=dtype)

    @staticmethod
    def _masked_mean_2d(feature_map: torch.Tensor, mask: torch.Tensor | None, eps: float = 1e-6) -> torch.Tensor:
        flat_feature = feature_map.float().flatten(2)
        pooled_full = flat_feature.mean(dim=-1)
        resized_mask = DamageInstanceModel._resize_mask(
            mask,
            feature_map.shape[-2:],
            dtype=feature_map.dtype,
            device=feature_map.device,
        )
        if resized_mask is None:
            return pooled_full

        flat_mask = resized_mask.flatten(2)
        denominator = flat_mask.sum(dim=-1)
        pooled_masked = (flat_feature * flat_mask).sum(dim=-1) / denominator.clamp_min(eps)
        valid = denominator >= eps
        return torch.where(valid, pooled_masked, pooled_full)

    @staticmethod
    def _compute_map_diagnostics(single_channel_map: torch.Tensor, mask: torch.Tensor | None, eps: float = 1e-6) -> dict[str, torch.Tensor]:
        values = single_channel_map.float().flatten(1)
        map_mean = values.mean(dim=1)
        map_std = values.std(dim=1, unbiased=False)
        resized_mask = DamageInstanceModel._resize_mask(
            mask,
            single_channel_map.shape[-2:],
            dtype=single_channel_map.dtype,
            device=single_channel_map.device,
        )
        if resized_mask is None:
            valid = torch.ones_like(map_mean, dtype=torch.bool)
            zeros = torch.zeros_like(map_mean, dtype=torch.bool)
            return {
                "inside_mean": map_mean,
                "outside_mean": map_mean,
                "gap": torch.zeros_like(map_mean),
                "map_mean": map_mean,
                "map_std": map_std,
                "inside_valid": valid,
                "outside_valid": zeros,
            }

        flat_mask = resized_mask.float().flatten(1)
        inside_denominator = flat_mask.sum(dim=1)
        outside_weight = 1.0 - flat_mask
        outside_denominator = outside_weight.sum(dim=1)

        inside_mean = (values * flat_mask).sum(dim=1) / inside_denominator.clamp_min(eps)
        outside_mean = (values * outside_weight).sum(dim=1) / outside_denominator.clamp_min(eps)

        inside_valid = inside_denominator >= eps
        outside_valid = outside_denominator >= eps
        inside_mean = torch.where(inside_valid, inside_mean, map_mean)
        outside_mean = torch.where(outside_valid, outside_mean, map_mean)

        return {
            "inside_mean": inside_mean,
            "outside_mean": outside_mean,
            "gap": inside_mean - outside_mean,
            "map_mean": map_mean,
            "map_std": map_std,
            "inside_valid": inside_valid,
            "outside_valid": outside_valid,
        }

    def forward(
        self,
        pre_image: torch.Tensor,
        post_image: torch.Tensor,
        instance_mask: torch.Tensor,
    ) -> dict[str, Any]:
        pre_feature, post_feature = self._encode_pair(pre_image, post_image, instance_mask)
        aligned_pre = self.alignment(pre_feature, post_feature)
        fused_feature_map = self.fusion(aligned_pre, post_feature)
        feat_pre_refined = aligned_pre
        feat_post_refined = post_feature
        change_feature = None
        change_gate = None
        damage_map_logits = None
        severity_logit_map = None
        damage_aux_score = None
        severity_score = None
        change_mask_resized = None
        diagnostics: dict[str, torch.Tensor | None] = {}

        if self.use_change_suppression:
            assert self.change_block is not None
            change_outputs = self.change_block(
                self.change_input_projection(aligned_pre),
                self.change_input_projection(post_feature),
                instance_mask,
            )
            feat_pre_refined = self.change_output_projection(change_outputs["feat_pre_refined"])
            feat_post_refined = self.change_output_projection(change_outputs["feat_post_refined"])
            change_feature = self.change_output_projection(change_outputs["change_feature"])
            change_gate_logits = change_outputs["change_gate_logits"]
            change_gate = change_outputs["change_gate"]
            damage_map_logits = change_outputs["damage_map_logits"]
            severity_logit_map = change_outputs.get("severity_logit_map")
            change_mask_resized = change_outputs["mask_resized"]
            damage_aux_score = self._masked_mean_2d(damage_map_logits, change_mask_resized).squeeze(1)
            if severity_logit_map is not None:
                severity_score = torch.sigmoid(self._masked_mean_2d(severity_logit_map, change_mask_resized).squeeze(1))

            gate_diagnostics = self._compute_map_diagnostics(change_gate, change_mask_resized)
            diagnostics.update(
                {
                    "change_gate_logits_mean": change_gate_logits.float().flatten(1).mean(dim=1),
                    "change_gate_inside_mean": gate_diagnostics["inside_mean"],
                    "change_gate_outside_mean": gate_diagnostics["outside_mean"],
                    "change_gate_gap": gate_diagnostics["gap"],
                    "change_gate_mean": gate_diagnostics["map_mean"],
                    "change_gate_std": gate_diagnostics["map_std"],
                    "change_gate_inside_valid": gate_diagnostics["inside_valid"],
                    "change_gate_outside_valid": gate_diagnostics["outside_valid"],
                    "damage_aux_score": torch.sigmoid(damage_aux_score),
                    "severity_score": severity_score,
                }
            )
            if self.fuse_change_to_tokens:
                gamma_change = self.gamma_change.to(device=fused_feature_map.device, dtype=fused_feature_map.dtype)
                fused_feature_map = fused_feature_map + (gamma_change * change_feature)

        token_outputs = self.token_aggregator(fused_feature_map, instance_mask)
        instance_feature = token_outputs["instance_feature"]
        calibrated_feature = self.feature_calibration(instance_feature)
        corn_logits = self.classifier(calibrated_feature)
        threshold_probabilities = corn_logits_to_threshold_probabilities(corn_logits)
        class_probabilities = decode_corn_probabilities(threshold_probabilities)
        pred_labels = class_probabilities.argmax(dim=1)

        return {
            "corn_logits": corn_logits,
            "threshold_probabilities": threshold_probabilities,
            "class_probabilities": class_probabilities,
            "pred_labels": pred_labels,
            "damage_aux_logits": damage_map_logits,
            "damage_aux_score": None if damage_aux_score is None else torch.sigmoid(damage_aux_score),
            "instance_feature": instance_feature,
            "calibrated_feature": calibrated_feature,
            "feat_pre_refined": feat_pre_refined,
            "feat_post_refined": feat_post_refined,
            "change_feature": change_feature,
            "damage_map_logits": damage_map_logits,
            "severity_logit_map": severity_logit_map,
            "severity_score": severity_score,
            "change_gate": change_gate,
            "change_mask_resized": change_mask_resized,
            "evidence_tokens": token_outputs["evidence_tokens"],
            "token_attention": token_outputs["token_attention"],
            "diagnostics": diagnostics,
            "feature_stats": {
                "aligned_pre_norm": aligned_pre.float().flatten(1).norm(dim=1),
                "post_feature_norm": post_feature.float().flatten(1).norm(dim=1),
                "fused_feature_norm": fused_feature_map.float().flatten(1).norm(dim=1),
                "instance_feature_norm": instance_feature.float().norm(dim=1),
                "calibrated_feature_norm": calibrated_feature.float().norm(dim=1),
                "feat_pre_refined_norm": feat_pre_refined.float().flatten(1).norm(dim=1),
                "feat_post_refined_norm": feat_post_refined.float().flatten(1).norm(dim=1),
            },
        }
