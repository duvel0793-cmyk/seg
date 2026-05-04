from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.damage.corn_head import FlatCORNOrdinalHead
from models.damage.evidence_pooling import crop_feature_by_boxes, dilate_masks, expand_boxes_cxcywh, masked_mean_pool
from models.damage.hierarchical_head import TwoStageHierarchicalOrdinalHead


class FeatureProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CropScaleEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8 if hidden_dim % 8 == 0 else 1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8 if hidden_dim % 8 == 0 else 1, hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        return encoded.flatten(1)


class ThreeScaleDamageBranch(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        query_dim: int,
        hidden_dim: int,
        branch_type: str,
        head_type: str,
        dropout: float,
        crop_feature_size: int,
        scale_factors: dict[str, float],
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.query_dim = int(query_dim)
        self.hidden_dim = int(hidden_dim)
        self.branch_type = str(branch_type)
        self.head_type = str(head_type)
        self.crop_feature_size = int(crop_feature_size)
        self.scale_factors = dict(scale_factors)

        pooled_dim = (self.feature_dim * 3) + self.query_dim
        self.base_projector = FeatureProjector(self.query_dim, self.hidden_dim, dropout)
        self.tight_projector = FeatureProjector(pooled_dim, self.hidden_dim, dropout)
        self.context_projector = FeatureProjector(pooled_dim, self.hidden_dim, dropout)
        self.neighborhood_projector = FeatureProjector(pooled_dim, self.hidden_dim, dropout)

        crop_in_channels = self.feature_dim * 3
        self.crop_tight = CropScaleEncoder(crop_in_channels, self.hidden_dim)
        self.crop_context = CropScaleEncoder(crop_in_channels, self.hidden_dim)
        self.crop_neighborhood = CropScaleEncoder(crop_in_channels, self.hidden_dim)

        gate_dim = self.hidden_dim * 4
        self.gate = nn.Sequential(
            nn.LayerNorm(gate_dim),
            nn.Linear(gate_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 4),
        )
        if self.head_type == "corn":
            self.damage_head: nn.Module = FlatCORNOrdinalHead(
                self.hidden_dim,
                hidden_features=self.hidden_dim * 2,
                num_classes=4,
                dropout=dropout,
            )
        elif self.head_type == "hierarchical":
            self.damage_head = TwoStageHierarchicalOrdinalHead(
                self.hidden_dim,
                hidden_features=self.hidden_dim * 2,
                dropout=dropout,
            )
        elif self.head_type == "ce":
            self.damage_head = nn.Linear(self.hidden_dim, 4)
        else:
            raise ValueError(f"Unsupported damage head_type='{self.head_type}'.")

    def _pool_feature_triplet(self, feature: torch.Tensor, masks: torch.Tensor, query_features: torch.Tensor) -> torch.Tensor:
        pre_pool = masked_mean_pool(feature["pre"], masks)
        post_pool = masked_mean_pool(feature["post"], masks)
        diff_pool = masked_mean_pool(feature["diff"], masks)
        combined = torch.cat([pre_pool, post_pool, diff_pool, query_features], dim=-1)
        return combined

    def _prepare_feature_triplets(self, pre_pyramid: dict[str, torch.Tensor], post_pyramid: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
        feature_triplets: dict[str, dict[str, torch.Tensor]] = {}
        for key in ("p2", "p3", "p4"):
            pre = pre_pyramid[key]
            post = post_pyramid[key]
            feature_triplets[key] = {"pre": pre, "post": post, "diff": torch.abs(post - pre)}
        return feature_triplets

    def _forward_mask_pool(
        self,
        *,
        pre_pyramid: dict[str, torch.Tensor],
        post_pyramid: dict[str, torch.Tensor],
        query_features: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        triplets = self._prepare_feature_triplets(pre_pyramid, post_pyramid)
        tight_masks = dilate_masks(masks, kernel_size=1)
        context_masks = dilate_masks(masks, kernel_size=5)
        neighborhood_masks = dilate_masks(masks, kernel_size=9)

        tight_masks = F.interpolate(tight_masks, size=triplets["p2"]["pre"].shape[-2:], mode="bilinear", align_corners=False)
        context_masks = F.interpolate(context_masks, size=triplets["p3"]["pre"].shape[-2:], mode="bilinear", align_corners=False)
        neighborhood_masks = F.interpolate(
            neighborhood_masks,
            size=triplets["p4"]["pre"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        tight_feature = self.tight_projector(self._pool_feature_triplet(triplets["p2"], tight_masks, query_features))
        context_feature = self.context_projector(self._pool_feature_triplet(triplets["p3"], context_masks, query_features))
        neighborhood_feature = self.neighborhood_projector(
            self._pool_feature_triplet(triplets["p4"], neighborhood_masks, query_features)
        )
        return tight_feature, {
            "tight": tight_feature,
            "context": context_feature,
            "neighborhood": neighborhood_feature,
        }

    def _forward_crop_3scale(
        self,
        *,
        pre_pyramid: dict[str, torch.Tensor],
        post_pyramid: dict[str, torch.Tensor],
        query_features: torch.Tensor,
        boxes: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del query_features
        outputs: dict[str, torch.Tensor] = {}
        settings = [
            ("tight", "p2", self.crop_tight),
            ("context", "p3", self.crop_context),
            ("neighborhood", "p4", self.crop_neighborhood),
        ]
        for scale_name, feature_key, encoder in settings:
            scaled_boxes = expand_boxes_cxcywh(boxes, self.scale_factors[scale_name])
            pre_crop = crop_feature_by_boxes(pre_pyramid[feature_key], scaled_boxes, self.crop_feature_size)
            post_crop = crop_feature_by_boxes(post_pyramid[feature_key], scaled_boxes, self.crop_feature_size)
            diff_crop = torch.abs(post_crop - pre_crop)
            crop_feature = torch.cat([pre_crop, post_crop, diff_crop], dim=2)
            batch_size, num_queries, channels, height, width = crop_feature.shape
            encoded = encoder(crop_feature.view(batch_size * num_queries, channels, height, width))
            outputs[scale_name] = encoded.view(batch_size, num_queries, -1)
        return outputs["tight"], outputs

    def _decode_head(self, fused_feature: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, num_queries, hidden_dim = fused_feature.shape
        flat = fused_feature.reshape(batch_size * num_queries, hidden_dim)
        if self.head_type == "corn":
            head_outputs = self.damage_head(flat)
            return {
                "damage_logits": head_outputs["corn_logits"].view(batch_size, num_queries, -1),
                "damage_probabilities": head_outputs["class_probabilities"].view(batch_size, num_queries, -1),
                "damage_pred_labels": head_outputs["pred_labels"].view(batch_size, num_queries),
            }
        if self.head_type == "hierarchical":
            head_outputs = self.damage_head(flat)
            return {
                "damage_binary_logits": head_outputs["damage_binary_logit"].view(batch_size, num_queries),
                "damage_severity_logits": head_outputs["severity_corn_logits"].view(batch_size, num_queries, -1),
                "damage_probabilities": head_outputs["class_probabilities"].view(batch_size, num_queries, -1),
                "damage_pred_labels": head_outputs["pred_labels"].view(batch_size, num_queries),
            }
        logits = self.damage_head(flat).view(batch_size, num_queries, 4)
        probabilities = logits.softmax(dim=-1)
        return {
            "damage_logits": logits,
            "damage_probabilities": probabilities,
            "damage_pred_labels": probabilities.argmax(dim=-1),
        }

    def forward(
        self,
        *,
        pre_pyramid: dict[str, torch.Tensor],
        post_pyramid: dict[str, torch.Tensor],
        query_features: torch.Tensor,
        masks: torch.Tensor,
        boxes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        base_feature = self.base_projector(query_features)
        if self.branch_type == "mask_pool":
            _, scale_outputs = self._forward_mask_pool(
                pre_pyramid=pre_pyramid,
                post_pyramid=post_pyramid,
                query_features=query_features,
                masks=masks,
            )
        elif self.branch_type == "crop_3scale":
            _, scale_outputs = self._forward_crop_3scale(
                pre_pyramid=pre_pyramid,
                post_pyramid=post_pyramid,
                query_features=query_features,
                boxes=boxes,
            )
        else:
            raise ValueError(f"Unsupported damage branch_type='{self.branch_type}'.")

        tight_feature = scale_outputs["tight"]
        context_feature = scale_outputs["context"]
        neighborhood_feature = scale_outputs["neighborhood"]
        gate_input = torch.cat([base_feature, tight_feature, context_feature, neighborhood_feature], dim=-1)
        fusion_weights = torch.softmax(self.gate(gate_input), dim=-1)
        fused_feature = (
            fusion_weights[..., 0:1] * base_feature
            + fusion_weights[..., 1:2] * tight_feature
            + fusion_weights[..., 2:3] * context_feature
            + fusion_weights[..., 3:4] * neighborhood_feature
        )
        head_outputs = self._decode_head(fused_feature)
        head_outputs.update(
            {
                "damage_fused_feature": fused_feature,
                "damage_base_feature": base_feature,
                "damage_fusion_weights": fusion_weights,
                "damage_tight_feature": tight_feature,
                "damage_context_feature": context_feature,
                "damage_neighborhood_feature": neighborhood_feature,
            }
        )
        return head_outputs
