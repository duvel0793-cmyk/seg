"""Oracle instance-level damage classifier with shared ConvNeXt branches."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.convnext_backbone import ConvNeXtBackbone
from models.fusion_heads import MLPFusionHead
from models.tau_corn_head import TauCORNHead


class MaskedGlobalPooling(nn.Module):
    """Masked average pooling over backbone feature maps using oracle instance masks."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, feature_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("Mask must have shape [B, 1, H, W].")

        resized_mask = F.interpolate(mask.float(), size=feature_map.shape[-2:], mode="nearest")
        masked_sum = (feature_map * resized_mask).sum(dim=(2, 3))
        mask_area = resized_mask.sum(dim=(2, 3))
        pooled = masked_sum / mask_area.clamp_min(self.eps)
        global_pool = feature_map.mean(dim=(2, 3))
        valid_mask = (mask_area > self.eps).expand_as(pooled)
        return torch.where(valid_mask, pooled, global_pool)


class ConvNeXtOracleTauCORNModel(nn.Module):
    """Shared-weight pre/post ConvNeXt with masked pooling and tau-scaled CORN head."""

    def __init__(
        self,
        backbone_variant: str = "tiny",
        num_classes: int = 4,
        feature_dim: int = 512,
        fusion_hidden_dim: int = 1024,
        pretrained_path: str = "",
        tau_mode: str = "shared",
        tau_init: float = 1.0,
        tau_min: float = 0.5,
        tau_max: float = 2.0,
        decode_mode: str = "threshold_count",
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        head_dropout: float = 0.1,
        **_: Dict,
    ) -> None:
        super().__init__()
        self.backbone = ConvNeXtBackbone(
            variant=backbone_variant,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            pretrained_path=pretrained_path,
            return_stages=False,
        )
        self.pool = MaskedGlobalPooling()
        pooled_dim = self.backbone.out_channels
        fused_dim = pooled_dim * 4
        self.fusion_head = MLPFusionHead(
            input_dim=fused_dim,
            hidden_dim=fusion_hidden_dim,
            output_dim=feature_dim,
            dropout=head_dropout,
        )
        self.classifier = TauCORNHead(
            input_dim=feature_dim,
            num_classes=num_classes,
            hidden_dim=max(feature_dim // 2, 64),
            tau_mode=tau_mode,
            tau_init=tau_init,
            tau_min=tau_min,
            tau_max=tau_max,
            dropout=head_dropout,
            decode_mode=decode_mode,
        )

    def extract_features(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        feature_map, _ = self.backbone(image)
        return self.pool(feature_map, mask)

    def get_regularization_terms(self) -> Dict[str, torch.Tensor]:
        return self.classifier.regularization_terms()

    def forward(
        self,
        pre_image: torch.Tensor,
        post_image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pre_feature_map, _ = self.backbone(pre_image)
        post_feature_map, _ = self.backbone(post_image)

        pooled_pre = self.pool(pre_feature_map, mask)
        pooled_post = self.pool(post_feature_map, mask)
        delta = pooled_post - pooled_pre

        fused_input = torch.cat([pooled_pre, pooled_post, delta, delta.abs()], dim=1)
        fused_features = self.fusion_head(fused_input)
        head_outputs = self.classifier(fused_features)

        return {
            **head_outputs,
            "features_pre": pooled_pre,
            "features_post": pooled_post,
            "fused_features": fused_features,
        }
