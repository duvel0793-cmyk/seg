from __future__ import annotations

import torch
import torch.nn as nn

from models.encoder import build_feature_encoder
from models.heads import InstanceClassifierHead
from models.pooling import MaskedMultiScalePooling


class PostOnlyDamageClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "convnextv2_tiny",
        pretrained: bool = True,
        dropout: float = 0.2,
        encoder_in_channels: int = 4,
    ) -> None:
        super().__init__()
        self.encoder_in_channels = int(encoder_in_channels)
        self.encoder = build_feature_encoder(
            backbone=backbone,
            in_channels=self.encoder_in_channels,
            pretrained=pretrained,
        )
        self.pool = MaskedMultiScalePooling()
        deepest_channels = self.encoder.feature_channels["c5"]
        self.classifier = InstanceClassifierHead(
            in_features=deepest_channels * 2,
            hidden_features=max(256, deepest_channels),
            dropout=dropout,
            num_classes=4,
        )

    def _compose_input(
        self,
        image: torch.Tensor,
        instance_mask: torch.Tensor,
        instance_boundary: torch.Tensor | None,
    ) -> torch.Tensor:
        channels = [image, instance_mask]
        if self.encoder_in_channels >= 5:
            if instance_boundary is None:
                instance_boundary = torch.zeros_like(instance_mask)
            channels.append(instance_boundary)
        return torch.cat(channels, dim=1)

    def forward(
        self,
        pre_image: torch.Tensor,
        post_image: torch.Tensor,
        instance_mask: torch.Tensor,
        instance_boundary: torch.Tensor | None = None,
        context_pre_image: torch.Tensor | None = None,
        context_post_image: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        context_boundary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del pre_image
        del context_pre_image, context_post_image, context_mask, context_boundary
        post_input = self._compose_input(post_image, instance_mask, instance_boundary)
        features = self.encoder(post_input)
        pooled = self.pool([features["c5"]], instance_mask)
        return self.classifier(pooled)


class SiameseSimpleDamageClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "convnextv2_tiny",
        pretrained: bool = True,
        dropout: float = 0.2,
        encoder_in_channels: int = 4,
    ) -> None:
        super().__init__()
        self.encoder_in_channels = int(encoder_in_channels)
        self.encoder = build_feature_encoder(
            backbone=backbone,
            in_channels=self.encoder_in_channels,
            pretrained=pretrained,
        )
        deepest_channels = self.encoder.feature_channels["c5"]
        self.deep_fusion = nn.Sequential(
            nn.Conv2d(deepest_channels * 2, deepest_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(deepest_channels),
            nn.GELU(),
        )
        self.pool = MaskedMultiScalePooling()
        self.classifier = InstanceClassifierHead(
            in_features=deepest_channels * 2,
            hidden_features=max(256, deepest_channels),
            dropout=dropout,
            num_classes=4,
        )

    def _compose_input(
        self,
        image: torch.Tensor,
        instance_mask: torch.Tensor,
        instance_boundary: torch.Tensor | None,
    ) -> torch.Tensor:
        channels = [image, instance_mask]
        if self.encoder_in_channels >= 5:
            if instance_boundary is None:
                instance_boundary = torch.zeros_like(instance_mask)
            channels.append(instance_boundary)
        return torch.cat(channels, dim=1)

    def forward(
        self,
        pre_image: torch.Tensor,
        post_image: torch.Tensor,
        instance_mask: torch.Tensor,
        instance_boundary: torch.Tensor | None = None,
        context_pre_image: torch.Tensor | None = None,
        context_post_image: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        context_boundary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del context_pre_image, context_post_image, context_mask, context_boundary
        pre_input = self._compose_input(pre_image, instance_mask, instance_boundary)
        post_input = self._compose_input(post_image, instance_mask, instance_boundary)
        pre_feat = self.encoder(pre_input)["c5"]
        post_feat = self.encoder(post_input)["c5"]
        fused = self.deep_fusion(torch.cat([pre_feat, post_feat], dim=1))
        pooled = self.pool([fused], instance_mask)
        return self.classifier(pooled)
