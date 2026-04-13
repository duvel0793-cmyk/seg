from __future__ import annotations

import torch
import torch.nn as nn

from models.encoder_factory import build_encoder
from models.heads import InstanceClassifierHead
from models.pooling import MaskedMultiScalePooling


class PostOnlyDamageClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.2,
        drop_path_rate: float = 0.1,
        vmamba_pretrained_weight_path: str = "",
    ) -> None:
        super().__init__()
        self.encoder = build_encoder(
            backbone,
            in_channels=4,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            vmamba_pretrained_weight_path=vmamba_pretrained_weight_path,
        )
        self.pool = MaskedMultiScalePooling(pool_modes=("avg", "max"))
        deepest_channels = self.encoder.feature_channels["c5"]
        self.classifier = InstanceClassifierHead(
            in_features=deepest_channels * 2,
            hidden_features=max(256, deepest_channels),
            dropout=dropout,
            num_classes=4,
        )

    def forward(self, pre_image: torch.Tensor, post_image: torch.Tensor, instance_mask: torch.Tensor) -> torch.Tensor:
        del pre_image
        post_input = torch.cat([post_image, instance_mask], dim=1)
        features = self.encoder(post_input)
        pooled = self.pool([features["c5"]], instance_mask)
        return self.classifier(pooled)


class SiameseSimpleDamageClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.2,
        drop_path_rate: float = 0.1,
        vmamba_pretrained_weight_path: str = "",
    ) -> None:
        super().__init__()
        self.encoder = build_encoder(
            backbone,
            in_channels=4,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            vmamba_pretrained_weight_path=vmamba_pretrained_weight_path,
        )
        deepest_channels = self.encoder.feature_channels["c5"]
        self.deep_fusion = nn.Sequential(
            nn.Conv2d(deepest_channels * 2, deepest_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(deepest_channels),
            nn.GELU(),
        )
        self.pool = MaskedMultiScalePooling(pool_modes=("avg", "max"))
        self.classifier = InstanceClassifierHead(
            in_features=deepest_channels * 2,
            hidden_features=max(256, deepest_channels),
            dropout=dropout,
            num_classes=4,
        )

    def forward(self, pre_image: torch.Tensor, post_image: torch.Tensor, instance_mask: torch.Tensor) -> torch.Tensor:
        pre_input = torch.cat([pre_image, instance_mask], dim=1)
        post_input = torch.cat([post_image, instance_mask], dim=1)
        pre_feat = self.encoder(pre_input)["c5"]
        post_feat = self.encoder(post_input)["c5"]
        fused = self.deep_fusion(torch.cat([pre_feat, post_feat], dim=1))
        pooled = self.pool([fused], instance_mask)
        return self.classifier(pooled)
