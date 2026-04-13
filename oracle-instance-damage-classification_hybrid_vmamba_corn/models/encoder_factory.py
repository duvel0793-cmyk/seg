from __future__ import annotations

import torch.nn as nn

from models.encoder_hybrid_vmamba import HybridConvVMambaEncoder
from models.encoder_resnet import ResNetFeatureEncoder


def build_encoder(
    backbone: str,
    *,
    in_channels: int = 4,
    pretrained: bool = True,
    drop_path_rate: float = 0.1,
    vmamba_pretrained_weight_path: str = "",
) -> nn.Module:
    backbone = str(backbone)
    if backbone == "resnet18":
        return ResNetFeatureEncoder(
            backbone=backbone,
            in_channels=in_channels,
            pretrained=pretrained,
        )
    if backbone == "hybrid_vmamba":
        return HybridConvVMambaEncoder(
            backbone=backbone,
            in_channels=in_channels,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            vmamba_pretrained_weight_path=vmamba_pretrained_weight_path,
        )
    raise ValueError(f"Unsupported backbone='{backbone}'.")
