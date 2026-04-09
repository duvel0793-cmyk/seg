from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

try:
    from torchvision.models import ResNet18_Weights, resnet18
except ImportError:  # pragma: no cover
    ResNet18_Weights = None
    from torchvision.models import resnet18


def _load_resnet18(pretrained: bool) -> nn.Module:
    if ResNet18_Weights is not None:
        if pretrained:
            try:
                return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            except Exception:
                return resnet18(weights=None)
        return resnet18(weights=None)
    return resnet18(pretrained=pretrained)


class ResNetFeatureEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet18", in_channels: int = 4, pretrained: bool = True) -> None:
        super().__init__()
        if backbone != "resnet18":
            raise ValueError("Current implementation only supports resnet18.")

        backbone_model = _load_resnet18(pretrained=pretrained)
        self.stem = nn.Sequential(
            self._patch_first_conv(backbone_model.conv1, in_channels=in_channels),
            backbone_model.bn1,
            backbone_model.relu,
            backbone_model.maxpool,
        )
        self.layer1 = backbone_model.layer1
        self.layer2 = backbone_model.layer2
        self.layer3 = backbone_model.layer3
        self.layer4 = backbone_model.layer4

        self.feature_channels = OrderedDict(
            [
                ("c2", 64),
                ("c3", 128),
                ("c4", 256),
                ("c5", 512),
            ]
        )

    @staticmethod
    def _patch_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
        new_conv = nn.Conv2d(
            in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=False,
        )
        with torch.no_grad():
            if in_channels == conv.in_channels:
                new_conv.weight.copy_(conv.weight)
            elif in_channels > conv.in_channels:
                new_conv.weight[:, : conv.in_channels].copy_(conv.weight)
                extra = conv.weight.mean(dim=1, keepdim=True)
                for idx in range(conv.in_channels, in_channels):
                    new_conv.weight[:, idx : idx + 1].copy_(extra)
            else:
                new_conv.weight.copy_(conv.weight[:, :in_channels])
        return new_conv

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return OrderedDict([("c2", c2), ("c3", c3), ("c4", c4), ("c5", c5)])
