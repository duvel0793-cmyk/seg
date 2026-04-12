from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


def _load_resnet18(pretrained: bool) -> models.ResNet:
    if pretrained:
        try:
            return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except AttributeError:
            return models.resnet18(pretrained=True)
    try:
        return models.resnet18(weights=None)
    except TypeError:
        return models.resnet18(pretrained=False)


def _adapt_input_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    if conv.in_channels == in_channels:
        return conv

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )

    with torch.no_grad():
        if in_channels == 6:
            new_conv.weight[:, :3] = conv.weight
            new_conv.weight[:, 3:6] = conv.weight
        elif in_channels == 7:
            new_conv.weight[:, :3] = conv.weight
            new_conv.weight[:, 3:6] = conv.weight
            new_conv.weight[:, 6:7] = conv.weight.mean(dim=1, keepdim=True)
        else:
            repeats = min(in_channels, 3)
            new_conv.weight[:, :repeats] = conv.weight[:, :repeats]
            for index in range(repeats, in_channels):
                new_conv.weight[:, index : index + 1] = conv.weight.mean(dim=1, keepdim=True)

        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias)

    return new_conv


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class FPNDecoder(nn.Module):
    def __init__(self, decoder_channels: int) -> None:
        super().__init__()
        self.lateral2 = nn.Conv2d(64, decoder_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(128, decoder_channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(256, decoder_channels, kernel_size=1)
        self.lateral5 = nn.Conv2d(512, decoder_channels, kernel_size=1)

        self.smooth2 = ConvBNReLU(decoder_channels, decoder_channels)
        self.smooth3 = ConvBNReLU(decoder_channels, decoder_channels)
        self.smooth4 = ConvBNReLU(decoder_channels, decoder_channels)
        self.smooth5 = ConvBNReLU(decoder_channels, decoder_channels)

        self.fusion = nn.Sequential(
            ConvBNReLU(decoder_channels * 4, decoder_channels),
            ConvBNReLU(decoder_channels, decoder_channels),
        )

    def forward(
        self,
        c2: torch.Tensor,
        c3: torch.Tensor,
        c4: torch.Tensor,
        c5: torch.Tensor,
    ) -> torch.Tensor:
        p5 = self.smooth5(self.lateral5(c5))
        p4 = self.smooth4(
            self.lateral4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        )
        p3 = self.smooth3(
            self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        )
        p2 = self.smooth2(
            self.lateral2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        )

        target_size = p2.shape[-2:]
        fused = torch.cat(
            [
                p2,
                F.interpolate(p3, size=target_size, mode="bilinear", align_corners=False),
                F.interpolate(p4, size=target_size, mode="bilinear", align_corners=False),
                F.interpolate(p5, size=target_size, mode="bilinear", align_corners=False),
            ],
            dim=1,
        )
        return self.fusion(fused)


class ResNet18FPNBDA(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        decoder_channels: int = 128,
        num_classes: int = 5,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        backbone = _load_resnet18(pretrained=pretrained)
        backbone.conv1 = _adapt_input_conv(backbone.conv1, in_channels=in_channels)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.decoder = FPNDecoder(decoder_channels=decoder_channels)
        self.damage_head = nn.Sequential(
            ConvBNReLU(decoder_channels, decoder_channels),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )
        self.localization_head = nn.Sequential(
            ConvBNReLU(decoder_channels, decoder_channels // 2),
            nn.Conv2d(decoder_channels // 2, 1, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = inputs.shape[-2:]

        x = self.stem(inputs)
        x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        features = self.decoder(c2, c3, c4, c5)
        damage_logits = self.damage_head(features)
        loc_logits = self.localization_head(features)

        damage_logits = F.interpolate(
            damage_logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        loc_logits = F.interpolate(
            loc_logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        return {"damage_logits": damage_logits, "loc_logits": loc_logits}


def build_resnet18_fpn_bda(model_config: dict[str, Any]) -> ResNet18FPNBDA:
    return ResNet18FPNBDA(
        in_channels=int(model_config["in_channels"]),
        decoder_channels=int(model_config["decoder_channels"]),
        pretrained=bool(model_config.get("pretrained", True)),
    )
