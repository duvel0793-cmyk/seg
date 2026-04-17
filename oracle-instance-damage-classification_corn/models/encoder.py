from __future__ import annotations

import os
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn

try:
    import timm
except ImportError:  # pragma: no cover
    timm = None

try:
    from torchvision.models import (
        ConvNeXt_Base_Weights,
        ConvNeXt_Small_Weights,
        ConvNeXt_Tiny_Weights,
        ResNet18_Weights,
        convnext_base,
        convnext_small,
        convnext_tiny,
        resnet18,
    )
except ImportError:  # pragma: no cover
    ConvNeXt_Base_Weights = None
    ConvNeXt_Small_Weights = None
    ConvNeXt_Tiny_Weights = None
    ResNet18_Weights = None
    from torchvision.models import convnext_base, convnext_small, convnext_tiny, resnet18


CONVNEXT_SPECS = {
    "convnext_tiny": {
        "builder": convnext_tiny,
        "weights_enum": ConvNeXt_Tiny_Weights,
        "display_name": "ConvNeXt-Tiny",
        "channels": OrderedDict([("c2", 96), ("c3", 192), ("c4", 384), ("c5", 768)]),
    },
    "convnext_small": {
        "builder": convnext_small,
        "weights_enum": ConvNeXt_Small_Weights,
        "display_name": "ConvNeXt-Small",
        "channels": OrderedDict([("c2", 96), ("c3", 192), ("c4", 384), ("c5", 768)]),
    },
    "convnext_base": {
        "builder": convnext_base,
        "weights_enum": ConvNeXt_Base_Weights,
        "display_name": "ConvNeXt-Base",
        "channels": OrderedDict([("c2", 128), ("c3", 256), ("c4", 512), ("c5", 1024)]),
    },
}
CONVNEXTV2_TIMM_SPECS = {
    "convnextv2_tiny": {
        "display_name": "ConvNeXtV2-Tiny",
        "timm_names": [
            "convnextv2_tiny.fcmae_ft_in22k_in1k",
            "convnextv2_tiny.fcmae_ft_in1k",
            "convnextv2_tiny",
        ],
    },
    "convnextv2_base": {
        "display_name": "ConvNeXtV2-Base",
        "timm_names": [
            "convnextv2_base.fcmae_ft_in22k_in1k",
            "convnextv2_base.fcmae_ft_in1k",
            "convnextv2_base",
        ],
    },
}
SUPPORTED_BACKBONES = ["resnet18", *CONVNEXT_SPECS.keys(), *CONVNEXTV2_TIMM_SPECS.keys()]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_WEIGHTS_DIR = PROJECT_ROOT / "weights"
LOCAL_CHECKPOINT_DIR = LOCAL_WEIGHTS_DIR / "checkpoints"


@contextmanager
def _use_local_torch_hub_dir():
    LOCAL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    previous_dir = torch.hub.get_dir()
    previous_torch_home = os.environ.get("TORCH_HOME")
    os.environ["TORCH_HOME"] = str(LOCAL_WEIGHTS_DIR)
    torch.hub.set_dir(str(LOCAL_WEIGHTS_DIR))
    try:
        yield
    finally:
        if previous_torch_home is None:
            os.environ.pop("TORCH_HOME", None)
        else:
            os.environ["TORCH_HOME"] = previous_torch_home
        torch.hub.set_dir(previous_dir)


def _adapt_input_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
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


def _replace_first_conv(module: nn.Module, in_channels: int) -> nn.Module:
    if isinstance(module, nn.Conv2d):
        return _adapt_input_conv(module, in_channels=in_channels)

    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, _adapt_input_conv(child, in_channels=in_channels))
            return module
        updated_child = _replace_first_conv(child, in_channels=in_channels)
        if updated_child is not child:
            setattr(module, name, updated_child)
            return module
    return module


def _find_first_conv(module: nn.Module) -> nn.Conv2d | None:
    if isinstance(module, nn.Conv2d):
        return module
    for child in module.children():
        conv = _find_first_conv(child)
        if conv is not None:
            return conv
    return None


def _load_resnet18(pretrained: bool) -> nn.Module:
    if ResNet18_Weights is not None:
        if pretrained:
            try:
                with _use_local_torch_hub_dir():
                    return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            except Exception:
                return resnet18(weights=None)
        return resnet18(weights=None)
    return resnet18(pretrained=pretrained)


def _load_convnext(backbone: str, pretrained: bool) -> nn.Module:
    if backbone not in CONVNEXT_SPECS:
        raise ValueError(f"Unsupported ConvNeXt backbone='{backbone}'. Supported ConvNeXt backbones: {list(CONVNEXT_SPECS)}")

    spec = CONVNEXT_SPECS[backbone]
    builder = spec["builder"]
    weights_enum = spec["weights_enum"]
    display_name = spec["display_name"]

    if pretrained:
        print(
            f"[INFO] Loading torchvision pretrained {display_name} weights. "
            f"If weights are not cached, torchvision will download them automatically into {LOCAL_CHECKPOINT_DIR}."
        )
    else:
        print(f"[INFO] Building {display_name} without pretrained weights.")

    if weights_enum is not None:
        try:
            weights = weights_enum.IMAGENET1K_V1 if pretrained else None
            with _use_local_torch_hub_dir():
                return builder(weights=weights)
        except Exception as exc:
            if pretrained:
                raise RuntimeError(
                    "Failed to load/download ConvNeXt pretrained weights. "
                    "Check network connection or set pretrained=False."
                ) from exc
            raise

    try:  # pragma: no cover
        with _use_local_torch_hub_dir():
            return builder(pretrained=pretrained)
    except Exception as exc:  # pragma: no cover
        if pretrained:
            raise RuntimeError(
                "Failed to load/download ConvNeXt pretrained weights. "
                "Check network connection or set pretrained=False."
            ) from exc
        raise


def _is_unknown_timm_model_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return ("unknown model" in message) or ("invalid model" in message) or ("not a valid model" in message)


def _load_convnextv2_features(backbone: str, pretrained: bool, in_channels: int) -> tuple[nn.Module, str]:
    if backbone not in CONVNEXTV2_TIMM_SPECS:
        raise ValueError(
            f"Unsupported ConvNeXtV2 backbone='{backbone}'. Supported ConvNeXtV2 backbones: {list(CONVNEXTV2_TIMM_SPECS)}"
        )
    if timm is None:
        raise RuntimeError(
            "convnextv2 backbone requires timm. Please install timm or switch backbone to an existing supported option."
        )

    spec = CONVNEXTV2_TIMM_SPECS[backbone]
    display_name = spec["display_name"]
    candidate_names = list(spec["timm_names"])

    if pretrained:
        print(
            f"[INFO] Loading timm pretrained {display_name} weights. "
            f"If weights are not cached, timm will download them automatically into {LOCAL_CHECKPOINT_DIR}."
        )
    else:
        print(f"[INFO] Building {display_name} without pretrained weights.")

    extractor: nn.Module | None = None
    resolved_name: str | None = None
    non_unknown_error: Exception | None = None

    with _use_local_torch_hub_dir():
        for timm_name in candidate_names:
            try:
                extractor = timm.create_model(
                    timm_name,
                    pretrained=pretrained,
                    features_only=True,
                    out_indices=(0, 1, 2, 3),
                    in_chans=3,
                )
                resolved_name = timm_name
                break
            except Exception as exc:
                if _is_unknown_timm_model_error(exc):
                    continue
                non_unknown_error = exc
                # Keep trying other variants, but fail loudly afterwards if none work.
                continue

    if extractor is None or resolved_name is None:
        if non_unknown_error is not None:
            if pretrained:
                raise RuntimeError(
                    "Failed to load/download ConvNeXtV2 pretrained weights. "
                    "Check network connection or set pretrained=False."
                ) from non_unknown_error
            raise RuntimeError("Failed to build ConvNeXtV2 feature extractor.") from non_unknown_error
        raise RuntimeError(
            f"No available timm model variant was found for '{backbone}'. Tried: {candidate_names}."
        )

    if in_channels != 3:
        extractor = _replace_first_conv(extractor, in_channels=in_channels)
    stem_conv = _find_first_conv(extractor)
    if stem_conv is None or stem_conv.in_channels != in_channels:
        raise RuntimeError(f"Failed to adapt ConvNeXtV2 stem to in_channels={in_channels}.")
    return extractor, resolved_name


class ResNetFeatureEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet18", in_channels: int = 4, pretrained: bool = True) -> None:
        super().__init__()
        if backbone != "resnet18":
            raise ValueError("Unsupported ResNet backbone='{}'. Supported ResNet backbones: ['resnet18']".format(backbone))

        backbone_model = _load_resnet18(pretrained=pretrained)
        self.stem = nn.Sequential(
            _adapt_input_conv(backbone_model.conv1, in_channels=in_channels),
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

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return OrderedDict([("c2", c2), ("c3", c3), ("c4", c4), ("c5", c5)])


class ConvNeXtFeatureEncoder(nn.Module):
    def __init__(self, backbone: str = "convnext_tiny", in_channels: int = 4, pretrained: bool = True) -> None:
        super().__init__()
        if backbone not in CONVNEXT_SPECS:
            raise ValueError(f"Unsupported ConvNeXt backbone='{backbone}'. Supported ConvNeXt backbones: {list(CONVNEXT_SPECS)}")

        backbone_model = _load_convnext(backbone=backbone, pretrained=pretrained)
        self.stem = _replace_first_conv(backbone_model.features[0], in_channels=in_channels)
        stem_conv = _find_first_conv(self.stem)
        if stem_conv is None or stem_conv.in_channels != in_channels:
            raise RuntimeError(f"Failed to adapt ConvNeXt stem to in_channels={in_channels}.")
        self.stage1 = backbone_model.features[1]
        self.downsample1 = backbone_model.features[2]
        self.stage2 = backbone_model.features[3]
        self.downsample2 = backbone_model.features[4]
        self.stage3 = backbone_model.features[5]
        self.downsample3 = backbone_model.features[6]
        self.stage4 = backbone_model.features[7]

        self.feature_channels = OrderedDict(CONVNEXT_SPECS[backbone]["channels"])

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        x = self.stem(x)
        c2 = self.stage1(x)
        x = self.downsample1(c2)
        c3 = self.stage2(x)
        x = self.downsample2(c3)
        c4 = self.stage3(x)
        x = self.downsample3(c4)
        c5 = self.stage4(x)
        return OrderedDict([("c2", c2), ("c3", c3), ("c4", c4), ("c5", c5)])


class ConvNeXtV2FeatureEncoder(nn.Module):
    def __init__(self, backbone: str = "convnextv2_tiny", in_channels: int = 4, pretrained: bool = True) -> None:
        super().__init__()
        self.extractor, resolved_name = _load_convnextv2_features(
            backbone=backbone,
            pretrained=pretrained,
            in_channels=in_channels,
        )
        channels = [int(channel) for channel in self.extractor.feature_info.channels()]
        if len(channels) < 4:
            raise RuntimeError(
                f"ConvNeXtV2 features_only must expose at least 4 stages; got channels={channels}."
            )
        self.feature_channels = OrderedDict(
            zip(("c2", "c3", "c4", "c5"), channels[:4], strict=True)
        )
        print(
            f"[INFO] ConvNeXtV2 features_only encoder resolved to timm='{resolved_name}' "
            f"with channels={list(self.feature_channels.values())}."
        )

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        features = self.extractor(x)
        if len(features) < 4:
            raise RuntimeError(f"ConvNeXtV2 features_only forward returned {len(features)} levels, expected >=4.")
        c2, c3, c4, c5 = features[:4]
        return OrderedDict([("c2", c2), ("c3", c3), ("c4", c4), ("c5", c5)])


def build_feature_encoder(backbone: str, in_channels: int = 4, pretrained: bool = True) -> nn.Module:
    if backbone.startswith("resnet"):
        return ResNetFeatureEncoder(backbone=backbone, in_channels=in_channels, pretrained=pretrained)
    if backbone in CONVNEXTV2_TIMM_SPECS:
        return ConvNeXtV2FeatureEncoder(backbone=backbone, in_channels=in_channels, pretrained=pretrained)
    if backbone.startswith("convnext"):
        return ConvNeXtFeatureEncoder(backbone=backbone, in_channels=in_channels, pretrained=pretrained)
    raise ValueError(f"Unsupported backbone='{backbone}'. Supported backbones: {SUPPORTED_BACKBONES}")
