from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.convnext import ConvNeXt
from utils.pretrained import ensure_pretrained_checkpoint, load_convnext_pretrained


_VARIANTS = {
    "convnext_tiny": {"depths": [3, 3, 9, 3], "dims": [96, 192, 384, 768]},
}


class ConvNeXtFeatureEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        pretrained_path: str = "",
        pretrained_url: str = "",
        auto_download_pretrained: bool = True,
        use_4ch_stem: bool = False,
        use_mask_gating: bool = True,
        mask_gate_strength: float = 0.2,
        return_multiscale: bool = True,
        drop_path_rate: float = 0.1,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        if backbone not in _VARIANTS:
            raise ValueError(f"Unsupported ConvNeXt backbone='{backbone}'.")
        variant_cfg = _VARIANTS[backbone]
        in_chans = 4 if use_4ch_stem else 3
        self.backbone_name = backbone
        self.use_4ch_stem = bool(use_4ch_stem)
        self.use_mask_gating = bool(use_mask_gating)
        self.mask_gate_strength = float(mask_gate_strength)
        self.return_multiscale = bool(return_multiscale)
        self.network = ConvNeXt(
            in_chans=in_chans,
            depths=variant_cfg["depths"],
            dims=variant_cfg["dims"],
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
        )
        self.feature_channels = self.network.feature_channels
        self.pretrained_download_report: dict[str, Any] = {
            "path": str(pretrained_path),
            "url": str(pretrained_url),
            "exists": False,
            "download_attempted": False,
            "download_succeeded": False,
            "message": "Not attempted.",
        }
        self.pretrained_load_report: dict[str, Any] = {
            "path": str(pretrained_path),
            "load_attempted": False,
            "load_succeeded": False,
            "loaded_key_count": 0,
            "message": "Not attempted.",
        }

        if pretrained:
            self.pretrained_download_report = ensure_pretrained_checkpoint(
                pretrained_path=pretrained_path,
                pretrained_url=pretrained_url,
                auto_download=auto_download_pretrained,
            )
            self.pretrained_load_report = load_convnext_pretrained(
                self.network,
                pretrained_path=pretrained_path,
                use_4ch_stem=self.use_4ch_stem,
            )

    def get_pretrained_report(self) -> dict[str, Any]:
        return {
            "download": dict(self.pretrained_download_report),
            "load": dict(self.pretrained_load_report),
        }

    def _apply_mask_gate(self, feature: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if not self.use_mask_gating or mask is None:
            return feature
        resized_mask = F.interpolate(mask.float(), size=feature.shape[-2:], mode="nearest")
        return feature * (1.0 + (self.mask_gate_strength * resized_mask))

    def forward(self, image: torch.Tensor, mask: torch.Tensor | None = None) -> OrderedDict[str, torch.Tensor]:
        if self.use_4ch_stem:
            if mask is None:
                raise ValueError("use_4ch_stem=True requires instance_mask.")
            x = torch.cat([image, mask], dim=1)
        else:
            x = image

        features = OrderedDict()
        stage_names = list(self.feature_channels.keys())
        for stage_idx, stage_name in enumerate(stage_names):
            x = self.network.downsample_layers[stage_idx](x)
            x = self.network.stages[stage_idx](x)
            x = self._apply_mask_gate(x, mask)
            features[stage_name] = x

        features["final_feature"] = self.network.output_norm(features["c5"])
        return features

