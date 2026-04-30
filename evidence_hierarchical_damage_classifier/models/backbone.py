from __future__ import annotations

from typing import Any

import timm
import torch
import torch.nn as nn

from models.convnextv2 import DEFAULT_CONVNEXTV2_MODEL, load_convnextv2_pretrained


INPUT_MODE_SPECS = {
    "rgb": {"append_mask_channel": False, "branch_in_channels": 3},
    "rgbm": {"append_mask_channel": True, "branch_in_channels": 4},
}


def resolve_input_mode(input_mode: str) -> dict[str, Any]:
    key = str(input_mode).lower()
    if key not in INPUT_MODE_SPECS:
        raise ValueError(f"Unsupported input_mode='{input_mode}'")
    return dict(INPUT_MODE_SPECS[key])


class ConvNeXtV2Backbone(nn.Module):
    def __init__(
        self,
        backbone_name: str = DEFAULT_CONVNEXTV2_MODEL,
        *,
        in_channels: int = 4,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = str(backbone_name)
        self.in_channels = int(in_channels)
        self.pretrained = bool(pretrained)
        self.load_logs: list[str] = []
        self.net = self._build_backbone()
        feature_info = self.net.feature_info.get_dicts()
        self.feature_channels = {
            "c3": int(feature_info[-3]["num_chs"]),
            "c4": int(feature_info[-2]["num_chs"]),
            "c5": int(feature_info[-1]["num_chs"]),
        }

    def _build_backbone(self) -> nn.Module:
        if not timm.models.is_model(self.backbone_name):
            raise RuntimeError(
                f"Backbone '{self.backbone_name}' is not available in timm {timm.__version__}."
            )
        model = timm.create_model(
            self.backbone_name,
            pretrained=False,
            features_only=True,
            out_indices=(1, 2, 3),
            in_chans=self.in_channels,
        )
        if self.pretrained:
            self.load_logs = load_convnextv2_pretrained(model, self.backbone_name)
        return model

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        c3, c4, c5 = self.net(x)
        return {"c3": c3, "c4": c4, "c5": c5}

