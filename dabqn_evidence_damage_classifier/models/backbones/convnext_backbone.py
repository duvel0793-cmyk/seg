from __future__ import annotations

import timm
import torch
import torch.nn as nn

from models.backbones.convnextv2 import DEFAULT_CONVNEXTV2_MODEL, load_convnextv2_pretrained


class ConvNeXtV2Backbone(nn.Module):
    def __init__(
        self,
        backbone_name: str = DEFAULT_CONVNEXTV2_MODEL,
        *,
        in_channels: int = 3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = str(backbone_name)
        self.in_channels = int(in_channels)
        self.pretrained = bool(pretrained)
        if not timm.models.is_model(self.backbone_name):
            raise RuntimeError(f"Backbone '{self.backbone_name}' is not available in timm {timm.__version__}.")
        self.net = timm.create_model(
            self.backbone_name,
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            in_chans=self.in_channels,
        )
        if self.pretrained:
            load_convnextv2_pretrained(self.net, self.backbone_name)
        feature_info = self.net.feature_info.get_dicts()
        self.feature_channels = {
            "c2": int(feature_info[0]["num_chs"]),
            "c3": int(feature_info[1]["num_chs"]),
            "c4": int(feature_info[2]["num_chs"]),
            "c5": int(feature_info[3]["num_chs"]),
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        c2, c3, c4, c5 = self.net(x)
        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}
