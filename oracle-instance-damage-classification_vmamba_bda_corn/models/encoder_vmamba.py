from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from models.vmamba_basic import VMambaBasicBackbone


class VMambaFeatureEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = "vmamba_tiny",
        in_channels: int = 4,
        pretrained: bool = True,
        drop_path_rate: float = 0.1,
        vmamba_pretrained_weight_path: str = "",
    ) -> None:
        super().__init__()
        self.backbone = VMambaBasicBackbone(
            variant=backbone,
            in_channels=in_channels,
            drop_path_rate=drop_path_rate,
        )
        self.feature_channels = OrderedDict(self.backbone.feature_channels)
        if pretrained:
            self.pretrained_load_summary = self.backbone.load_pretrained_weights(vmamba_pretrained_weight_path)
        else:
            self.pretrained_load_summary = {
                "requested": False,
                "loaded": False,
                "status": "disabled",
                "path": str(vmamba_pretrained_weight_path or ""),
                "loaded_keys": 0,
                "adapted_keys": [],
                "missing_keys": len(self.backbone.state_dict()),
                "unexpected_keys": 0,
                "skipped_shape_mismatch": [],
            }

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        return self.backbone(x)
