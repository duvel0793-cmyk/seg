from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from models.vmamba_blocks import (
    ConvDownsample,
    ConvStage,
    ConvStem,
    VMambaStage,
    build_drop_path_rates,
    load_vmamba_pretrained,
)


class HybridConvVMambaEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = "hybrid_vmamba",
        in_channels: int = 4,
        pretrained: bool = True,
        drop_path_rate: float = 0.1,
        vmamba_pretrained_weight_path: str = "",
        conv_stage_depths: tuple[int, int] = (2, 2),
        vmamba_stage_depths: tuple[int, int] = (6, 2),
        dims: tuple[int, int, int, int] = (96, 192, 384, 768),
    ) -> None:
        super().__init__()
        if backbone != "hybrid_vmamba":
            raise ValueError("HybridConvVMambaEncoder only supports backbone='hybrid_vmamba'.")

        c2_dim, c3_dim, c4_dim, c5_dim = [int(dim) for dim in dims]
        stage3_depth, stage4_depth = [int(depth) for depth in vmamba_stage_depths]
        drop_path_chunks = build_drop_path_rates([stage3_depth, stage4_depth], drop_path_rate)

        self.backbone = str(backbone)
        self.stem = ConvStem(in_channels=in_channels, out_channels=c2_dim)
        self.stage1 = ConvStage(channels=c2_dim, depth=int(conv_stage_depths[0]))
        self.downsample_c3 = ConvDownsample(in_channels=c2_dim, out_channels=c3_dim)
        self.stage2 = ConvStage(channels=c3_dim, depth=int(conv_stage_depths[1]))
        self.downsample_c4 = ConvDownsample(in_channels=c3_dim, out_channels=c4_dim)
        self.stage3 = VMambaStage(dim=c4_dim, depth=stage3_depth, drop_path_rates=drop_path_chunks[0], dropout=0.0)
        self.downsample_c5 = ConvDownsample(in_channels=c4_dim, out_channels=c5_dim)
        self.stage4 = VMambaStage(dim=c5_dim, depth=stage4_depth, drop_path_rates=drop_path_chunks[1], dropout=0.0)

        self.feature_channels = OrderedDict(
            [
                ("c2", c2_dim),
                ("c3", c3_dim),
                ("c4", c4_dim),
                ("c5", c5_dim),
            ]
        )
        self.pretrained_load_summary = {
            "requested": bool(pretrained),
            "loaded": False,
            "status": "disabled" if not pretrained else "not_loaded_yet",
            "path": str(vmamba_pretrained_weight_path or ""),
            "loaded_keys": [],
            "missing_keys": list(self.state_dict().keys()),
            "unexpected_keys": [],
            "adapted_keys": [],
            "skipped_shape_mismatch": [],
        }
        if pretrained:
            self.pretrained_load_summary = load_vmamba_pretrained(
                self,
                vmamba_pretrained_weight_path,
                depth_stage3=stage3_depth,
                depth_stage4=stage4_depth,
                verbose=True,
            )

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        c2 = self.stage1(self.stem(x))
        c3 = self.stage2(self.downsample_c3(c2))
        c4 = self.stage3(self.downsample_c4(c3))
        c5 = self.stage4(self.downsample_c5(c4))
        return OrderedDict([("c2", c2), ("c3", c3), ("c4", c4), ("c5", c5)])
