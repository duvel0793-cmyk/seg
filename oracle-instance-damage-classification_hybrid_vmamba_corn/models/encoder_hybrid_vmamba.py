from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from models.scan2d_backend import build_deep_stage
from models.vmamba_blocks import (
    ConvDownsample,
    ConvStage,
    ConvStem,
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
        conv_stage_depths: tuple[int, int] | list[int] = (2, 2),
        vmamba_stage_depths: tuple[int, int] | list[int] = (6, 2),
        dims: tuple[int, int, int, int] | list[int] = (96, 192, 384, 768),
        deep_scan_backend: str = "official_ss2d",
        vss_d_state: int = 16,
        vss_d_conv: int = 4,
        vss_expand: int = 2,
    ) -> None:
        super().__init__()
        if backbone != "hybrid_vmamba":
            raise ValueError("HybridConvVMambaEncoder only supports backbone='hybrid_vmamba'.")
        if len(conv_stage_depths) != 2:
            raise ValueError(f"conv_stage_depths must have length 2, got {conv_stage_depths}.")
        if len(vmamba_stage_depths) != 2:
            raise ValueError(f"vmamba_stage_depths must have length 2, got {vmamba_stage_depths}.")
        if len(dims) != 4:
            raise ValueError(f"dims must have length 4, got {dims}.")

        c2_dim, c3_dim, c4_dim, c5_dim = [int(dim) for dim in dims]
        stage3_depth, stage4_depth = [int(depth) for depth in vmamba_stage_depths]
        drop_path_chunks = build_drop_path_rates([stage3_depth, stage4_depth], drop_path_rate)

        self.backbone = str(backbone)
        self.deep_scan_backend = str(deep_scan_backend).lower()

        self.stem = ConvStem(in_channels=in_channels, out_channels=c2_dim)
        self.stage1 = ConvStage(channels=c2_dim, depth=int(conv_stage_depths[0]))

        self.downsample_c3 = ConvDownsample(in_channels=c2_dim, out_channels=c3_dim)
        self.stage2 = ConvStage(channels=c3_dim, depth=int(conv_stage_depths[1]))

        self.downsample_c4 = ConvDownsample(in_channels=c3_dim, out_channels=c4_dim)
        self.stage3 = build_deep_stage(
            backend=self.deep_scan_backend,
            dim=c4_dim,
            depth=stage3_depth,
            drop_path_rates=drop_path_chunks[0],
            dropout=0.0,
            d_state=vss_d_state,
            d_conv=vss_d_conv,
            expand=vss_expand,
        )

        self.downsample_c5 = ConvDownsample(in_channels=c4_dim, out_channels=c5_dim)
        self.stage4 = build_deep_stage(
            backend=self.deep_scan_backend,
            dim=c5_dim,
            depth=stage4_depth,
            drop_path_rates=drop_path_chunks[1],
            dropout=0.0,
            d_state=vss_d_state,
            d_conv=vss_d_conv,
            expand=vss_expand,
        )

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
            "backend": self.deep_scan_backend,
            "loaded_keys": [],
            "missing_keys": list(self.state_dict().keys()),
            "unexpected_keys": [],
            "adapted_keys": [],
            "skipped_shape_mismatch": [],
        }

        if pretrained:
            if self.deep_scan_backend == "legacy":
                self.pretrained_load_summary = load_vmamba_pretrained(
                    self,
                    vmamba_pretrained_weight_path,
                    depth_stage3=stage3_depth,
                    depth_stage4=stage4_depth,
                    verbose=True,
                )
                self.pretrained_load_summary["backend"] = self.deep_scan_backend
            else:
                self.pretrained_load_summary.update(
                    {
                        "loaded": False,
                        "status": "skipped_backend_not_supported_yet",
                    }
                )
                print(
                    "[HybridEncoder] deep_scan_backend='official_ss2d' is enabled, "
                    "but legacy VMamba checkpoint remapping is not compatible yet. "
                    "Skip pretrained loading for now."
                )

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        c2 = self.stage1(self.stem(x))
        c3 = self.stage2(self.downsample_c3(c2))
        c4 = self.stage3(self.downsample_c4(c3))
        c5 = self.stage4(self.downsample_c5(c4))
        return OrderedDict([("c2", c2), ("c3", c3), ("c4", c4), ("c5", c5)])