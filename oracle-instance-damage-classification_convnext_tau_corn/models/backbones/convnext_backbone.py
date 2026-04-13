"""Wrapper around the local ConvNeXt backbone implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from models.backbones.convnext_official_style import ConvNeXt


VARIANT_CONFIGS: Dict[str, Dict[str, List[int]]] = {
    "tiny": {"depths": [3, 3, 9, 3], "dims": [96, 192, 384, 768]},
    "small": {"depths": [3, 3, 27, 3], "dims": [96, 192, 384, 768]},
    "base": {"depths": [3, 3, 27, 3], "dims": [128, 256, 512, 1024]},
}


class ConvNeXtBackbone(nn.Module):
    """Variant-selectable ConvNeXt backbone with optional local checkpoint loading."""

    def __init__(
        self,
        variant: str = "tiny",
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        pretrained_path: str = "",
        return_stages: bool = False,
    ) -> None:
        super().__init__()
        if variant not in VARIANT_CONFIGS:
            raise ValueError(f"Unsupported ConvNeXt variant: {variant}")

        cfg = VARIANT_CONFIGS[variant]
        self.variant = variant
        self.return_stages = return_stages
        self.backbone = ConvNeXt(
            depths=cfg["depths"],
            dims=cfg["dims"],
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
        )
        self.out_channels = cfg["dims"][-1]

        if pretrained_path:
            self.load_pretrained(pretrained_path)

    def load_pretrained(self, pretrained_path: str) -> None:
        """Load a local checkpoint with relaxed key matching."""
        ckpt_path = Path(pretrained_path)
        if not ckpt_path.is_file():
            print(f"[ConvNeXtBackbone] pretrained checkpoint not found: {pretrained_path}")
            return

        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = checkpoint
        for candidate_key in ("state_dict", "model", "model_state", "backbone"):
            if isinstance(checkpoint, dict) and candidate_key in checkpoint:
                state_dict = checkpoint[candidate_key]
                break

        cleaned_state_dict = {}
        for key, value in state_dict.items():
            clean_key = key
            if clean_key.startswith("module."):
                clean_key = clean_key[len("module.") :]
            if clean_key.startswith("backbone."):
                clean_key = clean_key[len("backbone.") :]
            cleaned_state_dict[clean_key] = value

        missing, unexpected = self.backbone.load_state_dict(cleaned_state_dict, strict=False)
        print(f"[ConvNeXtBackbone] loaded checkpoint: {pretrained_path}")
        print(f"[ConvNeXtBackbone] missing keys: {list(missing)}")
        print(f"[ConvNeXtBackbone] unexpected keys: {list(unexpected)}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor] | None]:
        return self.backbone(x, return_stages=self.return_stages)

