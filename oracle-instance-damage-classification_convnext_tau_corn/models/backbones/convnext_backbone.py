"""Wrapper around the local ConvNeXt backbone implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from models.backbones.convnext_official_style import ConvNeXt
from utils.checkpoint import ensure_pretrained_checkpoint


VARIANT_CONFIGS: Dict[str, Dict[str, List[int]]] = {
    "tiny": {"depths": [3, 3, 9, 3], "dims": [96, 192, 384, 768]},
    "small": {"depths": [3, 3, 27, 3], "dims": [96, 192, 384, 768]},
    "base": {"depths": [3, 3, 27, 3], "dims": [128, 256, 512, 1024]},
}

_PREFIXES = ("module.", "model.", "backbone.", "encoder.")


class ConvNeXtBackbone(nn.Module):
    """Variant-selectable ConvNeXt backbone with optional pretrained loading."""

    def __init__(
        self,
        variant: str = "tiny",
        pretrained_path: str = "",
        pretrained_url: str = "",
        auto_download_pretrained: bool = False,
        load_pretrained: bool = False,
        freeze_backbone: bool = False,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        return_stages: bool = False,
    ) -> None:
        super().__init__()
        if variant not in VARIANT_CONFIGS:
            raise ValueError(f"Unsupported ConvNeXt variant: {variant}")

        cfg = VARIANT_CONFIGS[variant]
        self.variant = variant
        self.return_stages = return_stages
        self.pretrained_loaded = False
        self.pretrained_path = str(pretrained_path)
        self.pretrained_url = str(pretrained_url)
        self.pretrained_download_report: Dict[str, Any] = {
            "exists": False,
            "download_attempted": False,
            "download_succeeded": False,
            "path": str(pretrained_path),
            "url": str(pretrained_url),
            "filesize": 0,
            "message": "Download not attempted.",
        }
        self.pretrained_report: Dict[str, Any] = {
            "pretrained_path": str(pretrained_path),
            "pretrained_url": str(pretrained_url),
            "download_attempted": False,
            "download_succeeded": False,
            "load_attempted": False,
            "load_succeeded": False,
            "loaded_key_count": 0,
            "skipped_key_count": 0,
            "missing_key_count": 0,
            "unexpected_key_count": 0,
            "skipped_keys_preview": [],
            "missing_keys_preview": [],
            "unexpected_keys_preview": [],
            "frozen_param_count": 0,
            "message": "Pretrained loading not attempted.",
        }

        self.backbone = ConvNeXt(
            depths=cfg["depths"],
            dims=cfg["dims"],
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
        )
        self.out_channels = cfg["dims"][-1]

        if auto_download_pretrained and load_pretrained:
            self.pretrained_download_report = ensure_pretrained_checkpoint(
                pretrained_path=pretrained_path,
                pretrained_url=pretrained_url,
                auto_download=auto_download_pretrained,
                verbose=True,
            )
        elif pretrained_path:
            local_path = Path(pretrained_path).expanduser()
            self.pretrained_download_report = {
                **self.pretrained_download_report,
                "exists": local_path.is_file() and local_path.stat().st_size > 0,
                "filesize": int(local_path.stat().st_size) if local_path.is_file() else 0,
                "message": "Auto download disabled.",
            }

        if load_pretrained:
            self.load_pretrained(pretrained_path)

        if freeze_backbone:
            frozen_param_count = self.freeze()
            print(f"[ConvNeXtBackbone] freeze_backbone=True | frozen_params={frozen_param_count}")

    def load_pretrained(self, pretrained_path: str) -> Dict[str, Any]:
        """Load a local checkpoint with relaxed key matching and detailed reporting."""
        report = {
            "pretrained_path": str(pretrained_path),
            "pretrained_url": self.pretrained_url,
            "download_attempted": bool(self.pretrained_download_report.get("download_attempted", False)),
            "download_succeeded": bool(self.pretrained_download_report.get("download_succeeded", False)),
            "load_attempted": True,
            "load_succeeded": False,
            "loaded_key_count": 0,
            "skipped_key_count": 0,
            "missing_key_count": 0,
            "unexpected_key_count": 0,
            "skipped_keys_preview": [],
            "missing_keys_preview": [],
            "unexpected_keys_preview": [],
            "frozen_param_count": self.pretrained_report.get("frozen_param_count", 0),
            "message": "",
        }

        if not pretrained_path:
            report["message"] = "load_pretrained=True but pretrained_path is empty; using random initialization."
            self.pretrained_loaded = False
            self.pretrained_report = report
            print(f"[ConvNeXtBackbone] {report['message']}")
            return report

        ckpt_path = Path(pretrained_path).expanduser()
        if not ckpt_path.is_file() or ckpt_path.stat().st_size <= 0:
            report["message"] = f"Pretrained checkpoint not found at '{ckpt_path}'; using random initialization."
            self.pretrained_loaded = False
            self.pretrained_report = report
            print(f"[ConvNeXtBackbone] warning: {report['message']}")
            return report

        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = checkpoint
        for candidate_key in ("model", "state_dict", "model_state_dict", "backbone"):
            if isinstance(checkpoint, dict) and candidate_key in checkpoint:
                state_dict = checkpoint[candidate_key]
                break
        if not isinstance(state_dict, dict):
            report["message"] = f"Unsupported checkpoint format at '{ckpt_path}'."
            self.pretrained_loaded = False
            self.pretrained_report = report
            print(f"[ConvNeXtBackbone] warning: {report['message']}")
            return report

        current_state = self.backbone.state_dict()
        loadable_state: Dict[str, torch.Tensor] = {}
        skipped_keys: List[str] = []
        for key, value in state_dict.items():
            clean_key = str(key)
            cleaned = True
            while cleaned:
                cleaned = False
                for prefix in _PREFIXES:
                    if clean_key.startswith(prefix):
                        clean_key = clean_key[len(prefix) :]
                        cleaned = True
            if clean_key.startswith("head."):
                skipped_keys.append(f"{key} -> skipped_head")
                continue
            if clean_key not in current_state:
                skipped_keys.append(f"{key} -> missing_in_model")
                continue
            if tuple(value.shape) != tuple(current_state[clean_key].shape):
                skipped_keys.append(
                    f"{key} -> shape_mismatch{tuple(value.shape)}!={tuple(current_state[clean_key].shape)}"
                )
                continue
            loadable_state[clean_key] = value

        incompatible = self.backbone.load_state_dict(loadable_state, strict=False)
        missing_keys = list(incompatible.missing_keys)
        unexpected_keys = list(incompatible.unexpected_keys)
        self.pretrained_loaded = len(loadable_state) > 0

        report.update(
            {
                "load_succeeded": self.pretrained_loaded,
                "loaded_key_count": len(loadable_state),
                "skipped_key_count": len(skipped_keys),
                "missing_key_count": len(missing_keys),
                "unexpected_key_count": len(unexpected_keys),
                "skipped_keys_preview": skipped_keys[:20],
                "missing_keys_preview": missing_keys[:20],
                "unexpected_keys_preview": unexpected_keys[:20],
                "message": (
                    f"Loaded {len(loadable_state)} keys from '{ckpt_path}'."
                    if self.pretrained_loaded
                    else f"No compatible pretrained keys were loaded from '{ckpt_path}'."
                ),
            }
        )
        self.pretrained_report = report

        print(f"[ConvNeXtBackbone] pretrained_path={ckpt_path}")
        print(
            "[ConvNeXtBackbone] "
            f"download_attempted={report['download_attempted']} "
            f"download_succeeded={report['download_succeeded']} "
            f"load_succeeded={report['load_succeeded']}"
        )
        print(
            "[ConvNeXtBackbone] "
            f"loaded={report['loaded_key_count']} "
            f"skipped={report['skipped_key_count']} "
            f"missing={report['missing_key_count']} "
            f"unexpected={report['unexpected_key_count']}"
        )
        if skipped_keys:
            print(f"[ConvNeXtBackbone] skipped_keys_preview={report['skipped_keys_preview']}")
        return report

    def freeze(self) -> int:
        """Freeze backbone parameters and return the number of frozen parameters."""
        frozen_param_count = 0
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
            frozen_param_count += parameter.numel()
        self.pretrained_report["frozen_param_count"] = frozen_param_count
        return frozen_param_count

    def unfreeze(self) -> int:
        """Unfreeze backbone parameters and return the number of trainable parameters."""
        trainable_param_count = 0
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True
            trainable_param_count += parameter.numel()
        self.pretrained_report["frozen_param_count"] = 0
        return trainable_param_count

    def get_pretrained_report(self) -> Dict[str, Any]:
        """Return a shallow copy of the latest pretrained loading report."""
        report = dict(self.pretrained_report)
        report["download_report"] = dict(self.pretrained_download_report)
        report["pretrained_loaded"] = bool(self.pretrained_loaded)
        return report

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor] | None]:
        return self.backbone(x, return_stages=self.return_stages)
