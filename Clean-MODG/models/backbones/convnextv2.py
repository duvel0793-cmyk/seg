"""ConvNeXtV2/ConvNeXt backbone wrappers built on timm."""

from __future__ import annotations

from pathlib import Path
import warnings

import timm
import torch
import torch.nn as nn


def _candidate_names(name: str) -> list[str]:
    if name == "convnextv2_tiny":
        return [
            "convnextv2_tiny.fcmae_ft_in22k_in1k",
            "convnextv2_tiny.fcmae",
            "convnextv2_tiny",
            "convnext_tiny",
        ]
    return [name, "convnext_tiny"]


def _default_local_weight_path(name: str) -> Path | None:
    project_root = Path(__file__).resolve().parents[2]
    mapping = {
        "convnextv2_tiny": project_root / "weights" / "convnextv2_tiny_22k_224_ema.pt",
        "convnextv2_tiny.fcmae_ft_in22k_in1k": project_root / "weights" / "convnextv2_tiny_22k_224_ema.pt",
        "convnextv2_tiny.fcmae": project_root / "weights" / "convnextv2_tiny_22k_224_ema.pt",
    }
    path = mapping.get(name)
    return path if path is not None and path.exists() else None


class TimmBackbone(nn.Module):
    def __init__(self, name: str, pretrained: bool = True, in_chans: int = 3, checkpoint_path: str | Path | None = None) -> None:
        super().__init__()
        self.requested_name = name
        self.model_name = ""
        self.model: nn.Module | None = None
        self.feature_dim = 0
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve() if checkpoint_path else None
        self._using_auto_checkpoint = False

        default_local_weight = _default_local_weight_path(name)
        if self.checkpoint_path is None and pretrained and default_local_weight is not None:
            self.checkpoint_path = default_local_weight
            self._using_auto_checkpoint = True

        errors: list[str] = []
        for candidate in _candidate_names(name):
            try:
                pretrained_cfg_overlay = None
                use_builtin_pretrained = pretrained
                if pretrained and self.checkpoint_path is not None:
                    if not self._using_auto_checkpoint or "convnextv2_tiny" in candidate:
                        pretrained_cfg_overlay = {"file": str(self.checkpoint_path)}
                model = timm.create_model(
                    candidate,
                    pretrained=use_builtin_pretrained,
                    pretrained_cfg_overlay=pretrained_cfg_overlay,
                    in_chans=in_chans,
                    num_classes=0,
                    global_pool="",
                )
                self.model = model
                self.model_name = candidate
                self.feature_dim = self._infer_feature_dim(model)
                if candidate != name:
                    warnings.warn(
                        f"Backbone '{name}' is unavailable. Falling back to '{candidate}'.",
                        stacklevel=2,
                    )
                break
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
        if self.model is None:
            message = "\n".join(errors)
            raise RuntimeError(f"Unable to build timm backbone for '{name}'. Tried:\n{message}")

    @staticmethod
    def _infer_feature_dim(model: nn.Module) -> int:
        if hasattr(model, "num_features"):
            return int(getattr(model, "num_features"))
        if hasattr(model, "feature_info"):
            try:
                channels = model.feature_info.channels()
                if channels:
                    return int(channels[-1])
            except Exception:
                pass
        raise AttributeError("Could not infer feature dim from timm backbone.")

    def get_feature_dim(self) -> int:
        return self.feature_dim

    def _to_nchw(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4:
            return feature
        if feature.shape[-1] == self.feature_dim and feature.shape[1] != self.feature_dim:
            feature = feature.permute(0, 3, 1, 2).contiguous()
        return feature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        if hasattr(self.model, "forward_features"):
            feature = self.model.forward_features(x)
        else:
            feature = self.model(x)
        if isinstance(feature, (list, tuple)):
            feature = feature[-1]
        return self._to_nchw(feature)


def build_backbone(
    name: str,
    pretrained: bool = True,
    in_chans: int = 3,
    checkpoint_path: str | Path | None = None,
) -> TimmBackbone:
    return TimmBackbone(name=name, pretrained=pretrained, in_chans=in_chans, checkpoint_path=checkpoint_path)
