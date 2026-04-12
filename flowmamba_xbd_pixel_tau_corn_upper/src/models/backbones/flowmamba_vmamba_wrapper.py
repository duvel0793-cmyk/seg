"""FlowMamba / VMamba-inspired backbone wrapper with explicit backend metadata."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SharedConvEncoder(nn.Module):
    """Fallback shared encoder used when upstream repos cannot be imported."""

    def __init__(self, channels: List[int]) -> None:
        super().__init__()
        in_channels = 3
        self.stages = nn.ModuleList()
        for out_channels in channels:
            self.stages.append(ConvStage(in_channels, out_channels))
            in_channels = out_channels
        self.dims = channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class FlowMambaVMambaWrapper(nn.Module):
    """Unify local FlowMamba / VMamba style backbones behind one interface."""

    def __init__(self, model_cfg: Dict[str, object]) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.requested_backend = str(model_cfg.get("backend", model_cfg.get("backbone_backend", "auto"))).lower()
        self.fail_if_fallback = bool(model_cfg.get("fail_if_fallback", False))
        self.pretrained_required = bool(model_cfg.get("pretrained_required", False))
        self.pretrained_path = str(model_cfg.get("pretrained_path", "") or "")
        self.pretrained_key = str(model_cfg.get("pretrained_key", "model"))
        self.pretrained_strict = bool(model_cfg.get("pretrained_strict", False))

        self.encoder: nn.Module | None = None
        self.out_channels: List[int] = []
        self.feature_strides: List[List[int]] = []
        self.feature_shapes: List[List[int]] = []
        self.metadata: Dict[str, Any] = {
            "requested_backend": self.requested_backend,
            "backend_name": "uninitialized",
            "backend_reason": "",
            "pretrained_path": self.pretrained_path,
            "pretrained_key": self.pretrained_key,
            "pretrained_loaded": False,
            "pretrained_strict": self.pretrained_strict,
            "pretrained_report": {},
            "feature_channels": [],
            "feature_strides": [],
            "feature_shapes": [],
        }

        self._build_encoder()
        self.backend_name = str(self.metadata["backend_name"])
        self.backend_reason = str(self.metadata["backend_reason"])
        self.fuse_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch * 3, ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(ch),
                    nn.GELU(),
                )
                for ch in self.out_channels
            ]
        )

    def _build_encoder(self) -> None:
        fallback_channels = list(self.model_cfg.get("fallback", {}).get("channels", [32, 64, 128, 256]))
        attempts: List[Tuple[str, Any]] = []
        build_errors: List[str] = []

        if self.requested_backend == "fallback":
            self._use_fallback(fallback_channels, reason="Requested fallback backend.")
            return
        if self.requested_backend == "flowmamba":
            attempts = [("flowmamba", self._try_build_flowmamba)]
        elif self.requested_backend == "vmamba":
            attempts = [("vmamba", self._try_build_vmamba)]
        elif self.requested_backend == "auto":
            attempts = [("flowmamba", self._try_build_flowmamba), ("vmamba", self._try_build_vmamba)]
        else:
            raise ValueError(f"Unsupported model.backend value: {self.requested_backend}")

        for backend_name, builder in attempts:
            build_info = builder()
            if build_info["encoder"] is not None:
                self.encoder = build_info["encoder"]
                self.out_channels = list(build_info["feature_channels"])
                self.metadata.update(
                    {
                        "backend_name": backend_name,
                        "backend_reason": build_info["reason"],
                        "repo_path": build_info["repo_path"],
                        "feature_channels": self.out_channels,
                        "pretrained_loaded": build_info["pretrained_report"].get("loaded", False),
                        "pretrained_report": build_info["pretrained_report"],
                        "matched_pretrained_keys": build_info["pretrained_report"].get("matched_keys", 0),
                        "missing_pretrained_keys": build_info["pretrained_report"].get("missing_keys", 0),
                        "unexpected_pretrained_keys": build_info["pretrained_report"].get("unexpected_keys", 0),
                        "shape_mismatched_pretrained_keys": build_info["pretrained_report"].get("shape_mismatched_keys", 0),
                    }
                )
                self.backend_name = str(self.metadata["backend_name"])
                self.backend_reason = str(self.metadata["backend_reason"])
                if self.pretrained_required and not self.metadata["pretrained_loaded"]:
                    raise RuntimeError(
                        f"{backend_name} backend initialized but pretrained weights were required and not loaded: "
                        f"{self.metadata['pretrained_report'].get('reason', 'unknown')}"
                    )
                return
            build_errors.append(f"{backend_name}: {build_info['reason']}")

        if self.fail_if_fallback:
            raise RuntimeError(
                f"Failed to initialize requested backend={self.requested_backend!r} without fallback. "
                + " | ".join(build_errors)
            )
        self._use_fallback(
            fallback_channels,
            reason="All external backends unavailable; using internal Conv encoder. " + " | ".join(build_errors),
        )

    def _use_fallback(self, fallback_channels: List[int], reason: str) -> None:
        self.encoder = SharedConvEncoder(fallback_channels)
        self.out_channels = list(fallback_channels)
        self.metadata.update(
            {
                "backend_name": "fallback",
                "backend_reason": reason,
                "feature_channels": self.out_channels,
                "pretrained_loaded": False,
                "pretrained_report": {"loaded": False, "reason": "fallback encoder has no upstream pretrained weights"},
            }
        )
        self.backend_name = str(self.metadata["backend_name"])
        self.backend_reason = str(self.metadata["backend_reason"])

    def _try_build_flowmamba(self) -> Dict[str, Any]:
        repo = Path(str(self.model_cfg.get("flowmamba_repo", ""))).expanduser()
        if not repo.exists():
            return self._failed_build(f"FlowMamba repo not found: {repo}", repo)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        try:
            from changedetection.models.Mamba_backbone import Backbone_VSSM  # type: ignore
        except Exception as exc:
            return self._failed_build(f"FlowMamba import failed: {exc}", repo)

        kwargs = self._normalize_backbone_kwargs(dict(self.model_cfg.get("backbone_kwargs", {})))
        try:
            encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=None, **kwargs)
        except Exception as exc:
            return self._failed_build(f"FlowMamba init failed: {exc}", repo)

        feature_channels = self._normalize_feature_channels(getattr(encoder, "dims", None))
        pretrained_report = self._load_pretrained_weights(encoder)
        return {
            "encoder": encoder,
            "feature_channels": feature_channels,
            "reason": f"Loaded Backbone_VSSM from {repo}",
            "repo_path": str(repo),
            "pretrained_report": pretrained_report,
        }

    def _try_build_vmamba(self) -> Dict[str, Any]:
        repo = Path(str(self.model_cfg.get("vmamba_repo", ""))).expanduser()
        if not repo.exists():
            return self._failed_build(f"VMamba repo not found: {repo}", repo)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        try:
            from vmamba import LayerNorm, VSSM  # type: ignore
        except Exception as exc:
            return self._failed_build(f"VMamba import failed: {exc}", repo)

        kwargs = self._normalize_backbone_kwargs(dict(self.model_cfg.get("backbone_kwargs", {})))

        class VMambaBackbone(VSSM):
            def __init__(self, out_indices=(0, 1, 2, 3), norm_layer="ln2d", **inner_kwargs):
                inner_kwargs.update(norm_layer=norm_layer)
                super().__init__(**inner_kwargs)
                self.channel_first = norm_layer.lower() in {"bn", "ln2d"}
                self.out_indices = out_indices
                norm_cls = {
                    "ln": lambda dim: nn.LayerNorm(dim),
                    "ln2d": lambda dim: LayerNorm(dim, channel_first=True),
                    "bn": lambda dim: nn.BatchNorm2d(dim),
                }[norm_layer.lower()]
                for idx in out_indices:
                    self.add_module(f"outnorm{idx}", norm_cls(self.dims[idx]))
                del self.classifier

            def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
                x = self.patch_embed(x)
                if getattr(self, "pos_embed", None) is not None:
                    pos_embed = self.pos_embed.permute(0, 2, 3, 1) if not self.channel_first else self.pos_embed
                    x = x + pos_embed
                outs = []
                for idx, layer in enumerate(self.layers):
                    hidden = layer.blocks(x)
                    x = layer.downsample(hidden)
                    if idx in self.out_indices:
                        norm = getattr(self, f"outnorm{idx}")
                        hidden = norm(hidden)
                        if not self.channel_first:
                            hidden = hidden.permute(0, 3, 1, 2).contiguous()
                        outs.append(hidden)
                return outs

        try:
            encoder = VMambaBackbone(**kwargs)
        except Exception as exc:
            return self._failed_build(f"VMamba init failed: {exc}", repo)

        feature_channels = self._normalize_feature_channels(getattr(encoder, "dims", None))
        pretrained_report = self._load_pretrained_weights(encoder)
        return {
            "encoder": encoder,
            "feature_channels": feature_channels,
            "reason": f"Loaded VSSM from {repo}",
            "repo_path": str(repo),
            "pretrained_report": pretrained_report,
        }

    def _load_pretrained_weights(self, encoder: nn.Module) -> Dict[str, Any]:
        if not self.pretrained_path:
            return {"loaded": False, "reason": "pretrained_path is empty"}

        pretrained_path = Path(self.pretrained_path).expanduser().resolve()
        if not pretrained_path.exists():
            return {"loaded": False, "reason": f"pretrained checkpoint not found: {pretrained_path}"}

        try:
            checkpoint = torch.load(pretrained_path, map_location="cpu")
        except Exception as exc:
            return {"loaded": False, "reason": f"failed to read pretrained checkpoint: {exc}"}

        state_dict = self._extract_state_dict(checkpoint)
        if not state_dict:
            return {
                "loaded": False,
                "reason": f"failed to resolve pretrained state dict with key={self.pretrained_key!r}",
            }

        model_state = encoder.state_dict()
        matched_state: Dict[str, torch.Tensor] = {}
        unexpected = 0
        shape_mismatched = 0

        for key, value in state_dict.items():
            if key not in model_state:
                unexpected += 1
                continue
            if tuple(value.shape) != tuple(model_state[key].shape):
                shape_mismatched += 1
                continue
            matched_state[key] = value

        missing = len([key for key in model_state.keys() if key not in matched_state])
        if self.pretrained_strict and (unexpected > 0 or shape_mismatched > 0 or missing > 0):
            raise RuntimeError(
                "Strict pretrained loading failed: "
                f"missing={missing} unexpected={unexpected} shape_mismatched={shape_mismatched}"
            )

        if not matched_state:
            return {
                "loaded": False,
                "reason": "no overlapping pretrained keys after filtering",
                "matched_keys": 0,
                "missing_keys": missing,
                "unexpected_keys": unexpected,
                "shape_mismatched_keys": shape_mismatched,
            }

        encoder.load_state_dict(matched_state, strict=False)
        return {
            "loaded": True,
            "reason": "pretrained weights loaded with filtered state dict",
            "matched_keys": len(matched_state),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "shape_mismatched_keys": shape_mismatched,
            "path": str(pretrained_path),
            "strict": self.pretrained_strict,
        }

    def _extract_state_dict(self, checkpoint: Any) -> Dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict):
            if self.pretrained_key in checkpoint and isinstance(checkpoint[self.pretrained_key], dict):
                return checkpoint[self.pretrained_key]
            tensor_values = all(torch.is_tensor(value) for value in checkpoint.values())
            if tensor_values:
                return checkpoint
        return {}

    @staticmethod
    def _failed_build(reason: str, repo: Path | None = None) -> Dict[str, Any]:
        return {
            "encoder": None,
            "feature_channels": [],
            "reason": reason,
            "repo_path": str(repo) if repo is not None else "",
            "pretrained_report": {"loaded": False, "reason": reason},
        }

    @staticmethod
    def _normalize_feature_channels(dims: Any) -> List[int]:
        if isinstance(dims, (list, tuple)):
            return [int(ch) for ch in dims]
        if isinstance(dims, int):
            return [int(dims * (2**i)) for i in range(4)]
        return [32, 64, 128, 256]

    @staticmethod
    def _normalize_backbone_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        forward_type = kwargs.get("forward_type")
        if isinstance(forward_type, str):
            normalized = forward_type
            for suffix in ["nozact", "noz", "no32", "oact", "onnone", "ondwconv3", "oncnorm", "onsoftmax", "onsigmoid"]:
                normalized = normalized.replace(suffix, f"_{suffix}") if f"_{suffix}" not in normalized else normalized
                normalized = normalized.replace("__", "_")
            kwargs["forward_type"] = normalized
        return kwargs

    def get_metadata(self) -> Dict[str, Any]:
        return dict(self.metadata)

    def forward(self, pre_img: torch.Tensor, post_img: torch.Tensor) -> Dict[str, object]:
        if self.encoder is None:
            raise RuntimeError("Backbone encoder was not initialized.")

        pre_features = list(self.encoder(pre_img))
        post_features = list(self.encoder(post_img))
        if len(pre_features) != len(post_features):
            raise RuntimeError("Backbone returned mismatched pre/post feature lengths.")

        fused_features = []
        feature_strides: List[List[int]] = []
        feature_shapes: List[List[int]] = []
        for pre_feat, post_feat, fuse_layer in zip(pre_features, post_features, self.fuse_layers):
            if pre_feat.shape != post_feat.shape:
                post_feat = F.interpolate(post_feat, size=pre_feat.shape[-2:], mode="bilinear", align_corners=False)
            fused = torch.cat([pre_feat, post_feat, post_feat - pre_feat], dim=1)
            fused_features.append(fuse_layer(fused))
            stride_h = max(int(round(pre_img.shape[-2] / max(pre_feat.shape[-2], 1))), 1)
            stride_w = max(int(round(pre_img.shape[-1] / max(pre_feat.shape[-1], 1))), 1)
            feature_strides.append([stride_h, stride_w])
            feature_shapes.append([int(pre_feat.shape[-2]), int(pre_feat.shape[-1])])

        self.feature_strides = feature_strides
        self.feature_shapes = feature_shapes
        self.metadata["feature_strides"] = feature_strides
        self.metadata["feature_shapes"] = feature_shapes

        return {
            "pre_features": pre_features,
            "post_features": post_features,
            "fused_features": fused_features,
            "high_level": fused_features[-1],
            "backend_name": self.metadata["backend_name"],
            "backend_reason": self.metadata["backend_reason"],
            "feature_channels": self.out_channels,
            "feature_strides": feature_strides,
        }
