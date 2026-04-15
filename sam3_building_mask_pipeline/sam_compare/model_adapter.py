"""Adapter layer between this project and the upstream SAM3 repository."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def _append_repo_to_syspath(sam3_repo: Path) -> None:
    repo_path = str(sam3_repo.resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def _parse_amp_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported AMP dtype: {name}")
    return mapping[key]


def get_autocast_context(
    device: torch.device,
    enabled: bool,
    amp_dtype: str,
):
    """Return an autocast context compatible with SAM3's upstream runtime."""
    if not enabled:
        return nullcontext()
    dtype = _parse_amp_dtype(amp_dtype)
    if dtype == torch.float32:
        return nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    if device.type == "cpu":
        cpu_dtype = torch.bfloat16 if dtype == torch.float16 else dtype
        return torch.autocast(device_type="cpu", dtype=cpu_dtype)
    return nullcontext()


def _resolve_group_norm_groups(num_channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    """Small conv block used by the local decoders."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = dilation if kernel_size == 3 else 0
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=_resolve_group_norm_groups(out_channels),
                num_channels=out_channels,
            ),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualRefineBlock(nn.Module):
    """A shallow residual block that sharpens fused decoder features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        group_count = _resolve_group_norm_groups(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=group_count, num_channels=channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=group_count, num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class ResidualASPPBlock(nn.Module):
    """Multi-dilation context aggregation with a residual shortcut."""

    def __init__(self, channels: int, dilations: tuple[int, ...] = (1, 2, 4)) -> None:
        super().__init__()
        self.pre = ConvBlock(channels, channels)
        self.branches = nn.ModuleList(
            ConvBlock(channels, channels, dilation=dilation)
            for dilation in dilations
        )
        fused_channels = channels * (len(dilations) + 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(
                num_groups=_resolve_group_norm_groups(channels),
                num_channels=channels,
            ),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(
                num_groups=_resolve_group_norm_groups(channels),
                num_channels=channels,
            ),
        )
        self.out_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pre(x)
        multi_scale_features = [x]
        multi_scale_features.extend(branch(x) for branch in self.branches)
        fused = self.fuse(torch.cat(multi_scale_features, dim=1))
        return self.out_act(fused + residual)


class GatedFusionBlock(nn.Module):
    """Lightweight attention-style top-down fusion for FPN features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.lateral_refine = ConvBlock(channels, channels)
        self.top_refine = ConvBlock(channels, channels)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.GroupNorm(
                num_groups=_resolve_group_norm_groups(channels),
                num_channels=channels,
            ),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out_refine = ResidualRefineBlock(channels)

    def forward(self, lateral: torch.Tensor, top: torch.Tensor) -> torch.Tensor:
        lateral_x = self.lateral_refine(lateral)
        top_x = F.interpolate(
            self.top_refine(top),
            size=lateral.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        gate = self.gate(torch.cat([lateral_x, top_x], dim=1))
        return self.out_refine(lateral_x + gate * top_x)


class PresenceGatingHead(nn.Module):
    """Image-level presence classifier that also gates decoder channels."""

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = max(channels // 4, 32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity(),
        )
        self.gate = nn.Linear(hidden_dim, channels)
        self.logit = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.gate.bias, 1.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.pool(x).flatten(1)
        hidden = self.mlp(pooled)
        channel_gate = torch.sigmoid(self.gate(hidden)).unsqueeze(-1).unsqueeze(-1)
        gated_features = x * (0.25 + 0.75 * channel_gate)
        presence_logit = self.logit(hidden).squeeze(-1)
        return gated_features, presence_logit


class PredictionHead(nn.Module):
    """Shared prediction head template for binary logits."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LegacyLightweightBinaryDecoder(nn.Module):
    """Original lightweight FPN decoder kept for old experiment checkpoints."""

    def __init__(
        self,
        in_channels: int = 256,
        decoder_channels: int = 128,
        decoder_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj_low = nn.Conv2d(in_channels, decoder_channels, kernel_size=1)
        self.proj_mid = nn.Conv2d(in_channels, decoder_channels, kernel_size=1)
        self.proj_high = nn.Conv2d(in_channels, decoder_channels, kernel_size=1)
        self.proj_low_refine = ConvBlock(decoder_channels, decoder_channels)
        self.proj_mid_refine = ConvBlock(decoder_channels, decoder_channels)
        self.proj_high_refine = ConvBlock(decoder_channels, decoder_channels)
        self.refine_mid = ConvBlock(decoder_channels, decoder_channels)
        self.refine_low = ConvBlock(decoder_channels, decoder_channels)
        self.fusion_refine = ResidualRefineBlock(decoder_channels)
        self.dropout = (
            nn.Dropout2d(p=decoder_dropout)
            if decoder_dropout > 0.0
            else nn.Identity()
        )
        self.head = PredictionHead(decoder_channels)

    def forward(
        self,
        features: list[torch.Tensor],
        *,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        if len(features) != 3:
            raise ValueError(f"Expected 3 FPN features from SAM3, got {len(features)}")
        low, mid, high = features
        high_x = self.proj_high_refine(self.proj_high(high))
        mid_x = self.proj_mid_refine(self.proj_mid(mid)) + F.interpolate(
            high_x,
            size=mid.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        mid_x = self.refine_mid(mid_x)
        low_x = self.proj_low_refine(self.proj_low(low)) + F.interpolate(
            mid_x,
            size=low.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        low_x = self.refine_low(low_x)
        low_x = self.fusion_refine(low_x)
        low_x = self.dropout(low_x)
        logits = self.head(low_x)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class BoundaryAwareBinaryDecoder(nn.Module):
    """Upgraded decoder with multi-scale context, gated fusion, and auxiliary heads."""

    def __init__(
        self,
        in_channels: int = 256,
        decoder_channels: int = 128,
        decoder_dropout: float = 0.1,
        *,
        use_resaspp: bool = True,
        use_attention_fusion: bool = True,
        use_presence_head: bool = True,
        use_boundary_head: bool = True,
    ) -> None:
        super().__init__()
        self.use_attention_fusion = use_attention_fusion
        self.use_presence_head = use_presence_head
        self.use_boundary_head = use_boundary_head

        self.proj_low = nn.Conv2d(in_channels, decoder_channels, kernel_size=1)
        self.proj_mid = nn.Conv2d(in_channels, decoder_channels, kernel_size=1)
        self.proj_high = nn.Conv2d(in_channels, decoder_channels, kernel_size=1)
        self.proj_low_refine = ConvBlock(decoder_channels, decoder_channels)
        self.proj_mid_refine = ConvBlock(decoder_channels, decoder_channels)
        self.proj_high_refine = ConvBlock(decoder_channels, decoder_channels)

        self.context_block = (
            ResidualASPPBlock(decoder_channels)
            if use_resaspp
            else ResidualRefineBlock(decoder_channels)
        )
        if use_attention_fusion:
            self.mid_fusion = GatedFusionBlock(decoder_channels)
            self.low_fusion = GatedFusionBlock(decoder_channels)
        else:
            self.refine_mid = ConvBlock(decoder_channels, decoder_channels)
            self.refine_low = ConvBlock(decoder_channels, decoder_channels)

        self.fusion_refine = ResidualRefineBlock(decoder_channels)
        self.dropout = (
            nn.Dropout2d(p=decoder_dropout)
            if decoder_dropout > 0.0
            else nn.Identity()
        )
        self.mask_head = PredictionHead(decoder_channels)
        self.boundary_head = PredictionHead(decoder_channels) if use_boundary_head else None
        self.presence_head = (
            PresenceGatingHead(decoder_channels, decoder_dropout)
            if use_presence_head
            else None
        )

    def _fuse_mid(self, projected_mid: torch.Tensor, high_x: torch.Tensor) -> torch.Tensor:
        if self.use_attention_fusion:
            return self.mid_fusion(projected_mid, high_x)
        mid_x = projected_mid + F.interpolate(
            high_x,
            size=projected_mid.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.refine_mid(mid_x)

    def _fuse_low(self, projected_low: torch.Tensor, mid_x: torch.Tensor) -> torch.Tensor:
        if self.use_attention_fusion:
            return self.low_fusion(projected_low, mid_x)
        low_x = projected_low + F.interpolate(
            mid_x,
            size=projected_low.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.refine_low(low_x)

    def forward(
        self,
        features: list[torch.Tensor],
        *,
        output_size: tuple[int, int],
    ) -> dict[str, torch.Tensor | None]:
        if len(features) != 3:
            raise ValueError(f"Expected 3 FPN features from SAM3, got {len(features)}")

        low, mid, high = features
        high_x = self.proj_high_refine(self.proj_high(high))
        high_x = self.context_block(high_x)

        mid_x = self._fuse_mid(self.proj_mid_refine(self.proj_mid(mid)), high_x)
        low_x = self._fuse_low(self.proj_low_refine(self.proj_low(low)), mid_x)
        shared_features = self.dropout(self.fusion_refine(low_x))

        presence_logit: torch.Tensor | None = None
        if self.presence_head is not None:
            shared_features, presence_logit = self.presence_head(shared_features)

        mask_logits = self.mask_head(shared_features)
        boundary_logits = (
            self.boundary_head(shared_features)
            if self.boundary_head is not None
            else None
        )
        mask_logits = F.interpolate(
            mask_logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        if boundary_logits is not None:
            boundary_logits = F.interpolate(
                boundary_logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        return {
            "mask_logits": mask_logits,
            "boundary_logits": boundary_logits,
            "presence_logit": presence_logit,
        }


def _build_sam3_backbone(
    sam3_repo: Path,
    checkpoint_path: Optional[Path],
) -> nn.Module:
    """Build the real upstream SAM3 image model, then keep only its visual backbone."""
    _append_repo_to_syspath(sam3_repo)
    from sam3.model import vitdet as sam3_vitdet
    from sam3 import build_sam3_image_model

    if not hasattr(sam3_vitdet, "_sam_compare_original_addmm_act"):
        sam3_vitdet._sam_compare_original_addmm_act = sam3_vitdet.addmm_act
    original_addmm_act = sam3_vitdet._sam_compare_original_addmm_act

    def _train_safe_addmm_act(activation, linear, mat1):
        if not torch.is_grad_enabled():
            return original_addmm_act(activation, linear, mat1)
        output = linear(mat1)
        if activation in (torch.nn.functional.relu, torch.nn.ReLU):
            return torch.nn.functional.relu(output)
        if activation in (torch.nn.functional.gelu, torch.nn.GELU):
            return torch.nn.functional.gelu(output)
        raise ValueError(f"Unexpected activation {activation}")

    sam3_vitdet.addmm_act = _train_safe_addmm_act

    sam3_model = build_sam3_image_model(
        device="cpu",
        eval_mode=False,
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        load_from_HF=False,
        enable_segmentation=False,
        enable_inst_interactivity=False,
    )
    backbone = sam3_model.backbone
    backbone.language_backbone = None
    return backbone


@dataclass
class CheckpointInspection:
    checkpoint_type: str
    payload: dict[str, Any]


@dataclass
class SegmentationModelOutput:
    mask_logits: torch.Tensor
    boundary_logits: Optional[torch.Tensor] = None
    presence_logit: Optional[torch.Tensor] = None


@dataclass
class ExperimentCheckpointRestoreHints:
    backbone_checkpoint: Optional[Path]
    legacy_decoder: bool


def inspect_checkpoint(checkpoint_path: str | Path) -> CheckpointInspection:
    """Inspect a checkpoint and tell whether it is a full experiment checkpoint."""
    checkpoint = torch.load(
        Path(checkpoint_path).expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return CheckpointInspection(checkpoint_type="experiment", payload=checkpoint)
    if isinstance(checkpoint, dict):
        return CheckpointInspection(checkpoint_type="backbone", payload=checkpoint)
    return CheckpointInspection(checkpoint_type="unknown", payload={})


def normalize_model_output(model_output: torch.Tensor | Mapping[str, Any]) -> SegmentationModelOutput:
    """Normalize legacy tensor outputs and new dict outputs to a shared shape."""
    if isinstance(model_output, torch.Tensor):
        return SegmentationModelOutput(mask_logits=model_output)
    if not isinstance(model_output, Mapping):
        raise TypeError(
            "Model output must be either a tensor or a mapping containing mask_logits. "
            f"Got {type(model_output)!r}."
        )

    if "mask_logits" in model_output:
        mask_logits = model_output["mask_logits"]
    elif "logits" in model_output:
        mask_logits = model_output["logits"]
    else:
        raise KeyError("Model output mapping must contain `mask_logits` or `logits`.")

    boundary_logits = model_output.get("boundary_logits")
    presence_logit = model_output.get("presence_logit")
    if isinstance(presence_logit, torch.Tensor) and presence_logit.ndim > 1:
        presence_logit = presence_logit.reshape(presence_logit.shape[0], -1).squeeze(-1)

    return SegmentationModelOutput(
        mask_logits=mask_logits,
        boundary_logits=boundary_logits,
        presence_logit=presence_logit,
    )


def is_legacy_model_config(model_config: Mapping[str, Any]) -> bool:
    """Old checkpoints had only the lightweight mask head and no auxiliary flags."""
    return not any(
        key in model_config
        for key in (
            "use_resaspp",
            "use_attention_fusion",
            "use_presence_head",
            "use_boundary_head",
        )
    )


def apply_checkpoint_model_config(model_config: Mapping[str, Any], model_settings: Any) -> bool:
    """Apply saved model settings onto the current config object."""
    scalar_fields = {
        "decoder_channels": int,
        "freeze_backbone": bool,
        "use_amp": bool,
        "amp_dtype": str,
        "decoder_dropout": float,
    }
    for key, converter in scalar_fields.items():
        if key in model_config:
            setattr(model_settings, key, converter(model_config[key]))

    legacy_decoder = is_legacy_model_config(model_config)
    if legacy_decoder:
        model_settings.use_resaspp = False
        model_settings.use_attention_fusion = False
        model_settings.use_presence_head = False
        model_settings.use_boundary_head = False
        return True

    bool_fields = (
        "use_resaspp",
        "use_attention_fusion",
        "use_presence_head",
        "use_boundary_head",
    )
    for key in bool_fields:
        if key in model_config:
            setattr(model_settings, key, bool(model_config[key]))
    return False


def apply_experiment_checkpoint_config(
    config: Any,
    checkpoint_payload: Mapping[str, Any],
) -> ExperimentCheckpointRestoreHints:
    """Update config from an experiment checkpoint before rebuilding the model."""
    payload_config = checkpoint_payload.get("config", {})
    paths_config = payload_config.get("paths", {})
    model_config = payload_config.get("model", {})
    data_config = payload_config.get("data", {})

    saved_sam3_repo = paths_config.get("sam3_repo")
    if saved_sam3_repo:
        config.paths.sam3_repo = Path(saved_sam3_repo).expanduser().resolve()
    saved_image_size = data_config.get("image_size")
    if saved_image_size:
        config.data.image_size = int(saved_image_size)

    legacy_decoder = apply_checkpoint_model_config(model_config, config.model)
    payload_backbone = checkpoint_payload.get("backbone_checkpoint")
    backbone_checkpoint = (
        Path(payload_backbone).expanduser().resolve()
        if payload_backbone
        else None
    )
    return ExperimentCheckpointRestoreHints(
        backbone_checkpoint=backbone_checkpoint,
        legacy_decoder=legacy_decoder,
    )


class SAM3BinarySegmentationModel(nn.Module):
    """Standalone binary segmentation model that reuses SAM3's visual backbone."""

    def __init__(
        self,
        *,
        sam3_repo: str | Path,
        backbone_checkpoint: Optional[str | Path],
        decoder_channels: int,
        freeze_backbone: bool,
        device: str | torch.device,
        use_amp: bool,
        amp_dtype: str,
        decoder_dropout: float,
        use_resaspp: bool,
        use_attention_fusion: bool,
        use_presence_head: bool,
        use_boundary_head: bool,
    ) -> None:
        super().__init__()
        self.sam3_repo = Path(sam3_repo).expanduser().resolve()
        self.backbone_checkpoint = (
            Path(backbone_checkpoint).expanduser().resolve()
            if backbone_checkpoint is not None
            else None
        )
        self.device_obj = torch.device(device)
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.freeze_backbone = freeze_backbone

        self.backbone = _build_sam3_backbone(self.sam3_repo, self.backbone_checkpoint)
        if not any((use_resaspp, use_attention_fusion, use_presence_head, use_boundary_head)):
            self.decoder: nn.Module = LegacyLightweightBinaryDecoder(
                in_channels=256,
                decoder_channels=decoder_channels,
                decoder_dropout=decoder_dropout,
            )
        else:
            self.decoder = BoundaryAwareBinaryDecoder(
                in_channels=256,
                decoder_channels=decoder_channels,
                decoder_dropout=decoder_dropout,
                use_resaspp=use_resaspp,
                use_attention_fusion=use_attention_fusion,
                use_presence_head=use_presence_head,
                use_boundary_head=use_boundary_head,
            )

        self.set_backbone_trainable(not freeze_backbone)
        self.to(self.device_obj)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable

    def backbone_parameters(self):
        return self.backbone.parameters()

    def decoder_parameters(self):
        return self.decoder.parameters()

    def forward(self, images: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor | None]:
        autocast_context = get_autocast_context(
            self.device_obj,
            enabled=self.use_amp,
            amp_dtype=self.amp_dtype,
        )
        with autocast_context:
            backbone_out = self.backbone.forward_image(images)
            features = backbone_out["backbone_fpn"]
            output = self.decoder(features, output_size=tuple(images.shape[-2:]))

        if isinstance(output, torch.Tensor):
            return output.float()

        normalized_output: dict[str, torch.Tensor | None] = {}
        for key, value in output.items():
            normalized_output[key] = value.float() if isinstance(value, torch.Tensor) else value
        return normalized_output


def load_experiment_state(
    model: nn.Module,
    checkpoint_payload: dict[str, Any],
    *,
    strict: bool = True,
) -> None:
    """Load a full experiment checkpoint into the local model."""
    incompatible_keys = model.load_state_dict(
        checkpoint_payload["model_state_dict"],
        strict=False,
    )
    missing_keys = list(incompatible_keys.missing_keys)
    unexpected_keys = list(incompatible_keys.unexpected_keys)

    decoder_prefix = "decoder."
    disallowed_missing = [key for key in missing_keys if not key.startswith(decoder_prefix)]
    disallowed_unexpected = [
        key for key in unexpected_keys if not key.startswith(decoder_prefix)
    ]

    if strict and (disallowed_missing or disallowed_unexpected):
        raise RuntimeError(
            "Failed to load experiment checkpoint strictly. "
            f"Missing keys: {missing_keys}. Unexpected keys: {unexpected_keys}."
        )
    if missing_keys or unexpected_keys:
        warnings.warn(
            "Loaded checkpoint with partial decoder compatibility. "
            f"Missing keys: {missing_keys}. Unexpected keys: {unexpected_keys}.",
            stacklevel=2,
        )
