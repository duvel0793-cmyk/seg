from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import timm
from safetensors.torch import load_file as load_safetensors


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights" / "convnextv2"
LEGACY_WEIGHTS_DIR = PROJECT_ROOT.parent / "Calibrated Building Damage Classifier" / "weights" / "convnextv2"
DEFAULT_CONVNEXTV2_MODEL = "convnextv2_tiny.fcmae_ft_in22k_in1k"


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class GRN(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class Block(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = timm.layers.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


class ConvNeXtV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        raise NotImplementedError("This project uses timm-backed ConvNeXtV2 weights/features.")

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.forward_features(x)


def _strip_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("model.", "module.", "backbone.", "encoder.", "net.")
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix) :]
                    changed = True
        normalized[normalized_key] = value
    return normalized


def _remap_timm_convnextv2_key(key: str) -> str:
    remapped = key
    if remapped.startswith("stem."):
        remapped = remapped.replace("stem.", "stem_", 1)
    for stage_index in range(4):
        remapped = remapped.replace(f"stages.{stage_index}.", f"stages_{stage_index}.")
    return remapped


def adapt_input_conv_weight(weight: torch.Tensor, target_shape: torch.Size) -> tuple[torch.Tensor, bool]:
    target_in_channels = int(target_shape[1])
    source_in_channels = int(weight.shape[1])
    if source_in_channels == target_in_channels:
        return weight, False
    if source_in_channels == 3 and target_in_channels == 4:
        extra_channel = weight.mean(dim=1, keepdim=True)
        return torch.cat([weight, extra_channel], dim=1), True
    if source_in_channels == 4 and target_in_channels == 3:
        return weight[:, :3, :, :], True
    raise ValueError(f"Unsupported conv adaptation {source_in_channels} -> {target_in_channels}")


def resolve_local_weight_path(model_name: str) -> Path:
    if model_name.startswith("hf_hub:timm/"):
        model_name = model_name.removeprefix("hf_hub:timm/")
    candidate = WEIGHTS_DIR / f"{model_name}.safetensors"
    if candidate.exists():
        return candidate
    legacy_candidate = LEGACY_WEIGHTS_DIR / f"{model_name}.safetensors"
    if legacy_candidate.exists():
        return legacy_candidate
    return candidate


def load_convnextv2_pretrained(model: nn.Module, model_name: str) -> list[str]:
    logs: list[str] = []
    weight_path = resolve_local_weight_path(model_name)
    if not weight_path.exists():
        raise FileNotFoundError(f"ConvNeXtV2 weights not found: {weight_path}")
    state_dict = load_safetensors(str(weight_path))
    state_dict = _strip_prefixes(state_dict)
    state_dict = {_remap_timm_convnextv2_key(key): value for key, value in state_dict.items()}
    model_state = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    adapted_first_conv = False
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        candidate = value
        target = model_state[key]
        if candidate.ndim == 4 and target.ndim == 4 and candidate.shape != target.shape:
            if candidate.shape[0] == target.shape[0] and candidate.shape[2:] == target.shape[2:]:
                candidate, adapted = adapt_input_conv_weight(candidate, target.shape)
                adapted_first_conv = adapted_first_conv or adapted
        if candidate.shape == target.shape:
            filtered[key] = candidate
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    missing = [key for key in missing if not key.startswith(("head.", "classifier."))]
    unexpected = [key for key in unexpected if not key.startswith(("head.", "classifier."))]
    if missing or unexpected:
        raise RuntimeError(f"Incompatible ConvNeXtV2 weights. Missing={missing} Unexpected={unexpected}")
    logs.append(f"loaded pretrained weights from {weight_path}")
    if adapted_first_conv:
        logs.append("adapted first conv from 3ch to 4ch")
    return logs
