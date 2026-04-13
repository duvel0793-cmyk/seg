from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return normalized * self.weight[:, None, None] + self.bias[:, None, None]


class ConvStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        hidden_channels = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1, bias=True)
        self.norm1 = nn.BatchNorm2d(hidden_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = self.act(x + residual)
        return x


class ConvStage(nn.Module):
    def __init__(self, channels: int, depth: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualConvBlock(channels) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ConvDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)
        self.norm = LayerNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class VMambaConvMlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LegacyScan2DOperator(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.inner_dim = int(dim) * 2
        self.in_proj = nn.Linear(dim, self.inner_dim)
        self.conv2d = nn.Conv2d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=3,
            padding=1,
            groups=self.inner_dim,
            bias=False,
        )
        self.out_norm = nn.LayerNorm(self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, dim)

    @staticmethod
    def _scan_width(x: torch.Tensor) -> torch.Tensor:
        state = x.new_zeros(x.shape[0], x.shape[1], x.shape[2])
        outputs = []
        for index in range(x.shape[-1]):
            state = 0.6 * state + 0.4 * x[:, :, :, index]
            outputs.append(state)
        return torch.stack(outputs, dim=-1)

    @staticmethod
    def _scan_height(x: torch.Tensor) -> torch.Tensor:
        state = x.new_zeros(x.shape[0], x.shape[1], x.shape[3])
        outputs = []
        for index in range(x.shape[-2]):
            state = 0.6 * state + 0.4 * x[:, :, index, :]
            outputs.append(state)
        return torch.stack(outputs, dim=-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.in_proj(x)
        projected_cf = projected.permute(0, 3, 1, 2).contiguous()
        local = self.conv2d(projected_cf)
        left_right = self._scan_width(local)
        right_left = torch.flip(self._scan_width(torch.flip(local, dims=[-1])), dims=[-1])
        top_bottom = self._scan_height(local)
        bottom_top = torch.flip(self._scan_height(torch.flip(local, dims=[-2])), dims=[-2])
        mixed = 0.25 * (left_right + right_left + top_bottom + bottom_top) + local
        mixed = mixed.permute(0, 2, 3, 1).contiguous()
        mixed = self.out_norm(mixed)
        return self.out_proj(mixed)


class LegacyVMambaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = max(int(dim * mlp_ratio), dim)
        self.norm = nn.LayerNorm(dim)
        self.op = LegacyScan2DOperator(dim)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = VMambaConvMlp(dim=dim, hidden_dim=hidden_dim, dropout=dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.op(self.norm(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class LegacyVMambaStage(nn.Module):
    def __init__(self, dim: int, depth: int, drop_path_rates: list[float], dropout: float = 0.0) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                LegacyVMambaBlock(dim=dim, dropout=dropout, drop_path=float(drop_path_rates[index]))
                for index in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.blocks(x)
        return x.permute(0, 3, 1, 2).contiguous()


def build_drop_path_rates(depths: list[int], drop_path_rate: float) -> list[list[float]]:
    flat = torch.linspace(0.0, float(drop_path_rate), steps=sum(depths)).tolist()
    chunks: list[list[float]] = []
    cursor = 0
    for depth in depths:
        chunks.append(flat[cursor : cursor + depth])
        cursor += depth
    return chunks


def extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "network"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if not isinstance(checkpoint, dict):
        raise TypeError("Pretrained checkpoint must be a dict or contain a state_dict-like payload.")
    return checkpoint


def strip_common_prefixes(key: str) -> str:
    normalized = str(key)
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "model.", "backbone.", "encoder.", "net."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                changed = True
    return normalized


def maybe_expand_input_weight(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
    if source.ndim != 4 or target.ndim != 4:
        return None
    if source.shape[0] != target.shape[0] or source.shape[2:] != target.shape[2:]:
        return None
    if source.shape[1] == 3 and target.shape[1] == 4:
        extra_channel = source.mean(dim=1, keepdim=True)
        return torch.cat([source, extra_channel], dim=1)
    return None


def _build_vmamba_key_mapping(depth_stage3: int, depth_stage4: int) -> list[tuple[str, str]]:
    mapping = [
        ("patch_embed.0.", "stem.conv1."),
        ("patch_embed.2.", "stem.norm1."),
        ("patch_embed.5.", "stem.conv2."),
        ("patch_embed.7.", "stem.norm2."),
        ("layers.0.downsample.1.", "downsample_c3.conv."),
        ("layers.0.downsample.3.", "downsample_c3.norm."),
        ("layers.1.downsample.1.", "downsample_c4.conv."),
        ("layers.1.downsample.3.", "downsample_c4.norm."),
        ("layers.2.downsample.1.", "downsample_c5.conv."),
        ("layers.2.downsample.3.", "downsample_c5.norm."),
    ]
    for block_index in range(depth_stage3):
        mapping.append((f"layers.2.blocks.{block_index}.", f"stage3.blocks.{block_index}."))
    for block_index in range(depth_stage4):
        mapping.append((f"layers.3.blocks.{block_index}.", f"stage4.blocks.{block_index}."))
    return mapping


def remap_vmamba_key(key: str, depth_stage3: int, depth_stage4: int) -> str:
    normalized = strip_common_prefixes(key)
    for source_prefix, target_prefix in _build_vmamba_key_mapping(depth_stage3, depth_stage4):
        if normalized.startswith(source_prefix):
            return target_prefix + normalized[len(source_prefix) :]
    return normalized


def load_vmamba_pretrained(
    module: nn.Module,
    weight_path: str | Path,
    *,
    depth_stage3: int,
    depth_stage4: int,
    verbose: bool = True,
) -> dict[str, Any]:
    path = Path(weight_path) if str(weight_path) else None
    if path is None or not str(weight_path):
        summary = {
            "requested": True,
            "loaded": False,
            "status": "skipped_empty_path",
            "path": "",
            "loaded_keys": [],
            "missing_keys": list(module.state_dict().keys()),
            "unexpected_keys": [],
            "adapted_keys": [],
            "skipped_shape_mismatch": [],
        }
        if verbose:
            print("[VMamba] No pretrained weight path provided. Skip loading.")
        return summary
    if not path.exists():
        summary = {
            "requested": True,
            "loaded": False,
            "status": "missing_path",
            "path": str(path),
            "loaded_keys": [],
            "missing_keys": list(module.state_dict().keys()),
            "unexpected_keys": [],
            "adapted_keys": [],
            "skipped_shape_mismatch": [],
        }
        if verbose:
            print(f"[VMamba] Warning: pretrained weight not found at {path}. Use random initialization.")
        return summary

    checkpoint = torch.load(path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    model_state = module.state_dict()
    loadable_state: dict[str, torch.Tensor] = {}
    loaded_keys: list[str] = []
    adapted_keys: list[str] = []
    skipped_shape_mismatch: list[str] = []
    unexpected_keys: list[str] = []

    for raw_key, value in state_dict.items():
        mapped_key = remap_vmamba_key(raw_key, depth_stage3=depth_stage3, depth_stage4=depth_stage4)
        if mapped_key not in model_state:
            unexpected_keys.append(strip_common_prefixes(raw_key))
            continue

        target_tensor = model_state[mapped_key]
        source_tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            adapted_tensor = maybe_expand_input_weight(source_tensor, target_tensor)
            if adapted_tensor is None or tuple(adapted_tensor.shape) != tuple(target_tensor.shape):
                skipped_shape_mismatch.append(mapped_key)
                continue
            source_tensor = adapted_tensor
            adapted_keys.append(mapped_key)

        loadable_state[mapped_key] = source_tensor.to(dtype=target_tensor.dtype)
        loaded_keys.append(mapped_key)

    load_result = module.load_state_dict(loadable_state, strict=False)
    missing_keys = [key for key in load_result.missing_keys if key not in loadable_state]
    summary = {
        "requested": True,
        "loaded": bool(loadable_state),
        "status": "loaded_partial" if loadable_state else "no_matching_keys",
        "path": str(path),
        "loaded_keys": loaded_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "adapted_keys": adapted_keys,
        "skipped_shape_mismatch": skipped_shape_mismatch,
    }
    if verbose:
        print(f"[VMamba] Load path: {path}")
        print(f"[VMamba] Loaded keys ({len(loaded_keys)}): {loaded_keys[:20]}{' ...' if len(loaded_keys) > 20 else ''}")
        print(f"[VMamba] Missing keys ({len(missing_keys)}): {missing_keys[:20]}{' ...' if len(missing_keys) > 20 else ''}")
        print(
            f"[VMamba] Unexpected keys ({len(unexpected_keys)}): "
            f"{unexpected_keys[:20]}{' ...' if len(unexpected_keys) > 20 else ''}"
        )
        if adapted_keys:
            print(f"[VMamba] Adapted 3->4 channel keys: {adapted_keys}")
        if skipped_shape_mismatch:
            print(
                "[VMamba] Skipped shape mismatch keys "
                f"({len(skipped_shape_mismatch)}): {skipped_shape_mismatch[:20]}"
                f"{' ...' if len(skipped_shape_mismatch) > 20 else ''}"
            )
    return summary
