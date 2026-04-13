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


class PatchEmbed2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int = 4,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class PatchMerging2D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.reduction = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2, bias=False)
        self.norm = LayerNorm2d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.reduction(x)
        return self.norm(x)


class _DirectionalStateScan2D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.decay_logits = nn.Parameter(torch.zeros(4, channels))
        self.mix_logits = nn.Parameter(torch.zeros(4, channels))
        self.input_scale = nn.Parameter(torch.zeros(4, channels))

    @staticmethod
    def _scan_width(x: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        state = x.new_zeros(batch, channels, height)
        outputs = []
        for index in range(width):
            current = x[:, :, :, index]
            state = (decay * state) + ((1.0 - decay) * current)
            outputs.append(state)
        return torch.stack(outputs, dim=-1)

    @staticmethod
    def _scan_height(x: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        state = x.new_zeros(batch, channels, width)
        outputs = []
        for index in range(height):
            current = x[:, :, index, :]
            state = (decay * state) + ((1.0 - decay) * current)
            outputs.append(state)
        return torch.stack(outputs, dim=-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        decay = torch.sigmoid(self.decay_logits).clamp(0.05, 0.995)
        mix = torch.softmax(self.mix_logits, dim=0)
        input_scale = 1.0 + torch.tanh(self.input_scale)

        left_right = self._scan_width(
            x * input_scale[0].view(1, -1, 1, 1),
            decay[0].view(1, -1, 1),
        )
        right_left = torch.flip(
            self._scan_width(
                torch.flip(x, dims=[-1]) * input_scale[1].view(1, -1, 1, 1),
                decay[1].view(1, -1, 1),
            ),
            dims=[-1],
        )
        top_bottom = self._scan_height(
            x * input_scale[2].view(1, -1, 1, 1),
            decay[2].view(1, -1, 1),
        )
        bottom_top = torch.flip(
            self._scan_height(
                torch.flip(x, dims=[-2]) * input_scale[3].view(1, -1, 1, 1),
                decay[3].view(1, -1, 1),
            ),
            dims=[-2],
        )
        return (
            mix[0].view(1, -1, 1, 1) * left_right
            + mix[1].view(1, -1, 1, 1) * right_left
            + mix[2].view(1, -1, 1, 1) * top_bottom
            + mix[3].view(1, -1, 1, 1) * bottom_top
        )


class _ConvMlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class _SpatialStateSpaceMixer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=True)
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.local_enhance = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.scan = _DirectionalStateScan2D(dim)
        self.out_norm = nn.BatchNorm2d(dim)
        self.out_act = nn.GELU()
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected, gate = self.in_proj(x).chunk(2, dim=1)
        projected = self.depthwise(projected)
        scanned = self.scan(projected)
        local = self.local_enhance(projected)
        mixed = (scanned + local) * torch.sigmoid(gate)
        mixed = self.out_act(self.out_norm(mixed))
        return self.out_proj(mixed)


class VSSBlockBasic(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = max(int(dim * mlp_ratio), dim)
        self.norm1 = LayerNorm2d(dim)
        self.ssm = _SpatialStateSpaceMixer(dim)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = LayerNorm2d(dim)
        self.mlp = _ConvMlp(dim, hidden_dim=hidden_dim, dropout=dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ssm(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class VSSStageBasic(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        drop_path_rates: list[float],
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                VSSBlockBasic(
                    dim=dim,
                    drop_path=float(drop_path_rates[index]),
                )
                for index in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class VMambaBasicBackbone(nn.Module):
    ARCH_SPECS: dict[str, dict[str, list[int]]] = {
        "vmamba_tiny": {"depths": [2, 2, 6, 2], "dims": [96, 192, 384, 768]},
        "vmamba_small": {"depths": [2, 2, 12, 2], "dims": [96, 192, 384, 768]},
    }

    def __init__(
        self,
        variant: str = "vmamba_tiny",
        in_channels: int = 4,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        if variant not in self.ARCH_SPECS:
            raise ValueError(f"Unsupported VMamba variant='{variant}'.")
        spec = self.ARCH_SPECS[variant]
        dims = spec["dims"]
        depths = spec["depths"]

        self.variant = str(variant)
        self.patch_embed = PatchEmbed2D(in_channels=in_channels, embed_dim=dims[0], patch_size=4)
        drop_path_rates = torch.linspace(0.0, float(drop_path_rate), steps=sum(depths)).tolist()

        self.stages = nn.ModuleList()
        self.patch_merges = nn.ModuleList()
        cursor = 0
        for stage_index, (dim, depth) in enumerate(zip(dims, depths)):
            self.stages.append(
                VSSStageBasic(
                    dim=dim,
                    depth=depth,
                    drop_path_rates=drop_path_rates[cursor : cursor + depth],
                )
            )
            cursor += depth
            if stage_index < len(dims) - 1:
                self.patch_merges.append(PatchMerging2D(dim, dims[stage_index + 1]))

        self.feature_channels = OrderedDict(
            [
                ("c2", dims[0]),
                ("c3", dims[1]),
                ("c4", dims[2]),
                ("c5", dims[3]),
            ]
        )
        self.pretrained_load_summary = {
            "requested": False,
            "loaded": False,
            "status": "not_requested",
            "path": "",
            "loaded_keys": 0,
            "adapted_keys": [],
            "missing_keys": len(self.state_dict()),
            "unexpected_keys": 0,
            "skipped_shape_mismatch": [],
        }

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
        if isinstance(checkpoint, dict):
            for key in ["state_dict", "model_state_dict", "model", "network"]:
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    return checkpoint[key]
        if not isinstance(checkpoint, dict):
            raise TypeError("Pretrained checkpoint must be a dict or contain a state_dict-like payload.")
        return checkpoint

    @staticmethod
    def _normalize_key(key: str) -> str:
        normalized = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "model.", "backbone.", "encoder.", "net."):
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix) :]
                    changed = True
        return normalized

    @staticmethod
    def _maybe_expand_input_weight(
        key: str,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor | None:
        if key != "patch_embed.proj.weight":
            return None
        if source.ndim != 4 or target.ndim != 4:
            return None
        if source.shape[0] != target.shape[0] or source.shape[2:] != target.shape[2:]:
            return None
        if source.shape[1] == 3 and target.shape[1] == 4:
            extra_channel = source.mean(dim=1, keepdim=True)
            return torch.cat([source, extra_channel], dim=1)
        return None

    def load_pretrained_weights(self, weight_path: str | Path) -> dict[str, Any]:
        path = Path(weight_path)
        if not str(weight_path):
            self.pretrained_load_summary = {
                "requested": True,
                "loaded": False,
                "status": "skipped_empty_path",
                "path": "",
                "loaded_keys": 0,
                "adapted_keys": [],
                "missing_keys": len(self.state_dict()),
                "unexpected_keys": 0,
                "skipped_shape_mismatch": [],
            }
            return self.pretrained_load_summary
        if not path.exists():
            self.pretrained_load_summary = {
                "requested": True,
                "loaded": False,
                "status": "missing_path",
                "path": str(path),
                "loaded_keys": 0,
                "adapted_keys": [],
                "missing_keys": len(self.state_dict()),
                "unexpected_keys": 0,
                "skipped_shape_mismatch": [],
            }
            return self.pretrained_load_summary

        checkpoint = torch.load(path, map_location="cpu")
        state_dict = self._extract_state_dict(checkpoint)
        model_state = self.state_dict()
        loadable_state: dict[str, torch.Tensor] = {}
        adapted_keys: list[str] = []
        skipped_shape_mismatch: list[str] = []
        unexpected_keys = 0

        for raw_key, value in state_dict.items():
            normalized_key = self._normalize_key(raw_key)
            if normalized_key not in model_state:
                unexpected_keys += 1
                continue

            target_tensor = model_state[normalized_key]
            source_tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
            if tuple(source_tensor.shape) != tuple(target_tensor.shape):
                adapted_tensor = self._maybe_expand_input_weight(
                    normalized_key,
                    source_tensor,
                    target_tensor,
                )
                if adapted_tensor is None or tuple(adapted_tensor.shape) != tuple(target_tensor.shape):
                    skipped_shape_mismatch.append(normalized_key)
                    continue
                source_tensor = adapted_tensor
                adapted_keys.append(normalized_key)

            loadable_state[normalized_key] = source_tensor.to(dtype=target_tensor.dtype)

        load_result = self.load_state_dict(loadable_state, strict=False)
        missing_keys = [key for key in load_result.missing_keys if key not in loadable_state]
        self.pretrained_load_summary = {
            "requested": True,
            "loaded": len(loadable_state) > 0,
            "status": "loaded_partial" if len(loadable_state) > 0 else "no_matching_keys",
            "path": str(path),
            "loaded_keys": len(loadable_state),
            "adapted_keys": adapted_keys,
            "missing_keys": len(missing_keys),
            "unexpected_keys": int(unexpected_keys),
            "skipped_shape_mismatch": skipped_shape_mismatch,
        }
        return self.pretrained_load_summary

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        x = self.patch_embed(x)
        c2 = self.stages[0](x)
        x = self.patch_merges[0](c2)
        c3 = self.stages[1](x)
        x = self.patch_merges[1](c3)
        c4 = self.stages[2](x)
        x = self.patch_merges[2](c4)
        c5 = self.stages[3](x)
        return OrderedDict([("c2", c2), ("c3", c3), ("c4", c4), ("c5", c5)])
