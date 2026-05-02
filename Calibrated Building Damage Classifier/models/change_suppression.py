from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    """AMP-safe 2D conv block used by the lightweight change modeling path."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        norm_type: str = "group",
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        if norm_type == "batch":
            self.norm: nn.Module = nn.BatchNorm2d(out_channels)
        elif norm_type == "group":
            self.norm = nn.GroupNorm(_resolve_group_count(out_channels), out_channels)
        else:
            raise ValueError(f"Unsupported norm_type='{norm_type}'.")

        if activation == "gelu":
            self.act: nn.Module = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation='{activation}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DamageAwareChangeBlock(nn.Module):
    """
    Lightweight category-agnostic change modeling inspired by Seg2Change.

    This is not an open-vocabulary or full Seg2Change implementation. It keeps
    a small damage-aware change branch for xBD instance-level classification and
    uses soft pseudo-change suppression instead of any hard thresholding.
    """

    def __init__(
        self,
        channels: int,
        *,
        enable_pseudo_suppression: bool = True,
        enable_damage_aux: bool = True,
        enable_severity_aux: bool = True,
        residual_scale: float = 0.2,
        norm_type: str = "group",
        gate_temperature: float = 2.0,
        gate_bias_init: float = -2.0,
    ) -> None:
        super().__init__()
        hidden_channels = max(channels // 2, 32)
        self.enable_pseudo_suppression = bool(enable_pseudo_suppression)
        self.enable_damage_aux = bool(enable_damage_aux)
        self.enable_severity_aux = bool(enable_severity_aux)
        self.gate_temperature = max(float(gate_temperature), 1e-6)
        self.gate_bias_init = float(gate_bias_init)

        self.change_gate_head = nn.Sequential(
            ConvNormAct(channels, hidden_channels, kernel_size=3, norm_type=norm_type),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1, bias=True),
        )
        self.pre_enhance = ConvNormAct(channels, channels, kernel_size=3, norm_type=norm_type)
        self.post_enhance = ConvNormAct(channels, channels, kernel_size=3, norm_type=norm_type)
        self.change_project = ConvNormAct(channels * 3, channels, kernel_size=3, norm_type=norm_type)

        if self.enable_pseudo_suppression:
            self.stable_project = ConvNormAct(channels * 2, channels, kernel_size=1, padding=0, norm_type=norm_type)
            self.suppress_gate_head = nn.Sequential(
                ConvNormAct(channels * 2, hidden_channels, kernel_size=3, norm_type=norm_type),
                nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1, bias=True),
            )
            self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale), dtype=torch.float32))
        else:
            self.stable_project = None
            self.suppress_gate_head = None
            self.register_parameter("residual_scale", None)

        if self.enable_damage_aux:
            self.damage_head: nn.Module | None = nn.Sequential(
                ConvNormAct(channels, hidden_channels, kernel_size=3, norm_type=norm_type),
                nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
            )
        else:
            self.damage_head = None
        if self.enable_severity_aux:
            self.severity_head: nn.Module | None = nn.Sequential(
                nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
            )
        else:
            self.severity_head = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        change_gate_last = self.change_gate_head[-1]
        if isinstance(change_gate_last, nn.Conv2d) and change_gate_last.bias is not None:
            nn.init.constant_(change_gate_last.bias, self.gate_bias_init)
        if self.enable_pseudo_suppression and self.suppress_gate_head is not None:
            suppress_gate_last = self.suppress_gate_head[-1]
            if isinstance(suppress_gate_last, nn.Conv2d) and suppress_gate_last.bias is not None:
                nn.init.constant_(suppress_gate_last.bias, 0.0)
        if self.damage_head is not None:
            damage_last = self.damage_head[-1]
            if isinstance(damage_last, nn.Conv2d) and damage_last.bias is not None:
                nn.init.constant_(damage_last.bias, 0.0)
        if self.severity_head is not None:
            severity_last = self.severity_head[-1]
            if isinstance(severity_last, nn.Conv2d) and severity_last.bias is not None:
                nn.init.constant_(severity_last.bias, 0.0)

    def _resize_mask(
        self,
        mask: torch.Tensor | None,
        spatial_size: tuple[int, int],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4:
            raise ValueError(f"Expected mask with 3 or 4 dims, got shape={tuple(mask.shape)}.")
        resized_mask = F.interpolate(mask.float(), size=spatial_size, mode="nearest")
        return resized_mask.to(device=device, dtype=dtype)

    def forward(
        self,
        feat_pre: torch.Tensor,
        feat_post: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        diff = torch.abs(feat_pre - feat_post)
        change_gate_logits = self.change_gate_head(diff)
        change_gate = torch.sigmoid(change_gate_logits / self.gate_temperature)

        pre_enhanced = self.pre_enhance(feat_pre + (change_gate * feat_pre))
        post_enhanced = self.post_enhance(feat_post + (change_gate * feat_post))

        change_feature = self.change_project(torch.cat([pre_enhanced, post_enhanced, diff], dim=1))
        change_feature = change_feature * change_gate

        if self.enable_pseudo_suppression:
            assert self.stable_project is not None
            assert self.suppress_gate_head is not None
            stable_feature = self.stable_project(torch.cat([feat_pre, feat_post], dim=1))
            suppress_gate = torch.sigmoid(self.suppress_gate_head(torch.cat([change_feature, stable_feature], dim=1)))
            residual_scale = self.residual_scale.to(device=change_feature.device, dtype=change_feature.dtype)
            change_feature = (change_feature * suppress_gate) + (change_feature * residual_scale)

        damage_map_logits = None if self.damage_head is None else self.damage_head(change_feature)
        severity_logit_map = self.severity_head(change_feature) if self.severity_head is not None else None

        resized_mask = self._resize_mask(
            mask,
            feat_pre.shape[-2:],
            dtype=change_feature.dtype,
            device=change_feature.device,
        )
        if resized_mask is not None:
            # Keep the mask available for downstream weak supervision and logging.
            mask_for_output = resized_mask
        else:
            mask_for_output = torch.ones_like(change_gate)

        return {
            "feat_pre_refined": pre_enhanced,
            "feat_post_refined": post_enhanced,
            "change_feature": change_feature,
            "change_gate_logits": change_gate_logits,
            "change_gate": change_gate,
            "damage_map_logits": damage_map_logits,
            "severity_logit_map": severity_logit_map,
            "mask_resized": mask_for_output,
        }
