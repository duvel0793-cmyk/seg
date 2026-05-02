from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.alignment import ResidualGateAlignment
from models.change_suppression import DamageAwareChangeBlock
from models.fusion import PrePostFusion
from models.local_attention import LocalWindowAttention
from models.normalization import LayerScale, build_channel_norm_2d


def _resize_mask_to_feature_map(
    mask: torch.Tensor | None,
    spatial_size: tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError(f"Expected mask with 3 or 4 dims, got shape={tuple(mask.shape)}.")
    resized_mask = F.interpolate(mask.float(), size=spatial_size, mode="nearest")
    return resized_mask.to(device=device, dtype=dtype)


def _masked_average_pool_2d(feature_map: torch.Tensor, mask: torch.Tensor | None, eps: float = 1e-6) -> torch.Tensor:
    pooled_full = feature_map.float().flatten(2).mean(dim=-1)
    resized_mask = _resize_mask_to_feature_map(
        mask,
        feature_map.shape[-2:],
        device=feature_map.device,
        dtype=feature_map.dtype,
    )
    if resized_mask is None:
        return pooled_full
    flat_mask = resized_mask.flatten(2)
    flat_feature = feature_map.float().flatten(2)
    denominator = flat_mask.sum(dim=-1)
    pooled_masked = (flat_feature * flat_mask).sum(dim=-1) / denominator.clamp_min(eps)
    valid_rows = (denominator >= eps).expand_as(pooled_masked)
    return torch.where(valid_rows, pooled_masked, pooled_full)


def _compute_single_channel_region_statistics(
    single_channel_map: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    values = single_channel_map.float().flatten(1)
    map_mean = values.mean(dim=1)
    map_std = values.std(dim=1, unbiased=False)
    resized_mask = _resize_mask_to_feature_map(
        mask,
        single_channel_map.shape[-2:],
        device=single_channel_map.device,
        dtype=single_channel_map.dtype,
    )
    if resized_mask is None:
        valid = torch.ones_like(map_mean, dtype=torch.bool)
        invalid = torch.zeros_like(map_mean, dtype=torch.bool)
        return {
            "inside_mean": map_mean,
            "outside_mean": map_mean,
            "gap": torch.zeros_like(map_mean),
            "map_mean": map_mean,
            "map_std": map_std,
            "inside_valid": valid,
            "outside_valid": invalid,
        }
    flat_mask = resized_mask.float().flatten(1)
    inside_denominator = flat_mask.sum(dim=1)
    outside_weight = 1.0 - flat_mask
    outside_denominator = outside_weight.sum(dim=1)
    inside_mean = (values * flat_mask).sum(dim=1) / inside_denominator.clamp_min(eps)
    outside_mean = (values * outside_weight).sum(dim=1) / outside_denominator.clamp_min(eps)
    inside_valid = inside_denominator >= eps
    outside_valid = outside_denominator >= eps
    inside_mean = torch.where(inside_valid, inside_mean, map_mean)
    outside_mean = torch.where(outside_valid, outside_mean, map_mean)
    return {
        "inside_mean": inside_mean,
        "outside_mean": outside_mean,
        "gap": inside_mean - outside_mean,
        "map_mean": map_mean,
        "map_std": map_std,
        "inside_valid": inside_valid,
        "outside_valid": outside_valid,
    }


class ScaleBranch(nn.Module):
    def __init__(
        self,
        *,
        scale_name: str,
        c4_channels: int,
        c5_channels: int,
        feature_dim: int,
        token_count: int,
        window_size: int,
        local_attention_heads: int,
        local_attention_layers: int,
        dropout: float,
        use_change_suppression: bool,
        change_block_channels: int,
        enable_pseudo_suppression: bool,
        fuse_change_to_tokens: bool,
        change_residual_scale: float,
        change_gate_init_gamma: float,
        gate_temperature: float,
        gate_bias_init: float,
        enable_damage_aux: bool,
        enable_severity_aux: bool,
        use_local_attention: bool = True,
        norm_kind: str = "group",
    ) -> None:
        super().__init__()
        self.scale_name = str(scale_name)
        self.feature_dim = int(feature_dim)
        self.fuse_change_to_tokens = bool(fuse_change_to_tokens)
        self.use_change_suppression = bool(use_change_suppression)
        self.c4_projection = nn.Sequential(
            nn.Conv2d(c4_channels, feature_dim, kernel_size=1, bias=False),
            build_channel_norm_2d(feature_dim, kind=norm_kind),
            nn.GELU(),
        )
        self.c5_projection = nn.Sequential(
            nn.Conv2d(c5_channels, feature_dim, kernel_size=1, bias=False),
            build_channel_norm_2d(feature_dim, kind=norm_kind),
            nn.GELU(),
        )
        self.alignment = ResidualGateAlignment(feature_dim)
        self.fusion = PrePostFusion(feature_dim, feature_dim)
        self.pre_feature_norm = build_channel_norm_2d(feature_dim, kind=norm_kind)
        self.post_feature_norm = build_channel_norm_2d(feature_dim, kind=norm_kind)
        self.aligned_pre_norm = build_channel_norm_2d(feature_dim, kind=norm_kind)
        self.fused_feature_norm = build_channel_norm_2d(feature_dim, kind=norm_kind)
        self.refined_feature_norm = build_channel_norm_2d(feature_dim, kind=norm_kind)
        self.change_block_channels = int(change_block_channels or feature_dim)
        if self.change_block_channels != feature_dim:
            self.change_input_projection = nn.Conv2d(feature_dim, self.change_block_channels, kernel_size=1, bias=False)
            self.change_output_projection = nn.Conv2d(self.change_block_channels, feature_dim, kernel_size=1, bias=False)
        else:
            self.change_input_projection = nn.Identity()
            self.change_output_projection = nn.Identity()

        if self.use_change_suppression:
            self.change_block = DamageAwareChangeBlock(
                self.change_block_channels,
                enable_pseudo_suppression=enable_pseudo_suppression,
                enable_damage_aux=enable_damage_aux,
                enable_severity_aux=enable_severity_aux,
                residual_scale=change_residual_scale,
                gate_temperature=gate_temperature,
                gate_bias_init=gate_bias_init,
            )
            self.gamma_change = LayerScale(
                feature_dim,
                init_value=float(change_gate_init_gamma),
                max_scale=0.3,
                ndim=4,
            )
        else:
            self.change_block = None
            self.register_parameter("gamma_change", None)

        if self.scale_name == "tight":
            mask_bias, background_bias = 1.0, -0.2
        elif self.scale_name == "context":
            mask_bias, background_bias = 0.6, 0.0
        else:
            mask_bias, background_bias = 0.3, 0.05
        self.local_attention = LocalWindowAttention(
            feature_dim,
            token_count=token_count,
            window_size=window_size,
            num_heads=local_attention_heads,
            num_layers=local_attention_layers,
            dropout=dropout,
            mask_bias=mask_bias,
            background_bias=background_bias,
            enabled=use_local_attention,
        )

    def _encode_pair(
        self,
        backbone: nn.Module,
        pre_input: torch.Tensor,
        post_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pre_features = backbone(pre_input)
        post_features = backbone(post_input)
        pre_c4 = self.c4_projection(pre_features["c4"])
        post_c4 = self.c4_projection(post_features["c4"])
        pre_c5 = self.c5_projection(pre_features["c5"])
        post_c5 = self.c5_projection(post_features["c5"])
        pre_c4 = F.interpolate(pre_c4, size=pre_c5.shape[-2:], mode="bilinear", align_corners=False)
        post_c4 = F.interpolate(post_c4, size=post_c5.shape[-2:], mode="bilinear", align_corners=False)
        pre_feature = self.pre_feature_norm(pre_c5 + pre_c4)
        post_feature = self.post_feature_norm(post_c5 + post_c4)
        return pre_feature, post_feature

    def forward(
        self,
        *,
        backbone: nn.Module,
        pre_input: torch.Tensor,
        post_input: torch.Tensor,
        mask: torch.Tensor,
        collect_diagnostics: bool = True,
    ) -> dict[str, Any]:
        pre_feature, post_feature = self._encode_pair(backbone, pre_input, post_input)
        aligned_pre = self.aligned_pre_norm(self.alignment(pre_feature, post_feature))
        fused_feature = self.fused_feature_norm(self.fusion(aligned_pre, post_feature))
        feat_pre_refined = aligned_pre
        feat_post_refined = post_feature
        change_feature = None
        change_gate = None
        damage_map_logits = None
        severity_logit_map = None

        if self.use_change_suppression:
            assert self.change_block is not None
            change_outputs = self.change_block(
                self.change_input_projection(aligned_pre),
                self.change_input_projection(post_feature),
                mask,
            )
            feat_pre_refined = self.refined_feature_norm(self.change_output_projection(change_outputs["feat_pre_refined"]))
            feat_post_refined = self.refined_feature_norm(self.change_output_projection(change_outputs["feat_post_refined"]))
            change_feature = self.refined_feature_norm(self.change_output_projection(change_outputs["change_feature"]))
            change_gate = change_outputs["change_gate"]
            damage_map_logits = change_outputs["damage_map_logits"]
            severity_logit_map = change_outputs.get("severity_logit_map")
            if self.fuse_change_to_tokens and change_feature is not None:
                fused_feature = self.fused_feature_norm(fused_feature + self.gamma_change(change_feature))

        local_outputs = self.local_attention(fused_feature, mask)
        refined_feature = self.refined_feature_norm(local_outputs["refined_feature"])
        tokens = local_outputs["tokens"]
        resized_mask = local_outputs["resized_mask"]
        damage_aux_score = None if damage_map_logits is None else _masked_average_pool_2d(damage_map_logits, resized_mask).squeeze(1)
        severity_score = None
        if severity_logit_map is not None:
            severity_score = torch.sigmoid(_masked_average_pool_2d(severity_logit_map, resized_mask).squeeze(1))

        diagnostics: dict[str, torch.Tensor | None] = {}
        if collect_diagnostics:
            diagnostics = {
                f"{self.scale_name}_local_attention_entropy": local_outputs["attention_entropy"],
                f"{self.scale_name}_token_norm": tokens.float().norm(dim=-1).mean(dim=1),
            }
            if change_gate is not None:
                gate_stats = _compute_single_channel_region_statistics(change_gate, resized_mask)
                diagnostics.update(
                    {
                        f"{self.scale_name}_gate_inside": gate_stats["inside_mean"],
                        f"{self.scale_name}_gate_outside": gate_stats["outside_mean"],
                        f"{self.scale_name}_gate_gap": gate_stats["gap"],
                        f"{self.scale_name}_gate_mean": gate_stats["map_mean"],
                        f"{self.scale_name}_gate_std": gate_stats["map_std"],
                        f"{self.scale_name}_gate_inside_valid": gate_stats["inside_valid"],
                        f"{self.scale_name}_gate_outside_valid": gate_stats["outside_valid"],
                    }
                )

        return {
            "pre_feature": pre_feature,
            "post_feature": post_feature,
            "aligned_pre": aligned_pre,
            "feat_pre_refined": feat_pre_refined,
            "feat_post_refined": feat_post_refined,
            "fused_feature": refined_feature,
            "tokens": tokens,
            "change_feature": change_feature,
            "change_gate": change_gate,
            "damage_map_logits": damage_map_logits,
            "severity_logit_map": severity_logit_map,
            "severity_score": severity_score,
            "damage_aux_score": damage_aux_score,
            "mask_resized": resized_mask,
            "diagnostics": diagnostics,
        }
