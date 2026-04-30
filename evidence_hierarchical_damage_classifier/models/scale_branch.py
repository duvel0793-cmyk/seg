from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.evidence_head import EvidenceHead
from models.evidence_pooling import EvidencePooling


def _resolve_groups(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _resize_mask_to_feature_map(mask: torch.Tensor | None, spatial_size: tuple[int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    return F.interpolate(mask.float(), size=spatial_size, mode="nearest").to(device=device, dtype=dtype)


class ResidualGateAlignment(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 64)
        self.context = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=max(hidden // 8, 1), bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self.correction = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.gate = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, pre_feature: torch.Tensor, post_feature: torch.Tensor) -> torch.Tensor:
        diff = post_feature - pre_feature
        context = self.context(torch.cat([pre_feature, post_feature, diff], dim=1))
        correction = self.correction(context)
        gate = torch.sigmoid(self.gate(context))
        return pre_feature + (gate * correction)


class PrePostFusion(nn.Module):
    def __init__(self, channels: int, mode: str = "diff_prod_concat") -> None:
        super().__init__()
        self.mode = str(mode)
        if self.mode == "simple_diff_concat":
            self.block = nn.Sequential(
                nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False),
                nn.GroupNorm(_resolve_groups(channels), channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.GroupNorm(_resolve_groups(channels), channels),
                nn.GELU(),
            )
        elif self.mode == "diff_prod_concat":
            self.block = nn.Sequential(
                nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=max(channels // 8, 1), bias=False),
                nn.BatchNorm2d(channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.GELU(),
            )
        else:
            raise ValueError(f"Unsupported fusion mode '{self.mode}'.")

    def forward(self, aligned_pre: torch.Tensor, post_feature: torch.Tensor) -> torch.Tensor:
        diff = post_feature - aligned_pre
        if self.mode == "simple_diff_concat":
            return self.block(torch.cat([aligned_pre, post_feature, diff, torch.abs(diff)], dim=1))
        prod = post_feature * aligned_pre
        return self.block(torch.cat([aligned_pre, post_feature, diff, prod], dim=1))


class DamageAwareChangeBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 64)
        self.change_gate_head = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_resolve_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
        )
        self.pre_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_resolve_groups(channels), channels),
            nn.GELU(),
        )
        self.post_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_resolve_groups(channels), channels),
            nn.GELU(),
        )
        self.change_project = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_resolve_groups(channels), channels),
            nn.GELU(),
        )

    def forward(self, feat_pre: torch.Tensor, feat_post: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        diff = torch.abs(feat_pre - feat_post)
        gate_logits = self.change_gate_head(diff)
        gate = torch.sigmoid(gate_logits / 2.0)
        feat_pre_refined = self.pre_enhance(feat_pre + (gate * feat_pre))
        feat_post_refined = self.post_enhance(feat_post + (gate * feat_post))
        change_feature = self.change_project(torch.cat([feat_pre_refined, feat_post_refined, diff], dim=1)) * gate
        resized_mask = None if mask is None else _resize_mask_to_feature_map(mask, gate.shape[-2:], gate.device, gate.dtype)
        return {
            "feat_pre_refined": feat_pre_refined,
            "feat_post_refined": feat_post_refined,
            "change_feature": change_feature,
            "change_gate_logits": gate_logits,
            "change_gate": gate,
            "mask_resized": resized_mask if resized_mask is not None else torch.ones_like(gate),
        }


class WindowSelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.mask_embed = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, tokens: torch.Tensor, mask_tokens: torch.Tensor) -> torch.Tensor:
        x = self.norm1(tokens) + self.mask_embed(mask_tokens.unsqueeze(-1))
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class LocalWindowAttention(nn.Module):
    def __init__(self, dim: int, *, token_count: int, window_size: int, num_heads: int, num_layers: int, dropout: float, mask_bias: float, background_bias: float) -> None:
        super().__init__()
        self.dim = int(dim)
        self.token_count = int(token_count)
        self.window_size = int(window_size)
        self.mask_bias = float(mask_bias)
        self.background_bias = float(background_bias)
        self.query_tokens = nn.Parameter(torch.randn(self.token_count, self.dim) * 0.02)
        self.blocks = nn.ModuleList([WindowSelfAttentionBlock(dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)])
        self.token_input_norm = nn.LayerNorm(self.dim)
        self.token_output_norm = nn.LayerNorm(self.dim)
        self.feature_output_norm = nn.LayerNorm(self.dim)
        self.query_norm = nn.LayerNorm(self.dim)
        self.output_norm = nn.LayerNorm(self.dim)

    def _partition_windows(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int, int]]:
        batch_size, channels, height, width = x.shape
        pad_h = (self.window_size - (height % self.window_size)) % self.window_size
        pad_w = (self.window_size - (width % self.window_size)) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        padded_h, padded_w = x.shape[-2:]
        x = x.view(batch_size, channels, padded_h // self.window_size, self.window_size, padded_w // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, self.window_size * self.window_size, channels)
        return x, (batch_size, padded_h, padded_w, height, width)

    def _reverse_windows(self, windows: torch.Tensor, meta: tuple[int, int, int, int, int]) -> torch.Tensor:
        batch_size, padded_h, padded_w, height, width = meta
        x = windows.view(batch_size, padded_h // self.window_size, padded_w // self.window_size, self.window_size, self.window_size, self.dim)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(batch_size, self.dim, padded_h, padded_w)
        return x[:, :, :height, :width]

    def forward(self, fused_feature: torch.Tensor, mask: torch.Tensor | None) -> dict[str, torch.Tensor]:
        resized_mask = _resize_mask_to_feature_map(mask, fused_feature.shape[-2:], fused_feature.device, fused_feature.dtype)
        if resized_mask is None:
            resized_mask = torch.ones(fused_feature.size(0), 1, fused_feature.size(2), fused_feature.size(3), device=fused_feature.device, dtype=fused_feature.dtype)
        if self.blocks:
            feature_windows, meta = self._partition_windows(fused_feature)
            mask_windows, _ = self._partition_windows(resized_mask)
            tokens = self.token_input_norm(feature_windows)
            mask_tokens = mask_windows.mean(dim=-1)
            for block in self.blocks:
                tokens = block(tokens, mask_tokens)
            tokens = self.token_output_norm(tokens)
            refined_feature = self._reverse_windows(tokens, meta)
        else:
            refined_feature = fused_feature
        flat_feature = self.feature_output_norm(refined_feature.flatten(2).transpose(1, 2))
        flat_mask = resized_mask.flatten(2).squeeze(1)
        queries = self.query_norm(self.query_tokens).unsqueeze(0).expand(flat_feature.size(0), -1, -1)
        attn_logits = torch.matmul(queries, flat_feature.transpose(1, 2)) / (float(self.dim) ** 0.5)
        attn_logits = attn_logits + (((flat_mask * self.mask_bias) + ((1.0 - flat_mask) * self.background_bias)).unsqueeze(1))
        attention = torch.softmax(attn_logits, dim=-1)
        evidence_tokens = self.output_norm(torch.matmul(attention, flat_feature))
        entropy = -(attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()).sum(dim=-1).mean(dim=1)
        return {
            "refined_feature": refined_feature,
            "tokens": evidence_tokens,
            "attention_entropy": entropy,
            "resized_mask": resized_mask,
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
        evidence_dim: int,
        evidence_topk_ratio: float,
        evidence_threshold: float,
        use_evidence_head: bool,
        enable_alignment: bool = True,
        enable_prepost_fusion: bool = True,
        fusion_mode: str = "diff_prod_concat",
        enable_damage_aware_block: bool = True,
        enable_change_gate: bool = True,
    ) -> None:
        super().__init__()
        self.scale_name = str(scale_name)
        self.feature_dim = int(feature_dim)
        self.use_evidence_head = bool(use_evidence_head)
        self.evidence_dim = int(evidence_dim)
        self.enable_alignment = bool(enable_alignment)
        self.enable_prepost_fusion = bool(enable_prepost_fusion)
        self.enable_damage_aware_block = bool(enable_damage_aware_block)
        self.enable_change_gate = bool(enable_change_gate) and self.enable_damage_aware_block
        self.evidence_stats_dim = 18
        self.c4_projection = nn.Sequential(
            nn.Conv2d(c4_channels, feature_dim, kernel_size=1, bias=False),
            nn.GroupNorm(_resolve_groups(feature_dim), feature_dim),
            nn.GELU(),
        )
        self.c5_projection = nn.Sequential(
            nn.Conv2d(c5_channels, feature_dim, kernel_size=1, bias=False),
            nn.GroupNorm(_resolve_groups(feature_dim), feature_dim),
            nn.GELU(),
        )
        self.alignment = ResidualGateAlignment(feature_dim) if self.enable_alignment else None
        self.fusion = PrePostFusion(feature_dim, mode=fusion_mode) if self.enable_prepost_fusion else None
        self.change_block = DamageAwareChangeBlock(feature_dim) if self.enable_damage_aware_block else None
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
        )
        self.evidence_head = EvidenceHead(feature_dim) if self.use_evidence_head else None
        self.evidence_pooling = (
            EvidencePooling(topk_ratio=evidence_topk_ratio, threshold=evidence_threshold, out_dim=evidence_dim)
            if self.use_evidence_head
            else None
        )

    def _empty_change_outputs(
        self,
        feat_pre: torch.Tensor,
        feat_post: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor | None]:
        batch_size = feat_pre.size(0)
        resized_mask = _resize_mask_to_feature_map(mask, feat_pre.shape[-2:], feat_pre.device, feat_pre.dtype)
        if resized_mask is None:
            resized_mask = torch.ones(batch_size, 1, feat_pre.size(2), feat_pre.size(3), device=feat_pre.device, dtype=feat_pre.dtype)
        return {
            "feat_pre_refined": feat_pre,
            "feat_post_refined": feat_post,
            "change_feature": torch.zeros_like(feat_pre),
            "change_gate_logits": None,
            "change_gate": None,
            "mask_resized": resized_mask,
        }

    def _encode_pair(self, backbone: nn.Module, pre_input: torch.Tensor, post_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre_features = backbone(pre_input)
        post_features = backbone(post_input)
        pre_c4 = self.c4_projection(pre_features["c4"])
        post_c4 = self.c4_projection(post_features["c4"])
        pre_c5 = self.c5_projection(pre_features["c5"])
        post_c5 = self.c5_projection(post_features["c5"])
        pre_c4 = F.interpolate(pre_c4, size=pre_c5.shape[-2:], mode="bilinear", align_corners=False)
        post_c4 = F.interpolate(post_c4, size=post_c5.shape[-2:], mode="bilinear", align_corners=False)
        return pre_c5 + pre_c4, post_c5 + post_c4

    def forward(self, *, backbone: nn.Module, pre_input: torch.Tensor, post_input: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
        pre_feature, post_feature = self._encode_pair(backbone, pre_input, post_input)
        aligned_pre = self.alignment(pre_feature, post_feature) if self.alignment is not None else pre_feature
        fused_feature = self.fusion(aligned_pre, post_feature) if self.fusion is not None else post_feature
        change_outputs = (
            self.change_block(aligned_pre, post_feature, mask)
            if self.change_block is not None
            else self._empty_change_outputs(aligned_pre, post_feature, mask)
        )
        if not self.enable_change_gate:
            change_outputs["change_gate"] = None
            change_outputs["change_gate_logits"] = None
            change_outputs["change_feature"] = torch.zeros_like(fused_feature)
        fused_feature = fused_feature + change_outputs["change_feature"]
        local_outputs = self.local_attention(fused_feature, mask)
        refined_feature = local_outputs["refined_feature"]
        tokens = local_outputs["tokens"]
        resized_mask = local_outputs["resized_mask"]
        damage_aux_score = None if change_outputs["change_gate"] is None else change_outputs["change_gate"].flatten(1).mean(dim=1)
        severity_score = None
        evidence_logits = None
        severity_map = None
        evidence_enabled = self.evidence_head is not None
        if self.evidence_head is not None:
            evidence_logits, severity_map = self.evidence_head(refined_feature)
            pooled = self.evidence_pooling(evidence_logits=evidence_logits, severity_map=severity_map, target_mask=mask)
            evidence_stats = pooled["projected"]
            evidence_raw_stats = pooled["raw"]
        else:
            batch_size = refined_feature.size(0)
            evidence_stats = refined_feature.new_zeros(batch_size, self.evidence_dim)
            evidence_raw_stats = None
        if severity_map is not None:
            severity_score = torch.sigmoid(severity_map).flatten(1).mean(dim=1)
        gate_mean = (
            change_outputs["change_gate"].flatten(1).mean(dim=1)
            if change_outputs["change_gate"] is not None
            else refined_feature.new_zeros(refined_feature.size(0))
        )
        diagnostics = {
            f"{self.scale_name}_local_attention_entropy": local_outputs["attention_entropy"],
            f"{self.scale_name}_token_norm": tokens.float().norm(dim=-1).mean(dim=1),
            f"{self.scale_name}_gate_mean": gate_mean,
            f"{self.scale_name}_evidence_enabled": refined_feature.new_full((refined_feature.size(0),), float(evidence_enabled)),
        }
        return {
            "pre_feature": pre_feature,
            "post_feature": post_feature,
            "aligned_pre": aligned_pre,
            "feat_pre_refined": change_outputs["feat_pre_refined"],
            "feat_post_refined": change_outputs["feat_post_refined"],
            "fused_feature": refined_feature,
            "tokens": tokens,
            "evidence_logits": evidence_logits,
            "severity_map": severity_map,
            "evidence_stats": evidence_stats,
            "evidence_raw_stats": evidence_raw_stats,
            "damage_aux_score": damage_aux_score,
            "severity_score": severity_score,
            "change_gate": change_outputs["change_gate"],
            "mask_resized": resized_mask,
            "evidence_enabled": evidence_enabled,
            "diagnostics": diagnostics,
        }
