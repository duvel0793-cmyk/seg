from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_groups(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _resize_mask_to_feature_map(
    mask: torch.Tensor | None,
    spatial_size: tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if mask is None:
        return torch.ones(1, 1, *spatial_size, device=device, dtype=dtype)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    return F.interpolate(mask.float(), size=spatial_size, mode="nearest").to(device=device, dtype=dtype)


class DamageBDFMLite(nn.Module):
    def __init__(self, channels: int, *, hidden_channels: int | None = None, gate_temperature: float = 2.0) -> None:
        super().__init__()
        hidden = int(hidden_channels or max(channels // 2, 64))
        self.gate_temperature = float(gate_temperature)
        self.change_gate_head = nn.Sequential(
            nn.Conv2d((channels * 2) + 1, hidden, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(channels * 4, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_resolve_groups(channels), channels),
            nn.GELU(),
        )

    def forward(
        self,
        feat_pre: torch.Tensor,
        feat_post: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = feat_pre.size(0)
        resized_mask = _resize_mask_to_feature_map(
            mask,
            feat_pre.shape[-2:],
            device=feat_pre.device,
            dtype=feat_pre.dtype,
        )
        if resized_mask.size(0) == 1 and batch_size > 1:
            resized_mask = resized_mask.expand(batch_size, -1, -1, -1)

        diff = torch.abs(feat_post - feat_pre)
        prod = feat_post * feat_pre
        gate_logits = self.change_gate_head(torch.cat([diff, prod, resized_mask], dim=1))
        gate = torch.sigmoid(gate_logits / self.gate_temperature)
        feat_pre_refined = self.pre_enhance(feat_pre + (gate * feat_pre))
        feat_post_refined = self.post_enhance(feat_post + (gate * feat_post))
        change_feature = self.change_project(
            torch.cat([feat_pre_refined, feat_post_refined, diff, prod], dim=1)
        ) * gate
        return {
            "feat_pre_refined": feat_pre_refined,
            "feat_post_refined": feat_post_refined,
            "change_feature": change_feature,
            "change_gate_logits": gate_logits,
            "change_gate": gate,
            "mask_resized": resized_mask,
        }


class EDQALite(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        window_size: int,
        num_heads: int,
        dropout: float,
        init_alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.window_size = int(window_size)
        self.guidance = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_resolve_groups(channels), channels),
            nn.GELU(),
        )
        self.query_norm = nn.LayerNorm(channels)
        self.key_norm = nn.LayerNorm(channels)
        self.value_norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha), dtype=torch.float32))

    def _partition_windows(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int, int]]:
        batch_size, channels, height, width = x.shape
        pad_h = (self.window_size - (height % self.window_size)) % self.window_size
        pad_w = (self.window_size - (width % self.window_size)) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        padded_h, padded_w = x.shape[-2:]
        x = x.view(
            batch_size,
            channels,
            padded_h // self.window_size,
            self.window_size,
            padded_w // self.window_size,
            self.window_size,
        )
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, self.window_size * self.window_size, channels)
        return x, (batch_size, padded_h, padded_w, height, width)

    def _reverse_windows(self, windows: torch.Tensor, meta: tuple[int, int, int, int, int]) -> torch.Tensor:
        batch_size, padded_h, padded_w, height, width = meta
        x = windows.view(
            batch_size,
            padded_h // self.window_size,
            padded_w // self.window_size,
            self.window_size,
            self.window_size,
            self.channels,
        )
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(batch_size, self.channels, padded_h, padded_w)
        return x[:, :, :height, :width]

    def forward(
        self,
        fused_feature: torch.Tensor,
        damage_feature: torch.Tensor,
        feat_pre_refined: torch.Tensor,
        feat_post_refined: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del mask
        guide = self.guidance(torch.cat([feat_pre_refined, feat_post_refined, fused_feature], dim=1))
        damage_windows, meta = self._partition_windows(damage_feature)
        guide_windows, _ = self._partition_windows(guide)
        query = self.query_norm(damage_windows)
        key = self.key_norm(damage_windows)
        value = self.value_norm(guide_windows)
        attn_out, _ = self.attn(query, key, value, need_weights=False)
        calibrated_damage = self._reverse_windows(attn_out, meta)
        alpha = self.alpha.to(device=fused_feature.device, dtype=fused_feature.dtype)
        refined_feature = fused_feature + (alpha * self.out_proj(calibrated_damage))
        calibrated_norm = calibrated_damage.float().flatten(1).norm(dim=1)
        alpha_vector = calibrated_norm.new_full((calibrated_norm.size(0),), float(self.alpha.detach().item()))
        return {
            "refined_feature": refined_feature,
            "calibrated_damage": calibrated_damage,
            "edqa_alpha": alpha_vector,
            "edqa_calibrated_norm": calibrated_norm,
        }
