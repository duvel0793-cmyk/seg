from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.normalization import LayerScale


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


class WindowSelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float, layerscale_init: float = 1.0e-2) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn_scale = LayerScale(dim, init_value=layerscale_init, ndim=3)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.mlp_scale = LayerScale(dim, init_value=layerscale_init, ndim=3)
        self.mask_embed = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, tokens: torch.Tensor, mask_tokens: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(tokens) + self.mask_embed(mask_tokens.unsqueeze(-1))
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        tokens = tokens + self.attn_scale(attn_output)
        tokens = tokens + self.mlp_scale(self.mlp(self.norm2(tokens)))
        return tokens


class LocalWindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        token_count: int,
        window_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        mask_bias: float,
        background_bias: float,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.token_count = int(token_count)
        self.window_size = int(window_size)
        self.mask_bias = float(mask_bias)
        self.background_bias = float(background_bias)
        self.enabled = bool(enabled)
        self.query_tokens = nn.Parameter(torch.randn(self.token_count, self.dim) * 0.02)
        self.blocks = nn.ModuleList(
            [WindowSelfAttentionBlock(self.dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.token_input_norm = nn.LayerNorm(self.dim)
        self.token_output_norm = nn.LayerNorm(self.dim)
        self.feature_output_norm = nn.LayerNorm(self.dim)
        self.output_norm = nn.LayerNorm(self.dim)
        self.query_norm = nn.LayerNorm(self.dim)

    def _partition_windows(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
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
            self.dim,
        )
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(batch_size, self.dim, padded_h, padded_w)
        return x[:, :, :height, :width]

    def forward(self, fused_feature: torch.Tensor, mask: torch.Tensor | None) -> dict[str, torch.Tensor]:
        resized_mask = _resize_mask_to_feature_map(
            mask,
            fused_feature.shape[-2:],
            device=fused_feature.device,
            dtype=fused_feature.dtype,
        )
        if resized_mask is None:
            resized_mask = torch.ones(
                fused_feature.size(0),
                1,
                fused_feature.size(2),
                fused_feature.size(3),
                device=fused_feature.device,
                dtype=fused_feature.dtype,
            )

        feature_windows, window_meta = self._partition_windows(fused_feature)
        mask_windows, _ = self._partition_windows(resized_mask)
        tokens = self.token_input_norm(feature_windows)
        mask_tokens = mask_windows.mean(dim=-1)

        if self.enabled:
            for block in self.blocks:
                tokens = block(tokens, mask_tokens)
        tokens = self.token_output_norm(tokens)

        refined_feature = self._reverse_windows(tokens, window_meta)
        flat_feature = self.feature_output_norm(refined_feature.flatten(2).transpose(1, 2))
        flat_mask = resized_mask.flatten(2).squeeze(1)

        queries = self.query_norm(self.query_tokens).unsqueeze(0).expand(flat_feature.size(0), -1, -1)
        attention_logits = torch.matmul(queries, flat_feature.transpose(1, 2)) / math.sqrt(float(self.dim))
        mask_bias = (flat_mask * self.mask_bias) + ((1.0 - flat_mask) * self.background_bias)
        attention_logits = attention_logits + mask_bias.unsqueeze(1)
        attention = torch.softmax(attention_logits, dim=-1)
        evidence_tokens = torch.matmul(attention, flat_feature)
        evidence_tokens = self.output_norm(evidence_tokens)

        entropy = -(attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()).sum(dim=-1).mean(dim=1)
        return {
            "refined_feature": refined_feature,
            "tokens": evidence_tokens,
            "token_attention": attention,
            "attention_entropy": entropy,
            "resized_mask": resized_mask,
        }
