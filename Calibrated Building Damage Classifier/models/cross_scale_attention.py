from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.normalization import LayerScale


class TokenMixerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float, layerscale_init: float = 1.0e-3) -> None:
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

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(tokens)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        tokens = tokens + self.attn_scale(attn_output)
        tokens = tokens + self.mlp_scale(self.mlp(self.norm2(tokens)))
        return tokens


class PreNormCrossScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        dropout: float,
        layerscale_init: float,
        residual_max: float,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cross_scale = LayerScale(
            dim,
            init_value=layerscale_init,
            max_scale=residual_max,
            ndim=3,
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.ffn_scale = LayerScale(
            dim,
            init_value=layerscale_init,
            max_scale=residual_max,
            ndim=3,
        )

    def forward(
        self,
        tight_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        neighborhood_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv = torch.cat([context_tokens, neighborhood_tokens], dim=1)
        q = self.norm_q(tight_tokens)
        kv = self.norm_kv(kv)
        attn_output, attn_weights = self.attn(
            q,
            kv,
            kv,
            need_weights=True,
            average_attn_weights=False,
        )
        tight_tokens = tight_tokens + self.cross_scale(attn_output)
        tight_tokens = tight_tokens + self.ffn_scale(self.ffn(self.norm_ffn(tight_tokens)))
        return tight_tokens, attn_weights.mean(dim=1)


class CrossScaleTargetConditionedAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        num_layers: int,
        dropout: float,
        context_dropout_prob: float,
        neighborhood_dropout_prob: float,
        layerscale_init: float,
        residual_max: float,
    ) -> None:
        super().__init__()
        self.context_dropout_prob = float(context_dropout_prob)
        self.neighborhood_dropout_prob = float(neighborhood_dropout_prob)
        self.context_null_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.neighborhood_null_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.cross_blocks = nn.ModuleList(
            [
                PreNormCrossScaleBlock(
                    dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    layerscale_init=layerscale_init,
                    residual_max=residual_max,
                )
                for _ in range(num_layers)
            ]
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.output_mixer = nn.ModuleList([TokenMixerBlock(dim, num_heads=num_heads, dropout=dropout) for _ in range(max(1, num_layers))])
        self.output_norm = nn.LayerNorm(dim)

    def _apply_scale_dropout(
        self,
        tokens: torch.Tensor,
        *,
        dropout_prob: float,
        null_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (not self.training) or dropout_prob <= 0.0:
            mask = torch.zeros(tokens.size(0), device=tokens.device, dtype=tokens.dtype)
            return tokens, mask
        drop_mask = (torch.rand(tokens.size(0), device=tokens.device) < dropout_prob).to(dtype=tokens.dtype)
        replacement = null_token.expand(tokens.size(0), tokens.size(1), -1)
        dropped = torch.where(drop_mask[:, None, None] > 0, replacement, tokens)
        return dropped, drop_mask

    def forward(
        self,
        tight_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        neighborhood_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        context_tokens, context_drop = self._apply_scale_dropout(
            context_tokens,
            dropout_prob=self.context_dropout_prob,
            null_token=self.context_null_token,
        )
        neighborhood_tokens, neighborhood_drop = self._apply_scale_dropout(
            neighborhood_tokens,
            dropout_prob=self.neighborhood_dropout_prob,
            null_token=self.neighborhood_null_token,
        )

        tight = tight_tokens
        cross_attn_to_context: list[torch.Tensor] = []
        cross_attn_to_neighborhood: list[torch.Tensor] = []
        cross_attn_entropy: list[torch.Tensor] = []
        last_weights = None
        context_length = context_tokens.size(1)
        neighborhood_length = neighborhood_tokens.size(1)

        for block in self.cross_blocks:
            tight, weights = block(tight, context_tokens, neighborhood_tokens)
            last_weights = weights
            attention_context = weights[:, :, :context_length]
            attention_neighborhood = weights[:, :, context_length : context_length + neighborhood_length]
            cross_attn_to_context.append(attention_context.mean(dim=(1, 2)))
            cross_attn_to_neighborhood.append(attention_neighborhood.mean(dim=(1, 2)))
            cross_attn_entropy.append(
                -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1).mean(dim=1)
            )

        mixed_tokens = torch.cat([self.cls_token.expand(tight.size(0), -1, -1), tight], dim=1)
        for block in self.output_mixer:
            mixed_tokens = block(mixed_tokens)
        mixed_tokens = self.output_norm(mixed_tokens)
        instance_feature = mixed_tokens[:, 0, :]

        if last_weights is None:
            kv_length = context_length + neighborhood_length
            last_weights = torch.zeros(
                tight.size(0),
                tight.size(1),
                kv_length,
                device=tight.device,
                dtype=tight.dtype,
            )

        return {
            "instance_feature": instance_feature,
            "cross_scale_tokens": mixed_tokens[:, 1:, :],
            "tight_tokens": tight,
            "cross_attention_weights": last_weights,
            "cross_attn_to_context_mean": torch.stack(cross_attn_to_context, dim=0).mean(dim=0),
            "cross_attn_to_neighborhood_mean": torch.stack(cross_attn_to_neighborhood, dim=0).mean(dim=0),
            "cross_scale_attention_entropy": torch.stack(cross_attn_entropy, dim=0).mean(dim=0),
            "tight_token_norm": tight.float().norm(dim=-1).mean(dim=1),
            "context_token_norm": context_tokens.float().norm(dim=-1).mean(dim=1),
            "neighborhood_token_norm": neighborhood_tokens.float().norm(dim=-1).mean(dim=1),
            "context_dropout_rate_actual": context_drop,
            "neighborhood_dropout_rate_actual": neighborhood_drop,
        }
