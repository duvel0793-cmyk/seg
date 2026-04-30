from __future__ import annotations

import torch
import torch.nn as nn


class TokenMixerBlock(nn.Module):
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

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.norm1(tokens)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class CrossScaleBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, tight_tokens: torch.Tensor, context_tokens: torch.Tensor, neighborhood_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kv = torch.cat([context_tokens, neighborhood_tokens], dim=1)
        if kv.size(1) == 0:
            empty_weights = torch.zeros(
                tight_tokens.size(0),
                tight_tokens.size(1),
                0,
                device=tight_tokens.device,
                dtype=tight_tokens.dtype,
            )
            return tight_tokens, empty_weights
        q = self.norm_q(tight_tokens)
        kv_norm = self.norm_kv(kv)
        attn_out, weights = self.attn(q, kv_norm, kv_norm, need_weights=True, average_attn_weights=False)
        tight_tokens = tight_tokens + attn_out
        tight_tokens = tight_tokens + self.ffn(self.norm_ffn(tight_tokens))
        return tight_tokens, weights.mean(dim=1)


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
    ) -> None:
        super().__init__()
        self.context_dropout_prob = float(context_dropout_prob)
        self.neighborhood_dropout_prob = float(neighborhood_dropout_prob)
        self.context_null = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.neighborhood_null = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.blocks = nn.ModuleList([CrossScaleBlock(dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)])
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.mixers = nn.ModuleList([TokenMixerBlock(dim, num_heads=num_heads, dropout=dropout) for _ in range(max(1, num_layers))])
        self.output_norm = nn.LayerNorm(dim)

    def _drop_scale(self, tokens: torch.Tensor, prob: float, null_token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.size(1) == 0:
            return tokens, torch.zeros(tokens.size(0), device=tokens.device, dtype=tokens.dtype)
        if (not self.training) or prob <= 0.0:
            return tokens, torch.zeros(tokens.size(0), device=tokens.device, dtype=tokens.dtype)
        mask = (torch.rand(tokens.size(0), device=tokens.device) < prob).to(tokens.dtype)
        replacement = null_token.expand(tokens.size(0), tokens.size(1), -1)
        return torch.where(mask[:, None, None] > 0, replacement, tokens), mask

    def forward(self, *, tight_tokens: torch.Tensor, context_tokens: torch.Tensor, neighborhood_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        context_tokens, context_drop = self._drop_scale(context_tokens, self.context_dropout_prob, self.context_null)
        neighborhood_tokens, neighborhood_drop = self._drop_scale(neighborhood_tokens, self.neighborhood_dropout_prob, self.neighborhood_null)
        tight = tight_tokens
        all_entropy = []
        last_weights = None
        context_len = context_tokens.size(1)
        neighborhood_len = neighborhood_tokens.size(1)
        ctx_stats = []
        nbr_stats = []
        for block in self.blocks:
            tight, weights = block(tight, context_tokens, neighborhood_tokens)
            last_weights = weights
            ctx_weights = weights[:, :, :context_len]
            nbr_weights = weights[:, :, context_len : context_len + neighborhood_len]
            ctx_stats.append(
                ctx_weights.mean(dim=(1, 2))
                if context_len > 0
                else torch.zeros(tight.size(0), device=tight.device, dtype=tight.dtype)
            )
            nbr_stats.append(
                nbr_weights.mean(dim=(1, 2))
                if neighborhood_len > 0
                else torch.zeros(tight.size(0), device=tight.device, dtype=tight.dtype)
            )
            entropy = (
                -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1).mean(dim=1)
                if weights.size(-1) > 0
                else torch.zeros(tight.size(0), device=tight.device, dtype=tight.dtype)
            )
            all_entropy.append(entropy)
        tokens = torch.cat([self.cls_token.expand(tight.size(0), -1, -1), tight], dim=1)
        for mixer in self.mixers:
            tokens = mixer(tokens)
        tokens = self.output_norm(tokens)
        return {
            "instance_feature": tokens[:, 0, :],
            "cross_scale_tokens": tokens[:, 1:, :],
            "tight_tokens": tight,
            "cross_attention_weights": last_weights if last_weights is not None else torch.zeros(tight.size(0), tight.size(1), context_len + neighborhood_len, device=tight.device, dtype=tight.dtype),
            "cross_attn_to_context_mean": torch.stack(ctx_stats, dim=0).mean(dim=0) if ctx_stats else torch.zeros(tight.size(0), device=tight.device, dtype=tight.dtype),
            "cross_attn_to_neighborhood_mean": torch.stack(nbr_stats, dim=0).mean(dim=0) if nbr_stats else torch.zeros(tight.size(0), device=tight.device, dtype=tight.dtype),
            "cross_scale_attention_entropy": torch.stack(all_entropy, dim=0).mean(dim=0) if all_entropy else torch.zeros(tight.size(0), device=tight.device, dtype=tight.dtype),
            "context_dropout_rate_actual": context_drop,
            "neighborhood_dropout_rate_actual": neighborhood_drop,
        }
