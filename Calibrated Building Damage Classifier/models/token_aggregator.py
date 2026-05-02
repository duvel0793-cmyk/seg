from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        attn_input = self.norm1(tokens)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        tokens = tokens + attn_output
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class MaskTokenAggregator(nn.Module):
    """
    Extract K evidence tokens from inside the instance mask and mix them with
    a cls token. This keeps multiple local evidence slots instead of collapsing
    the building to a single pooled vector too early.
    """

    def __init__(
        self,
        dim: int,
        *,
        token_count: int = 8,
        mixer_layers: int = 1,
        mixer_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.token_count = int(token_count)
        self.query_tokens = nn.Parameter(torch.randn(self.token_count, self.dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim) * 0.02)
        self.position_scale = nn.Parameter(torch.tensor(1.0))
        self.mixer = nn.ModuleList(
            [TokenMixerBlock(self.dim, num_heads=mixer_heads, dropout=dropout) for _ in range(mixer_layers)]
        )
        self.output_norm = nn.LayerNorm(self.dim)

    def _masked_attention_pool(self, feature_map: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = feature_map.shape
        flat_features = feature_map.flatten(2).transpose(1, 2)
        flat_mask = F.interpolate(mask, size=(height, width), mode="nearest").flatten(2).squeeze(1) > 0.5
        valid_mask = flat_mask
        if not bool(valid_mask.any()):
            valid_mask = torch.ones_like(flat_mask, dtype=torch.bool)

        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        attention_logits = torch.matmul(queries, flat_features.transpose(1, 2)) / math.sqrt(float(channels))
        attention_logits = attention_logits.masked_fill(~valid_mask.unsqueeze(1), torch.finfo(attention_logits.dtype).min)

        empty_rows = ~flat_mask.any(dim=1)
        if empty_rows.any():
            attention_logits[empty_rows] = torch.matmul(
                queries[empty_rows],
                flat_features[empty_rows].transpose(1, 2),
            ) / math.sqrt(float(channels))

        attention = torch.softmax(attention_logits, dim=-1)
        tokens = torch.matmul(attention, flat_features)
        return tokens, attention

    def forward(self, feature_map: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens, attention = self._masked_attention_pool(feature_map, mask)
        cls_token = self.cls_token.expand(feature_map.size(0), -1, -1)
        mixed_tokens = torch.cat([cls_token, tokens], dim=1)
        for block in self.mixer:
            mixed_tokens = block(mixed_tokens)
        mixed_tokens = self.output_norm(mixed_tokens)
        return {
            "instance_feature": mixed_tokens[:, 0, :],
            "evidence_tokens": mixed_tokens[:, 1:, :],
            "token_attention": attention,
        }
