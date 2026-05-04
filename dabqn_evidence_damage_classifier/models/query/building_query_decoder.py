from __future__ import annotations

import math

import torch
import torch.nn as nn


def build_2d_sincos_position_embedding(
    height: int,
    width: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError(f"Position embedding dim must be divisible by 4, got {dim}.")
    grid_y = torch.arange(height, device=device, dtype=dtype)
    grid_x = torch.arange(width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    omega = torch.arange(dim // 4, device=device, dtype=dtype)
    omega = 1.0 / (10000 ** (omega / max((dim // 4) - 1, 1)))
    out_y = yy.reshape(-1, 1) * omega.reshape(1, -1)
    out_x = xx.reshape(-1, 1) * omega.reshape(1, -1)
    pos = torch.cat([torch.sin(out_y), torch.cos(out_y), torch.sin(out_x), torch.cos(out_x)], dim=1)
    return pos


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QueryDecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q = self.norm1(query)
        self_attn_out, _ = self.self_attn(q, q, q, need_weights=False)
        query = query + self_attn_out
        cross_q = self.norm2(query)
        cross_attn_out, _ = self.cross_attn(cross_q, memory, memory, need_weights=False)
        query = query + cross_attn_out
        query = query + self.ffn(self.norm3(query))
        return query


class BuildingQueryDecoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_queries: int,
        num_layers: int,
        num_heads: int,
        num_feature_levels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        self.query_content = nn.Embedding(self.num_queries, self.hidden_dim)
        self.level_embed = nn.Parameter(torch.randn(num_feature_levels, self.hidden_dim) * 0.02)
        self.input_proj = nn.ModuleList([nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1) for _ in range(num_feature_levels)])
        self.layers = nn.ModuleList([QueryDecoderLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.output_norm = nn.LayerNorm(hidden_dim)

    def _flatten_memory(self, memory_levels: list[torch.Tensor]) -> torch.Tensor:
        flattened_levels = []
        for level_index, feature in enumerate(memory_levels):
            batch_size, channels, height, width = feature.shape
            projected = self.input_proj[level_index](feature)
            pos = build_2d_sincos_position_embedding(
                height,
                width,
                self.hidden_dim,
                device=feature.device,
                dtype=feature.dtype,
            )
            pos = pos.unsqueeze(0).expand(batch_size, -1, -1)
            level_feature = projected.flatten(2).transpose(1, 2)
            level_feature = level_feature + pos + self.level_embed[level_index].view(1, 1, -1)
            flattened_levels.append(level_feature)
        return torch.cat(flattened_levels, dim=1)

    def forward(self, memory_levels: list[torch.Tensor]) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if not memory_levels:
            raise ValueError("BuildingQueryDecoder requires at least one memory level.")
        batch_size = int(memory_levels[0].shape[0])
        memory = self._flatten_memory(memory_levels)
        query = self.query_content.weight.unsqueeze(0).expand(batch_size, -1, -1)
        query = query + self.query_embed.weight.unsqueeze(0)
        aux_states: list[torch.Tensor] = []
        for layer in self.layers:
            query = layer(query, memory)
            aux_states.append(self.output_norm(query))
        output = self.output_norm(query)
        return {"query_features": output, "aux_query_features": aux_states}
