from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.normalization import LayerScale


class NeighborhoodGraphModule(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        use_distance_bias: bool,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.use_distance_bias = bool(use_distance_bias)
        self.target_norm = nn.LayerNorm(dim)
        self.neighbor_norm = nn.LayerNorm(dim)
        self.query_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.num_layers)])
        self.key_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.num_layers)])
        self.value_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.num_layers)])
        self.output_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.num_layers)])
        self.output_scale = nn.ModuleList([LayerScale(dim, init_value=1.0e-3, max_scale=0.2, ndim=3) for _ in range(self.num_layers)])
        self.relative_pos_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.distance_scale = nn.Parameter(torch.tensor(1.0))
        self.graph_gate = nn.Linear(dim, 1)
        nn.init.constant_(self.graph_gate.bias, -1.0)

    def forward(
        self,
        target_feature: torch.Tensor,
        neighbor_features: torch.Tensor,
        neighbor_mask: torch.Tensor,
        relative_positions: torch.Tensor,
        distances: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if neighbor_features.numel() == 0:
            zero_feature = torch.zeros_like(target_feature)
            zero_scalar = torch.zeros(target_feature.size(0), device=target_feature.device, dtype=target_feature.dtype)
            return {
                "graph_feature": zero_feature,
                "graph_gate": zero_scalar,
                "graph_attention_weights": zero_scalar[:, None],
                "graph_attention_entropy": zero_scalar,
                "graph_valid_neighbor_count": zero_scalar,
            }

        mask = neighbor_mask.bool()
        valid_count = mask.float().sum(dim=1)
        distance_feature = torch.cat([relative_positions, distances.unsqueeze(-1)], dim=-1)
        encoded_neighbors = self.neighbor_norm(neighbor_features + self.relative_pos_mlp(distance_feature))
        graph_feature = target_feature
        attention_weights = None

        for layer_index in range(self.num_layers):
            query = self.query_proj[layer_index](self.target_norm(graph_feature)).unsqueeze(1)
            key = self.key_proj[layer_index](encoded_neighbors)
            value = self.value_proj[layer_index](encoded_neighbors)
            attn_logits = torch.matmul(query, key.transpose(1, 2)).squeeze(1) / math.sqrt(float(self.dim))
            if self.use_distance_bias:
                attn_logits = attn_logits - (distances.float() * self.distance_scale.abs())
            attn_logits = attn_logits.masked_fill(~mask, torch.finfo(attn_logits.dtype).min)
            no_valid = ~mask.any(dim=1)
            if no_valid.any():
                attn_logits[no_valid] = 0.0
            attention_weights = torch.softmax(attn_logits, dim=-1)
            aggregated = torch.matmul(attention_weights.unsqueeze(1), value).squeeze(1)
            graph_feature = graph_feature + self.output_scale[layer_index](self.output_proj[layer_index](aggregated).unsqueeze(1)).squeeze(1)

        gate = torch.sigmoid(self.graph_gate(target_feature)).squeeze(1)
        entropy = -(attention_weights.clamp_min(1e-8) * attention_weights.clamp_min(1e-8).log()).sum(dim=1)
        graph_feature = graph_feature * gate.unsqueeze(1)
        return {
            "graph_feature": graph_feature,
            "graph_gate": gate,
            "graph_attention_weights": attention_weights,
            "graph_attention_entropy": entropy,
            "graph_valid_neighbor_count": valid_count,
        }
