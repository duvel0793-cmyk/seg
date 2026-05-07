from __future__ import annotations

import torch
import torch.nn as nn


class SeverityAwareScaleRouter(nn.Module):
    def __init__(
        self,
        token_dim: int,
        *,
        hidden_dim: int,
        dropout: float,
        init_gates: list[float] | tuple[float, ...],
    ) -> None:
        super().__init__()
        if len(init_gates) != 4:
            raise ValueError("scale router init_gates must have 4 values")
        self.tight_mlp = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )
        self.context_mlp = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )
        init_gate_tensor = torch.as_tensor(init_gates, dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
        self.gate_logits = nn.Parameter(torch.logit(init_gate_tensor))

    @staticmethod
    def _mean_tokens(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.size(1) == 0:
            return tokens.new_zeros((tokens.size(0), tokens.size(-1)))
        return tokens.mean(dim=1)

    def forward(self, tight_tokens: torch.Tensor, context_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        tight_feat = self._mean_tokens(tight_tokens)
        context_feat = self._mean_tokens(context_tokens)
        tight_logits = self.tight_mlp(tight_feat)
        context_logits = self.context_mlp(context_feat)
        gates = torch.sigmoid(self.gate_logits).to(device=tight_logits.device, dtype=tight_logits.dtype)
        router_logits = (gates.unsqueeze(0) * context_logits) + ((1.0 - gates).unsqueeze(0) * tight_logits)
        router_probs = torch.softmax(router_logits.float(), dim=1)
        return {
            "scale_router_logits": router_logits,
            "scale_router_probs": router_probs,
            "scale_router_pred_labels": router_probs.argmax(dim=1),
            "scale_router_gates": gates,
            "tight_logits": tight_logits,
            "context_logits": context_logits,
        }
