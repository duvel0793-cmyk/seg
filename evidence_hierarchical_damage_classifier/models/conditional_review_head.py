from __future__ import annotations

import math

import torch
import torch.nn as nn


def _alpha_raw_from_init(alpha_init: float, alpha_max: float) -> float:
    safe_alpha_max = max(float(alpha_max), 1e-6)
    ratio = max(1e-6, min(float(alpha_init) / safe_alpha_max, 1.0 - 1e-6))
    return float(math.log(ratio / (1.0 - ratio)))


class ConditionalReviewHead(nn.Module):
    def __init__(
        self,
        token_dim: int = 256,
        feature_dim: int = 768,
        num_classes: int = 4,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        alpha_init: float = 0.05,
        alpha_max: float = 0.30,
        detach_base_probs: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.alpha_max = float(alpha_max)
        self.detach_base_probs = bool(detach_base_probs)

        base_prob_feature_dim = max(16, min(64, int(hidden_dim) // 4 if int(hidden_dim) > 0 else 16))
        self.base_prob_project = nn.Sequential(
            nn.LayerNorm(self.num_classes),
            nn.Linear(self.num_classes, base_prob_feature_dim),
            nn.GELU(),
        )

        low_input_dim = int(feature_dim) + int(token_dim) + int(token_dim) + base_prob_feature_dim
        high_input_dim = int(feature_dim) + int(token_dim) + base_prob_feature_dim
        self.low_reviewer = nn.Sequential(
            nn.LayerNorm(low_input_dim),
            nn.Linear(low_input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim), self.num_classes),
        )
        self.high_reviewer = nn.Sequential(
            nn.LayerNorm(high_input_dim),
            nn.Linear(high_input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim), self.num_classes),
        )

        init_raw = _alpha_raw_from_init(alpha_init=float(alpha_init), alpha_max=self.alpha_max)
        self.alpha_low_raw = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))
        self.alpha_high_raw = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))

    @staticmethod
    def _mean_pool_tokens(tokens: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
        if tokens is None or tokens.ndim != 3 or tokens.size(1) == 0:
            return reference.new_zeros((reference.size(0), reference.size(-1)))
        return tokens.mean(dim=1)

    def _resolve_alpha(self, raw_alpha: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(raw_alpha).to(device=reference.device, dtype=reference.dtype) * float(self.alpha_max)

    def forward(
        self,
        *,
        base_class_probabilities: torch.Tensor,
        calibrated_feature: torch.Tensor,
        tight_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        neighborhood_tokens: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        gate_probabilities = base_class_probabilities.detach() if self.detach_base_probs else base_class_probabilities

        tight_summary = self._mean_pool_tokens(tight_tokens, tight_tokens)
        context_summary = self._mean_pool_tokens(context_tokens, tight_tokens)
        neighborhood_summary = self._mean_pool_tokens(neighborhood_tokens, context_tokens)

        base_probs_feature = self.base_prob_project(gate_probabilities)
        low_input = torch.cat(
            [
                calibrated_feature,
                context_summary,
                neighborhood_summary,
                base_probs_feature,
            ],
            dim=1,
        )
        high_input = torch.cat(
            [
                calibrated_feature,
                tight_summary,
                base_probs_feature,
            ],
            dim=1,
        )

        low_delta_logits = self.low_reviewer(low_input)
        high_delta_logits = self.high_reviewer(high_input)

        low_gate = gate_probabilities[:, 0:1] + gate_probabilities[:, 1:2]
        high_gate = gate_probabilities[:, 2:3] + gate_probabilities[:, 3:4]
        alpha_low = self._resolve_alpha(self.alpha_low_raw, calibrated_feature)
        alpha_high = self._resolve_alpha(self.alpha_high_raw, calibrated_feature)

        eps = 1e-8
        base_log_probs = torch.log(gate_probabilities.clamp_min(eps))
        final_review_logits = (
            base_log_probs
            + (alpha_low * low_gate * low_delta_logits)
            + (alpha_high * high_gate * high_delta_logits)
        )
        final_review_class_probabilities = torch.softmax(final_review_logits, dim=1)
        conditional_review_pred_labels = final_review_class_probabilities.argmax(dim=1)

        return {
            "conditional_review_logits": final_review_logits,
            "conditional_review_class_probabilities": final_review_class_probabilities,
            "conditional_review_pred_labels": conditional_review_pred_labels,
            "conditional_review_low_logits": low_delta_logits,
            "conditional_review_high_logits": high_delta_logits,
            "conditional_review_low_gate": low_gate.squeeze(1),
            "conditional_review_high_gate": high_gate.squeeze(1),
            "conditional_review_alpha_low": alpha_low,
            "conditional_review_alpha_high": alpha_high,
        }


TwoPassConditionalReviewHead = ConditionalReviewHead

