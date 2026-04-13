"""Decode utilities for CORN ordinal logits."""

from __future__ import annotations

import torch


def corn_conditional_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert logits to conditional probabilities q_k = P(y > k | y > k-1)."""
    return torch.sigmoid(logits)


def corn_cumulative_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert CORN conditional logits into cumulative probabilities P(y > k)."""
    cond_probs = corn_conditional_probabilities(logits)
    return torch.cumprod(cond_probs, dim=1)


def corn_class_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert CORN logits into class probabilities over K classes."""
    cum_probs = corn_cumulative_probabilities(logits)
    batch_size, num_thresholds = cum_probs.shape
    num_classes = num_thresholds + 1

    class_probs = logits.new_zeros((batch_size, num_classes))
    class_probs[:, 0] = 1.0 - cum_probs[:, 0]
    for class_idx in range(1, num_classes - 1):
        class_probs[:, class_idx] = cum_probs[:, class_idx - 1] - cum_probs[:, class_idx]
    class_probs[:, -1] = cum_probs[:, -1]
    return class_probs.clamp_min(0.0)


def decode_corn_logits(logits: torch.Tensor) -> torch.Tensor:
    """Decode CORN logits into class predictions using class probabilities."""
    class_probs = corn_class_probabilities(logits)
    return class_probs.argmax(dim=1)

