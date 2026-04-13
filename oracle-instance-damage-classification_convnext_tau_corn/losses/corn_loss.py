"""CORN loss for ordinal damage classification."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def build_corn_targets(labels: torch.Tensor, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct conditional CORN targets and active masks.

    For threshold k, the target is evaluated only on samples with y > k - 1.
    The target value is 1 if y > k, else 0.
    """

    labels = labels.long()
    num_thresholds = num_classes - 1
    targets = torch.zeros(labels.shape[0], num_thresholds, dtype=torch.float32, device=labels.device)
    active_mask = torch.zeros_like(targets, dtype=torch.bool)

    for threshold_idx in range(num_thresholds):
        if threshold_idx == 0:
            active = torch.ones_like(labels, dtype=torch.bool)
        else:
            active = labels > (threshold_idx - 1)

        active_mask[:, threshold_idx] = active
        targets[active, threshold_idx] = (labels[active] > threshold_idx).float()

    return targets, active_mask


def corn_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int | None = None) -> torch.Tensor:
    """Compute CORN loss over active conditional thresholds."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [B, K-1].")
    if num_classes is None:
        num_classes = logits.shape[1] + 1

    targets, active_mask = build_corn_targets(labels, num_classes)

    loss_sum = logits.new_zeros(())
    sample_count = logits.new_zeros(())
    for threshold_idx in range(num_classes - 1):
        active = active_mask[:, threshold_idx]
        if not active.any():
            continue
        threshold_loss = F.binary_cross_entropy_with_logits(
            logits[active, threshold_idx],
            targets[active, threshold_idx],
            reduction="sum",
        )
        loss_sum = loss_sum + threshold_loss
        sample_count = sample_count + active.sum()

    if sample_count.item() == 0:
        return loss_sum
    return loss_sum / sample_count

