"""CORN ordinal regression utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def corn_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    if labels.ndim != 1:
        labels = labels.view(-1)
    thresholds = torch.arange(1, num_classes, device=labels.device).view(1, -1)
    return (labels.view(-1, 1) >= thresholds).float()


def corn_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    targets = corn_targets(labels, num_classes=num_classes).to(logits.dtype)
    if logits.shape != targets.shape:
        raise ValueError(f"Logits shape {tuple(logits.shape)} does not match CORN targets {tuple(targets.shape)}.")
    return F.binary_cross_entropy_with_logits(logits, targets)


def corn_predict(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    passes = (probs > 0.5).long()
    # Enforce monotonic ordinal decisions: once one threshold fails, all later thresholds are treated as failed.
    monotonic_passes = torch.cumprod(passes, dim=1)
    return monotonic_passes.sum(dim=1)


def corn_class_probs(logits: torch.Tensor) -> torch.Tensor:
    probs_gt = torch.sigmoid(logits)
    bsz, num_thresholds = probs_gt.shape
    class_probs = torch.zeros((bsz, num_thresholds + 1), device=logits.device, dtype=logits.dtype)
    survival = torch.ones((bsz,), device=logits.device, dtype=logits.dtype)
    for idx in range(num_thresholds):
        class_probs[:, idx] = survival * (1.0 - probs_gt[:, idx])
        survival = survival * probs_gt[:, idx]
    class_probs[:, -1] = survival
    class_probs = class_probs.clamp_min(1e-8)
    return class_probs / class_probs.sum(dim=1, keepdim=True)
