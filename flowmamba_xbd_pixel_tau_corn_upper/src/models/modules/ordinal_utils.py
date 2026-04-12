"""Ordinal helper functions."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rank_targets_to_corn(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert rank labels in [0, K-1] to K-1 binary CORN targets."""

    boundaries = []
    for boundary_idx in range(num_classes - 1):
        boundaries.append((labels > boundary_idx).to(torch.float32))
    return torch.stack(boundaries, dim=-1)


def corn_logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """Convert K-1 CORN logits to K-way class probabilities."""

    squeeze = False
    if logits.ndim == 2:
        logits = logits[:, :, None, None]
        squeeze = True
    if logits.ndim != 4:
        raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")

    boundary_probs = torch.sigmoid(logits).clamp(1.0e-6, 1.0 - 1.0e-6)
    prefix = torch.ones_like(boundary_probs[:, :1])
    if boundary_probs.shape[1] > 1:
        prefix = torch.cat([prefix, torch.cumprod(boundary_probs[:, :-1], dim=1)], dim=1)
    class_probs = []
    for boundary_idx in range(boundary_probs.shape[1]):
        class_probs.append(prefix[:, boundary_idx : boundary_idx + 1] * (1.0 - boundary_probs[:, boundary_idx : boundary_idx + 1]))
    class_probs.append(torch.cumprod(boundary_probs, dim=1)[:, -1:])
    probs = torch.cat(class_probs, dim=1)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1.0e-6)

    if squeeze:
        return probs[:, :, 0, 0]
    return probs


def corn_logits_to_label(logits: torch.Tensor) -> torch.Tensor:
    """Decode K-1 ordinal logits back to rank labels using class probabilities."""

    probs = corn_logits_to_probs(logits)
    if probs.ndim == 4:
        return probs.argmax(dim=1).long()
    if probs.ndim == 2:
        return probs.argmax(dim=1).long()
    raise ValueError(f"Unsupported probs shape: {tuple(probs.shape)}")


def class_prob_at_label(probs: torch.Tensor, labels: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """Gather P(class=gt) at each valid label position."""

    if probs.ndim == 4:
        valid_mask = labels != ignore_index
        if not valid_mask.any():
            return probs.new_zeros((0,))
        probs_flat = probs.permute(0, 2, 3, 1)[valid_mask]
        labels_flat = labels[valid_mask].long()
        return probs_flat.gather(1, labels_flat[:, None]).squeeze(1)
    if probs.ndim == 2:
        if labels.numel() == 0:
            return probs.new_zeros((0,))
        labels = labels.long()
        return probs.gather(1, labels[:, None]).squeeze(1)
    raise ValueError(f"Unsupported probs shape: {tuple(probs.shape)}")


def one_hot_labels(labels: torch.Tensor, num_classes: int, ignore_index: int = 255) -> torch.Tensor:
    """Build one-hot targets for valid positions only."""

    valid_mask = labels != ignore_index
    if labels.ndim == 2:
        labels_valid = labels[valid_mask]
        if labels_valid.numel() == 0:
            return labels.new_zeros((0, num_classes), dtype=torch.float32)
        return F.one_hot(labels_valid.long(), num_classes=num_classes).to(torch.float32)
    if labels.ndim == 3:
        labels_valid = labels[valid_mask]
        if labels_valid.numel() == 0:
            return labels.new_zeros((0, num_classes), dtype=torch.float32)
        return F.one_hot(labels_valid.long(), num_classes=num_classes).to(torch.float32)
    raise ValueError(f"Unsupported labels shape: {tuple(labels.shape)}")


def combine_damage_and_loc(loc_pred: torch.Tensor, damage_rank_pred: torch.Tensor) -> torch.Tensor:
    """Combine binary localization with 0..3 damage ranks into 0..4 xBD labels."""

    combined = torch.zeros_like(damage_rank_pred, dtype=torch.long)
    building_mask = loc_pred > 0
    combined[building_mask] = damage_rank_pred[building_mask] + 1
    return combined
