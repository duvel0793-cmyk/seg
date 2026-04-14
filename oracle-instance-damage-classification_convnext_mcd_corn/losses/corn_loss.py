from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_corn_targets(labels: torch.Tensor, num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    labels = labels.long().view(-1)
    num_thresholds = int(num_classes) - 1
    targets = torch.zeros(labels.shape[0], num_thresholds, dtype=torch.float32, device=labels.device)
    active_mask = torch.zeros_like(targets, dtype=torch.bool)

    for threshold_idx in range(num_thresholds):
        active = torch.ones_like(labels, dtype=torch.bool) if threshold_idx == 0 else labels > (threshold_idx - 1)
        active_mask[:, threshold_idx] = active
        targets[active, threshold_idx] = (labels[active] > threshold_idx).float()
    return targets, active_mask


def logits_to_threshold_probabilities(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


def threshold_probabilities_to_class_probabilities(threshold_probabilities: torch.Tensor) -> torch.Tensor:
    if threshold_probabilities.ndim != 2:
        raise ValueError("threshold_probabilities must have shape [N, K-1].")
    probs = threshold_probabilities.clamp(1e-6, 1.0 - 1e-6)
    num_classes = probs.shape[1] + 1
    class_probs = probs.new_zeros(probs.shape[0], num_classes)
    class_probs[:, 0] = 1.0 - probs[:, 0]
    for class_idx in range(1, num_classes - 1):
        class_probs[:, class_idx] = probs[:, class_idx - 1] * (1.0 - probs[:, class_idx])
    class_probs[:, num_classes - 1] = probs[:, num_classes - 2]
    class_probs = class_probs / class_probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return class_probs


def logits_to_class_probabilities(logits: torch.Tensor) -> torch.Tensor:
    return threshold_probabilities_to_class_probabilities(logits_to_threshold_probabilities(logits))


def decode_threshold_count(logits: torch.Tensor, thresholds: list[float] | tuple[float, ...] | None = None) -> torch.Tensor:
    probs = logits_to_threshold_probabilities(logits)
    if thresholds is None:
        threshold_tensor = probs.new_full((probs.shape[1],), 0.5)
    else:
        if len(thresholds) != probs.shape[1]:
            raise ValueError(f"Expected {probs.shape[1]} thresholds, got {len(thresholds)}.")
        threshold_tensor = probs.new_tensor(list(thresholds))
    return (probs > threshold_tensor.unsqueeze(0)).sum(dim=1)


class CORNLoss(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        if int(num_classes) < 2:
            raise ValueError("CORNLoss expects num_classes >= 2.")
        self.num_classes = int(num_classes)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
        if logits.ndim != 2 or logits.shape[1] != self.num_classes - 1:
            raise ValueError(
                f"Expected CORN logits with shape [N, {self.num_classes - 1}], got {tuple(logits.shape)}."
            )
        labels = labels.long().view(-1)
        targets, active_mask = build_corn_targets(labels, self.num_classes)

        loss_sum = logits.new_zeros(())
        sample_count = logits.new_zeros(())
        task_losses = logits.new_zeros(self.num_classes - 1)
        task_counts = logits.new_zeros(self.num_classes - 1)
        for threshold_idx in range(self.num_classes - 1):
            active = active_mask[:, threshold_idx]
            if not active.any():
                continue
            threshold_loss = F.binary_cross_entropy_with_logits(
                logits[active, threshold_idx],
                targets[active, threshold_idx],
                reduction="sum",
            )
            loss_sum = loss_sum + threshold_loss
            count = active.sum()
            sample_count = sample_count + count
            task_losses[threshold_idx] = threshold_loss / count.clamp_min(1)
            task_counts[threshold_idx] = count.float()

        normalized_loss = loss_sum / sample_count.clamp_min(1)
        threshold_probabilities = logits_to_threshold_probabilities(logits)
        class_probabilities = threshold_probabilities_to_class_probabilities(threshold_probabilities)
        return {
            "loss": normalized_loss,
            "loss_corn_main": normalized_loss,
            "threshold_probabilities": threshold_probabilities,
            "class_probabilities": class_probabilities,
            "task_losses": task_losses,
            "task_counts": task_counts,
        }

