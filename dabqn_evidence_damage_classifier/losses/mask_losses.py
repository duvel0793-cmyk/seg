from __future__ import annotations

import torch
import torch.nn.functional as F


def sigmoid_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0 or targets.numel() == 0:
        return logits.new_tensor(0.0)
    return F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="mean")


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0 or targets.numel() == 0:
        return logits.new_tensor(0.0)
    probs = logits.sigmoid().flatten(1)
    targets = targets.float().flatten(1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (union + 1.0))
    return dice.mean()
