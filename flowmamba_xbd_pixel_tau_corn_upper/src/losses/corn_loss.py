"""CORN ordinal losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.modules.ordinal_utils import rank_targets_to_corn


class CORNLoss(nn.Module):
    """Binary cross entropy over ordinal decision boundaries."""

    def __init__(self, ignore_index: int = 255) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 4:
            valid_mask = labels != self.ignore_index
            if not valid_mask.any():
                return logits.new_tensor(0.0)
            logits = logits.permute(0, 2, 3, 1)[valid_mask]
            labels = labels[valid_mask]
        elif logits.ndim == 2:
            if labels.numel() == 0:
                return logits.new_tensor(0.0)
        else:
            raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")

        num_classes = logits.shape[-1] + 1
        corn_targets = rank_targets_to_corn(labels.long(), num_classes=num_classes).to(logits.device, logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, corn_targets, reduction="mean")
