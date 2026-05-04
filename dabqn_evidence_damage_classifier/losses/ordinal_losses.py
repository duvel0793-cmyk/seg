from __future__ import annotations

import torch
import torch.nn.functional as F


def corn_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0 or labels.numel() == 0:
        return logits.new_tensor(0.0)
    num_classes = logits.shape[1] + 1
    levels = [(labels > threshold).float() for threshold in range(num_classes - 1)]
    level_targets = torch.stack(levels, dim=1)
    return F.binary_cross_entropy_with_logits(logits.float(), level_targets, reduction="mean")
