"""Instance auxiliary CORN loss."""

from __future__ import annotations

import torch
import torch.nn as nn

from .corn_loss import CORNLoss


class InstanceCORNLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.corn = CORNLoss(ignore_index=-1)

    def forward(self, instance_logits, instance_targets):
        if instance_logits is None or instance_targets is None:
            if instance_logits is not None and torch.is_tensor(instance_logits):
                return instance_logits.new_tensor(0.0)
            return torch.tensor(0.0)
        return self.corn(instance_logits, instance_targets)

