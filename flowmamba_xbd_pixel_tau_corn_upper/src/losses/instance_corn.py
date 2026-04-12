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
            if torch.is_tensor(instance_logits):
                return instance_logits.new_zeros(())
            if torch.is_tensor(instance_targets):
                return instance_targets.new_zeros((), dtype=torch.float32)
            return torch.zeros((), dtype=torch.float32)

        return self.corn(instance_logits, instance_targets)