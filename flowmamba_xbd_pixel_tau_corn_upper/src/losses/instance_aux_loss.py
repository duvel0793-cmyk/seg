"""Instance auxiliary CORN loss."""

from __future__ import annotations

import torch.nn as nn

from .corn_loss import CORNLoss


class InstanceAuxLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.corn = CORNLoss(ignore_index=-1)

    def forward(self, instance_logits, instance_targets):
        if instance_logits is None or instance_targets is None:
            return instance_logits.new_tensor(0.0) if instance_logits is not None else 0.0
        return self.corn(instance_logits, instance_targets)
