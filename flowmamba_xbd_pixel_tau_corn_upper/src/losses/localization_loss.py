"""Localization loss."""

from __future__ import annotations

import torch.nn as nn


class LocalizationLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        return self.loss(logits, target)
