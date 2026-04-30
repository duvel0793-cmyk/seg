"""Cross-entropy classification head."""

from __future__ import annotations

import torch
import torch.nn as nn


class CEHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
