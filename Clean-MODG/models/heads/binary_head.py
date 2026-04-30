"""Auxiliary binary head for coarse damage grouping."""

from __future__ import annotations

import torch
import torch.nn as nn


class BinaryHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
