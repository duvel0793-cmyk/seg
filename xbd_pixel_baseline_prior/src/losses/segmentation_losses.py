from __future__ import annotations

import torch
from torch import nn


class SoftDiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        target = target.float()

        probabilities = probabilities.flatten(start_dim=1)
        target = target.flatten(start_dim=1)

        intersection = (probabilities * target).sum(dim=1)
        denominator = probabilities.sum(dim=1) + target.sum(dim=1)
        dice_score = (2.0 * intersection + self.eps) / (denominator + self.eps)
        return 1.0 - dice_score.mean()


class BCEWithDiceLoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, target.float())
        dice_loss = self.dice(logits, target.float())
        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)
