from __future__ import annotations

import math
from typing import Any


class WarmupCosineScheduler:
    def __init__(self, optimizer, *, total_epochs: int, warmup_epochs: int, min_lr_ratio: float = 0.01) -> None:
        self.optimizer = optimizer
        self.total_epochs = int(total_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.current_epoch = 0

    def _multiplier(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return float(epoch + 1) / max(self.warmup_epochs, 1)
        if self.total_epochs <= self.warmup_epochs:
            return 1.0
        progress = float(epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return self.min_lr_ratio + ((1.0 - self.min_lr_ratio) * cosine)

    def step(self, epoch: int) -> None:
        self.current_epoch = int(epoch)
        mult = self._multiplier(self.current_epoch)
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * mult

    def state_dict(self) -> dict[str, Any]:
        return {"current_epoch": self.current_epoch}

