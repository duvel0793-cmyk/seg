"""Loss builders."""

from .total_loss import TotalLoss


def build_criterion(cfg):
    return TotalLoss(cfg)


__all__ = ["TotalLoss", "build_criterion"]
