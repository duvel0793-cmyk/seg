"""Regularizers for the learnable tau module."""

from __future__ import annotations

import torch


def tau_center_regularization(tau: torch.Tensor) -> torch.Tensor:
    """Keep tau close to 1.0."""
    return ((tau - 1.0) ** 2).mean()


def tau_diff_regularization(tau: torch.Tensor) -> torch.Tensor:
    """Keep neighboring threshold taus similar."""
    if tau.numel() <= 1:
        return tau.new_zeros(())
    return ((tau[1:] - tau[:-1]) ** 2).mean()

