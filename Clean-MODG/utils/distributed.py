"""Single-GPU friendly distributed stubs."""

from __future__ import annotations

import torch


def is_distributed() -> bool:
    return False


def get_rank() -> int:
    return 0


def get_world_size() -> int:
    return 1


def is_main_process() -> bool:
    return True


def synchronize() -> None:
    return None


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
