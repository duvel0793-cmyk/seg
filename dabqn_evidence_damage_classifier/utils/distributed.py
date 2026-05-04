from __future__ import annotations


def is_distributed_available() -> bool:
    return False


def get_rank() -> int:
    return 0


def is_main_process() -> bool:
    return True
