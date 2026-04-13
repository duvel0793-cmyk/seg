"""Minimal hook system for the training loop."""

from __future__ import annotations


class HookBase:
    """No-op hook base class for extensible training events."""

    def before_epoch(self, trainer, epoch: int) -> None:
        del trainer, epoch

    def after_epoch(self, trainer, epoch: int, metrics) -> None:
        del trainer, epoch, metrics

    def before_step(self, trainer, step: int) -> None:
        del trainer, step

    def after_step(self, trainer, step: int, outputs) -> None:
        del trainer, step, outputs

