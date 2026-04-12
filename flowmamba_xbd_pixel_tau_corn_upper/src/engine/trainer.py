"""Single-card trainer."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast

from ..engine.evaluator import Evaluator
from ..utils import build_checkpoint_metadata, summarize_backend_metadata
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.misc import AverageMeter, ensure_dir, move_batch_to_device


class Trainer:
    def __init__(
        self,
        cfg,
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        device,
        logger,
        resolved_config_path: str | Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = logger
        self.resolved_config_path = resolved_config_path
        self.use_amp = bool(cfg.get("amp", False)) and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.evaluator = Evaluator(cfg, model, criterion, device, logger)
        self.output_dir = ensure_dir(cfg["output_dir"])
        self.checkpoint_dir = ensure_dir(self.output_dir / "checkpoints")
        self.validation_dir = ensure_dir(self.output_dir / "validation")

    def _checkpoint_state(self, epoch: int, best_metrics, extra: dict | None = None) -> dict:
        state = {
            "epoch": epoch,
            "best_metrics": best_metrics,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict() if self.use_amp else None,
            "config": self.cfg,
            "metadata": build_checkpoint_metadata(
                cfg=self.cfg,
                model=self.model,
                resolved_config_path=self.resolved_config_path,
            ),
        }
        if extra:
            state.update(extra)
        return state

    def _save(self, epoch: int, best_metrics, path: Path) -> None:
        save_checkpoint(self._checkpoint_state(epoch=epoch, best_metrics=best_metrics), path)

    def _load_start_state(self) -> tuple[int, dict]:
        train_cfg = self.cfg["train"]
        current_metadata = build_checkpoint_metadata(self.cfg, self.model, resolved_config_path=self.resolved_config_path)
        best_metrics = {"F1_oa": -1.0, "F1_bda": -1.0}
        start_epoch = 0

        resume_path = str(train_cfg.get("resume", "") or "")
        finetune_path = str(train_cfg.get("finetune_from", "") or "")
        if resume_path:
            state = load_checkpoint(
                path=resume_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                map_location=str(self.device),
                expected_metadata=current_metadata,
                strict_metadata=bool(train_cfg.get("strict_resume_metadata", True)),
                logger=self.logger,
                load_weights_only=False,
            )
            start_epoch = int(state.get("epoch", -1)) + 1
            best_metrics = state.get("best_metrics") or {"F1_oa": float(state.get("best_metric", -1.0)), "F1_bda": -1.0}
            self.logger.info("Resumed from %s at epoch=%d", resume_path, start_epoch)
        elif finetune_path:
            load_checkpoint(
                path=finetune_path,
                model=self.model,
                map_location=str(self.device),
                expected_metadata=current_metadata,
                strict_metadata=bool(train_cfg.get("strict_finetune_metadata", False)),
                logger=self.logger,
                load_weights_only=True,
            )
            self.logger.info("Loaded model weights for finetune from %s", finetune_path)
        return start_epoch, best_metrics

    def train(self) -> None:
        train_cfg = self.cfg["train"]
        start_epoch, best_metrics = self._load_start_state()

        self.logger.info("Backend summary=%s", summarize_backend_metadata(self.model.backbone.get_metadata()))

        for epoch in range(start_epoch, int(train_cfg["epochs"])):
            self.model.train()
            meters = {
                "loc_loss": AverageMeter(),
                "pixel_corn_main": AverageMeter(),
                "pixel_corn_soft": AverageMeter(),
                "tau_reg": AverageMeter(),
                "instance_corn_aux": AverageMeter(),
                "total_loss": AverageMeter(),
            }
            last_stats = {}

            for batch_idx, batch in enumerate(self.train_loader):
                max_train_batches = train_cfg.get("max_train_batches")
                if max_train_batches is not None and batch_idx >= int(max_train_batches):
                    break

                batch = move_batch_to_device(batch, self.device)
                self.optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=self.use_amp):
                    outputs = self.model(
                        batch,
                        epoch=epoch,
                        enable_instance_aux=True,
                    )
                    loss_dict = self.criterion(outputs, batch, epoch=epoch, is_train=True)
                    total_loss = loss_dict["total_loss"]

                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_clip = float(train_cfg.get("grad_clip", 0.0))
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                batch_size = batch["pre_image"].shape[0]
                for name, meter in meters.items():
                    meter.update(float(loss_dict[name].detach().item()), n=batch_size)

                last_stats = {
                    "tau_mean": loss_dict["tau_mean"],
                    "tau_std": loss_dict["tau_std"],
                    "tau_min": loss_dict["tau_min"],
                    "tau_max": loss_dict["tau_max"],
                    "corr_tau_difficulty": loss_dict["corr_tau_difficulty"],
                    "corr_raw_tau_difficulty": loss_dict["corr_raw_tau_difficulty"],
                    "tau_phase": loss_dict["tau_phase"],
                    "corn_soft_enabled": loss_dict["corn_soft_enabled"],
                    "valid_building_pixels": loss_dict["valid_building_pixels"],
                    "instance_aux_enabled": loss_dict["instance_aux_enabled"],
                }

                if batch_idx % int(self.cfg.get("print_freq", 10)) == 0:
                    lr = float(self.optimizer.param_groups[0]["lr"])
                    self.logger.info(
                        "epoch=%d step=%d lr=%.6e total_loss=%.4f loc_loss=%.4f pixel_corn_main=%.4f pixel_corn_soft=%.4f "
                        "tau_reg=%.6f instance_aux=%.4f tau_phase=%s tau_mean=%.4f tau_std=%.4f corr_tau_diff=%.4f corr_raw_tau_diff=%.4f",
                        epoch,
                        batch_idx,
                        lr,
                        meters["total_loss"].avg,
                        meters["loc_loss"].avg,
                        meters["pixel_corn_main"].avg,
                        meters["pixel_corn_soft"].avg,
                        meters["tau_reg"].avg,
                        meters["instance_corn_aux"].avg,
                        last_stats["tau_phase"],
                        last_stats["tau_mean"],
                        last_stats["tau_std"],
                        last_stats["corr_tau_difficulty"],
                        last_stats["corr_raw_tau_difficulty"],
                    )

            if self.scheduler is not None:
                self.scheduler.step()

            latest_path = self.checkpoint_dir / "latest.pth"
            self._save(epoch=epoch, best_metrics=best_metrics, path=latest_path)
            save_freq = int(max(train_cfg.get("save_freq", 0), 0))
            if save_freq > 0 and (epoch + 1) % save_freq == 0:
                self._save(epoch=epoch, best_metrics=best_metrics, path=self.checkpoint_dir / f"epoch_{epoch:03d}.pth")

            self.logger.info(
                "epoch=%d train_summary lr=%.6e total_loss=%.4f loc_loss=%.4f pixel_corn_main=%.4f pixel_corn_soft=%.4f "
                "tau_reg=%.6f instance_aux=%.4f tau_phase=%s best_f1_oa=%.4f best_f1_bda=%.4f",
                epoch,
                float(self.optimizer.param_groups[0]["lr"]),
                meters["total_loss"].avg,
                meters["loc_loss"].avg,
                meters["pixel_corn_main"].avg,
                meters["pixel_corn_soft"].avg,
                meters["tau_reg"].avg,
                meters["instance_corn_aux"].avg,
                last_stats.get("tau_phase", "unknown"),
                float(best_metrics.get("F1_oa", -1.0)),
                float(best_metrics.get("F1_bda", -1.0)),
            )

            if (epoch + 1) % int(train_cfg.get("val_interval", 1)) == 0:
                val_dir = self.validation_dir / f"epoch_{epoch:03d}"
                val_result = self.evaluator.evaluate(
                    self.val_loader,
                    max_batches=train_cfg.get("max_val_batches"),
                    epoch=epoch,
                    save_dir=val_dir,
                )
                self.logger.info(
                    "val epoch=%d F1_oa=%.4f F1_loc=%.4f F1_bda=%.4f tau_mean=%.4f tau_std=%.4f valid_instances=%d best_ckpts={oa:%s,bda:%s}",
                    epoch,
                    val_result["F1_oa"],
                    val_result["F1_loc"],
                    val_result["F1_bda"],
                    val_result["tau"]["mean"],
                    val_result["tau"]["std"],
                    val_result["instance_aux"]["valid_instances"],
                    self.checkpoint_dir / "best_f1_oa.pth",
                    self.checkpoint_dir / "best_f1_bda.pth",
                )

                if float(val_result["F1_oa"]) > float(best_metrics.get("F1_oa", -1.0)):
                    best_metrics["F1_oa"] = float(val_result["F1_oa"])
                    self._save(epoch=epoch, best_metrics=best_metrics, path=self.checkpoint_dir / "best_f1_oa.pth")
                if float(val_result["F1_bda"]) > float(best_metrics.get("F1_bda", -1.0)):
                    best_metrics["F1_bda"] = float(val_result["F1_bda"])
                    self._save(epoch=epoch, best_metrics=best_metrics, path=self.checkpoint_dir / "best_f1_bda.pth")
                self._save(epoch=epoch, best_metrics=best_metrics, path=latest_path)

        self.logger.info(
            "Training finished. best_F1_oa=%.4f best_F1_bda=%.4f",
            float(best_metrics.get("F1_oa", -1.0)),
            float(best_metrics.get("F1_bda", -1.0)),
        )

