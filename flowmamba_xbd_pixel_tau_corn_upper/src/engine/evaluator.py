"""Validation loop with full-image and tiled inference support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

from ..metrics import XBDMetrics
from ..utils.misc import AverageMeter, ensure_dir, move_batch_to_device
from ..utils.vis_predictions import save_prediction_bundle


class Evaluator:
    def __init__(self, cfg, model, criterion, device, logger=None) -> None:
        self.cfg = cfg
        self.model = model
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.validation_cfg = cfg.get("validation", {})
        self.debug_limit = int(self.validation_cfg.get("max_debug_images", 8))

    def _iter_single_sample_batches(self, batch: Dict[str, object]) -> Iterable[Dict[str, object]]:
        batch_size = batch["pre_image"].shape[0]
        for idx in range(batch_size):
            yield {
                "pre_image": batch["pre_image"][idx : idx + 1],
                "post_image": batch["post_image"][idx : idx + 1],
                "post_target": batch["post_target"][idx : idx + 1],
                "loc_target": batch["loc_target"][idx : idx + 1],
                "damage_rank_target": batch["damage_rank_target"][idx : idx + 1],
                "polygons": [batch["polygons"][idx]],
                "image_id": [batch["image_id"][idx]],
                "meta": [batch["meta"][idx]],
            }

    @staticmethod
    def _tile_positions(full_size: int, tile_size: int, stride: int) -> List[int]:
        if tile_size >= full_size:
            return [0]
        positions = list(range(0, max(full_size - tile_size, 0) + 1, stride))
        if positions[-1] != full_size - tile_size:
            positions.append(full_size - tile_size)
        return positions

    @staticmethod
    def _build_blend_weight(tile_h: int, tile_w: int, mode: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if mode == "hann" and tile_h > 1 and tile_w > 1:
            wy = torch.hann_window(tile_h, periodic=False, dtype=dtype, device=device)
            wx = torch.hann_window(tile_w, periodic=False, dtype=dtype, device=device)
            weight = (wy[:, None] * wx[None, :]).clamp_min(1.0e-3)
        else:
            weight = torch.ones((tile_h, tile_w), dtype=dtype, device=device)
        return weight.view(1, 1, tile_h, tile_w)

    def _merge_window_outputs(self, sample_batch: Dict[str, object], epoch: int, enable_instance_aux: bool) -> Dict[str, object]:
        pre_image = sample_batch["pre_image"]
        post_image = sample_batch["post_image"]
        _, _, full_h, full_w = post_image.shape

        tile_size = self.validation_cfg.get("tile_size", self.cfg["data"]["image_size"])
        tile_stride = self.validation_cfg.get("tile_stride", tile_size)
        tile_h, tile_w = int(tile_size[0]), int(tile_size[1])
        stride_h, stride_w = int(tile_stride[0]), int(tile_stride[1])
        blend_mode = str(self.validation_cfg.get("merge_window", "hann")).lower()

        y_positions = self._tile_positions(full_h, tile_h, stride_h)
        x_positions = self._tile_positions(full_w, tile_w, stride_w)

        accumulators = None
        weight = None
        last_output = None

        for top in y_positions:
            for left in x_positions:
                tile_batch = {
                    "pre_image": pre_image[:, :, top : top + tile_h, left : left + tile_w],
                    "post_image": post_image[:, :, top : top + tile_h, left : left + tile_w],
                    "polygons": [[]],
                }
                tile_output = self.model(tile_batch, epoch=epoch, enable_instance_aux=False)
                last_output = tile_output

                if accumulators is None:
                    loc_channels = tile_output["loc_logits"].shape[1]
                    ord_channels = tile_output["ordinal_logits"].shape[1]
                    feat_channels = tile_output["instance_feature_map"].shape[1]
                    accumulators = {
                        "loc_logits": torch.zeros((1, loc_channels, full_h, full_w), device=self.device, dtype=tile_output["loc_logits"].dtype),
                        "ordinal_logits": torch.zeros((1, ord_channels, full_h, full_w), device=self.device, dtype=tile_output["ordinal_logits"].dtype),
                        "raw_ordinal_logits": torch.zeros((1, ord_channels, full_h, full_w), device=self.device, dtype=tile_output["raw_ordinal_logits"].dtype),
                        "tau_values": torch.zeros((1, ord_channels, full_h, full_w), device=self.device, dtype=tile_output["tau_values"].dtype),
                        "adaptive_tau": torch.zeros((1, ord_channels, full_h, full_w), device=self.device, dtype=tile_output["adaptive_tau"].dtype),
                        "raw_tau": torch.zeros((1, ord_channels, full_h, full_w), device=self.device, dtype=tile_output["raw_tau"].dtype),
                        "fixed_tau": torch.zeros((1, ord_channels, full_h, full_w), device=self.device, dtype=tile_output["fixed_tau"].dtype),
                        "instance_feature_map": torch.zeros((1, feat_channels, full_h, full_w), device=self.device, dtype=tile_output["instance_feature_map"].dtype),
                        "weight": torch.zeros((1, 1, full_h, full_w), device=self.device, dtype=tile_output["loc_logits"].dtype),
                    }
                    weight = self._build_blend_weight(
                        tile_h=tile_output["loc_logits"].shape[-2],
                        tile_w=tile_output["loc_logits"].shape[-1],
                        mode=blend_mode,
                        device=self.device,
                        dtype=tile_output["loc_logits"].dtype,
                    )

                y_slice = slice(top, top + tile_output["loc_logits"].shape[-2])
                x_slice = slice(left, left + tile_output["loc_logits"].shape[-1])
                for key in [
                    "loc_logits",
                    "ordinal_logits",
                    "raw_ordinal_logits",
                    "tau_values",
                    "adaptive_tau",
                    "raw_tau",
                    "fixed_tau",
                    "instance_feature_map",
                ]:
                    accumulators[key][:, :, y_slice, x_slice] += tile_output[key] * weight
                accumulators["weight"][:, :, y_slice, x_slice] += weight

        if accumulators is None or last_output is None:
            raise RuntimeError("Sliding-window inference produced no tiles.")

        weight_sum = accumulators["weight"].clamp_min(1.0e-6)
        merged = {key: value / weight_sum for key, value in accumulators.items() if key != "weight"}
        merged["damage_rank_pred"] = self.model.pixel_corn_head.decode(merged["ordinal_logits"])
        merged["tau_phase"] = last_output["tau_phase"]
        merged["corn_soft_enabled"] = last_output["corn_soft_enabled"]
        merged["tau_stats"] = {
            "mean": float(merged["tau_values"].mean().item()),
            "std": float(merged["tau_values"].std(unbiased=False).item()),
            "min": float(merged["tau_values"].min().item()),
            "max": float(merged["tau_values"].max().item()),
        }
        merged["backbone_name"] = last_output["backbone_name"]
        merged["backbone_reason"] = last_output["backbone_reason"]
        merged["backbone_metadata"] = last_output["backbone_metadata"]
        merged["feature_channels"] = last_output["feature_channels"]
        merged["feature_strides"] = last_output["feature_strides"]

        if enable_instance_aux:
            instance_pool = self.model.instance_pooling(
                ordinal_logits=merged["ordinal_logits"],
                fused_feature=merged["instance_feature_map"],
                polygons=sample_batch.get("polygons", [[]]),
                damage_targets=sample_batch.get("damage_rank_target"),
            )
            merged["instance_pool"] = instance_pool
            merged["instance_logits"] = (
                self.model.instance_aux_head(instance_pool["pooled_representations"])
                if instance_pool["pooled_representations"] is not None
                else None
            )
        else:
            merged["instance_pool"] = {
                "pooled_representations": None,
                "targets": None,
                "valid_instances": 0,
                "instance_pixel_counts": [],
                "label_source_counts": {"polygon_subtype": 0, "pixel_majority": 0},
                "pool_source": "disabled",
            }
            merged["instance_logits"] = None
        return merged

    def _forward_validation_sample(self, sample_batch: Dict[str, object], epoch: int, enable_instance_aux: bool) -> Dict[str, object]:
        mode = str(self.validation_cfg.get("mode", "center_crop")).lower()
        if mode == "tiled":
            return self._merge_window_outputs(sample_batch=sample_batch, epoch=epoch, enable_instance_aux=enable_instance_aux)
        return self.model(sample_batch, epoch=epoch, enable_instance_aux=enable_instance_aux)

    def _save_metrics_artifacts(self, result: Dict[str, object], save_dir: Path) -> None:
        ensure_dir(save_dir)
        metrics_json = {
            "F1_loc": result["F1_loc"],
            "F1_subcls": result["F1_subcls"],
            "F1_bda": result["F1_bda"],
            "F1_oa": result["F1_oa"],
            "valid_building_pixels": result["valid_building_pixels"],
            "losses": result["losses"],
        }
        (save_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")
        (save_dir / "per_class_metrics.json").write_text(
            json.dumps(result["per_class_metrics"], indent=2), encoding="utf-8"
        )
        (save_dir / "tau_stats.json").write_text(json.dumps(result["tau"], indent=2), encoding="utf-8")
        (save_dir / "instance_aux_stats.json").write_text(
            json.dumps(result["instance_aux"], indent=2), encoding="utf-8"
        )
        np.savez_compressed(
            save_dir / "confusion_matrices.npz",
            loc=np.asarray(result["loc_confusion"], dtype=np.int64),
            damage=np.asarray(result["damage_confusion"], dtype=np.int64),
            overall=np.asarray(result["overall_confusion"], dtype=np.int64),
        )

    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
        max_batches: int | None = None,
        epoch: int = 0,
        save_dir: str | Path | None = None,
    ) -> Dict[str, object]:
        self.model.eval()
        metrics = XBDMetrics(ignore_index=int(self.cfg["data"]["ignore_index"]))
        meters = {
            "loc_loss": AverageMeter(),
            "pixel_corn_main": AverageMeter(),
            "pixel_corn_soft": AverageMeter(),
            "tau_reg": AverageMeter(),
            "instance_corn_aux": AverageMeter(),
            "total_loss": AverageMeter(),
        }

        output_dir = ensure_dir(save_dir) if save_dir is not None else None
        debug_dir = ensure_dir(Path(output_dir) / "debug_vis") if output_dir is not None else None
        debug_saved = 0
        enable_instance_aux = bool(self.cfg["model"].get("instance_aux_on_val", False))

        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            batch = move_batch_to_device(batch, self.device)

            for sample_batch in self._iter_single_sample_batches(batch):
                outputs = self._forward_validation_sample(sample_batch, epoch=epoch, enable_instance_aux=enable_instance_aux)
                loss_dict = self.criterion(outputs, sample_batch, epoch=epoch, is_train=False)

                for name, meter in meters.items():
                    meter.update(float(loss_dict[name].detach().item()), n=1)

                loc_pred = outputs["loc_logits"].argmax(dim=1)
                metrics.update(
                    loc_target=sample_batch["loc_target"],
                    loc_pred=loc_pred,
                    damage_rank_target=sample_batch["damage_rank_target"],
                    damage_rank_pred=outputs["damage_rank_pred"],
                    tau_values=outputs["tau_values"],
                    tau_phase=loss_dict["tau_phase"],
                    corr_tau_difficulty=loss_dict["corr_tau_difficulty"],
                    corr_raw_tau_difficulty=loss_dict["corr_raw_tau_difficulty"],
                    tau_by_difficulty_bin=loss_dict["tau_by_difficulty_bin"],
                    instance_stats={
                        "valid_instances": outputs["instance_pool"]["valid_instances"],
                        "instance_pixel_counts": outputs["instance_pool"]["instance_pixel_counts"],
                        "label_source_counts": outputs["instance_pool"]["label_source_counts"],
                        "pool_source": outputs["instance_pool"]["pool_source"],
                    },
                )

                if debug_dir is not None and debug_saved < self.debug_limit:
                    image_id = sample_batch["image_id"][0]
                    save_prediction_bundle(
                        save_dir=debug_dir,
                        sample=sample_batch,
                        outputs=outputs,
                        mean=self.cfg["data"]["mean"],
                        std=self.cfg["data"]["std"],
                        stem=f"{debug_saved:02d}_{image_id}",
                    )
                    debug_saved += 1

        result = metrics.compute()
        result["losses"] = {name: meter.avg for name, meter in meters.items()}
        if output_dir is not None:
            self._save_metrics_artifacts(result, Path(output_dir))
        return result

