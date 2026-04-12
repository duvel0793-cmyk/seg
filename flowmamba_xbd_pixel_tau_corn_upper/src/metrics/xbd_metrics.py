"""xBD evaluation helpers."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from ..models.modules.ordinal_utils import combine_damage_and_loc
from .confusion import ConfusionMatrix


class XBDMetrics:
    """Track loc/damage metrics in a FlowMamba-style report."""

    def __init__(self, ignore_index: int = 255) -> None:
        self.ignore_index = int(ignore_index)
        self.loc_confusion = ConfusionMatrix(num_classes=2, class_names=["background", "building"])
        self.damage_confusion = ConfusionMatrix(
            num_classes=4,
            ignore_index=self.ignore_index,
            class_names=["no_damage", "minor", "major", "destroyed"],
        )
        self.overall_confusion = ConfusionMatrix(
            num_classes=5,
            class_names=["background", "no_damage", "minor", "major", "destroyed"],
        )
        self.tau_history: List[np.ndarray] = []
        self.valid_instances = 0
        self.valid_building_pixels = 0
        self.instance_pixel_counts: List[int] = []
        self.instance_label_source_counts = {"polygon_subtype": 0, "pixel_majority": 0}
        self.instance_pool_source_counts: Dict[str, int] = {}
        self.tau_phase_counts: Dict[str, int] = {}
        self.corr_tau_history: List[float] = []
        self.corr_raw_tau_history: List[float] = []
        self.tau_bin_accumulator: Dict[int, Dict[str, float]] = {}

    def update(
        self,
        loc_target: torch.Tensor,
        loc_pred: torch.Tensor,
        damage_rank_target: torch.Tensor,
        damage_rank_pred: torch.Tensor,
        tau_values: torch.Tensor | None = None,
        tau_phase: str | None = None,
        corr_tau_difficulty: float | None = None,
        corr_raw_tau_difficulty: float | None = None,
        tau_by_difficulty_bin: List[Dict[str, object]] | None = None,
        instance_stats: Dict[str, object] | None = None,
    ) -> None:
        self.loc_confusion.update(loc_target, loc_pred)

        valid_damage = damage_rank_target != self.ignore_index
        if valid_damage.any():
            self.damage_confusion.update(damage_rank_target[valid_damage], damage_rank_pred[valid_damage])

        overall_target = torch.zeros_like(loc_target, dtype=torch.long)
        overall_target[valid_damage] = damage_rank_target[valid_damage] + 1
        overall_pred = combine_damage_and_loc(loc_pred=loc_pred, damage_rank_pred=damage_rank_pred)
        self.overall_confusion.update(overall_target, overall_pred)

        if tau_values is not None and torch.is_tensor(tau_values):
            self.tau_history.append(tau_values.detach().cpu().numpy().reshape(-1))
        if tau_phase:
            self.tau_phase_counts[tau_phase] = self.tau_phase_counts.get(tau_phase, 0) + 1
        if corr_tau_difficulty is not None:
            self.corr_tau_history.append(float(corr_tau_difficulty))
        if corr_raw_tau_difficulty is not None:
            self.corr_raw_tau_history.append(float(corr_raw_tau_difficulty))
        if tau_by_difficulty_bin:
            for bin_info in tau_by_difficulty_bin:
                bin_idx = int(bin_info["bin"])
                record = self.tau_bin_accumulator.setdefault(bin_idx, {"count": 0.0, "tau_sum": 0.0})
                record["count"] += float(bin_info["count"])
                record["tau_sum"] += float(bin_info["tau_mean"]) * float(bin_info["count"])

        self.valid_building_pixels += int(valid_damage.sum().item())

        if instance_stats:
            self.valid_instances += int(instance_stats.get("valid_instances", 0))
            self.instance_pixel_counts.extend(instance_stats.get("instance_pixel_counts", []) or [])
            for key, value in (instance_stats.get("label_source_counts", {}) or {}).items():
                self.instance_label_source_counts[key] = self.instance_label_source_counts.get(key, 0) + int(value)
            pool_source = str(instance_stats.get("pool_source", "unknown"))
            self.instance_pool_source_counts[pool_source] = self.instance_pool_source_counts.get(pool_source, 0) + 1

    def compute(self) -> Dict[str, object]:
        loc_f1 = float(self.loc_confusion.f1_per_class()[1])
        damage_f1 = self.damage_confusion.f1_per_class().astype(np.float64)
        if np.any(damage_f1 <= 0):
            harmonic_mean_f1 = 0.0
        else:
            harmonic_mean_f1 = float(len(damage_f1) / np.sum(1.0 / damage_f1))
        overall_score = 0.3 * loc_f1 + 0.7 * harmonic_mean_f1

        if self.tau_history:
            tau_array = np.concatenate(self.tau_history, axis=0)
            tau_stats = {
                "mean": float(np.mean(tau_array)),
                "std": float(np.std(tau_array)),
                "min": float(np.min(tau_array)),
                "max": float(np.max(tau_array)),
                "corr_tau_difficulty": float(np.mean(self.corr_tau_history)) if self.corr_tau_history else 0.0,
                "corr_raw_tau_difficulty": float(np.mean(self.corr_raw_tau_history)) if self.corr_raw_tau_history else 0.0,
                "tau_phase_counts": self.tau_phase_counts,
                "tau_by_difficulty_bin": [
                    {
                        "bin": int(bin_idx),
                        "count": int(record["count"]),
                        "tau_mean": float(record["tau_sum"] / max(record["count"], 1.0)),
                    }
                    for bin_idx, record in sorted(self.tau_bin_accumulator.items())
                ],
            }
        else:
            tau_stats = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "corr_tau_difficulty": 0.0,
                "corr_raw_tau_difficulty": 0.0,
                "tau_phase_counts": self.tau_phase_counts,
                "tau_by_difficulty_bin": [],
            }

        instance_aux_stats = {
            "valid_instances": int(self.valid_instances),
            "pixel_count_mean": float(np.mean(self.instance_pixel_counts)) if self.instance_pixel_counts else 0.0,
            "pixel_count_min": int(np.min(self.instance_pixel_counts)) if self.instance_pixel_counts else 0,
            "pixel_count_max": int(np.max(self.instance_pixel_counts)) if self.instance_pixel_counts else 0,
            "label_source_counts": self.instance_label_source_counts,
            "pool_source_counts": self.instance_pool_source_counts,
        }

        return {
            "F1_loc": loc_f1,
            "F1_subcls": damage_f1.tolist(),
            "F1_bda": harmonic_mean_f1,
            "F1_oa": overall_score,
            "loc_confusion": self.loc_confusion.matrix.tolist(),
            "damage_confusion": self.damage_confusion.matrix.tolist(),
            "overall_confusion": self.overall_confusion.matrix.tolist(),
            "per_class_metrics": {
                "loc": self.loc_confusion.summary(),
                "damage": self.damage_confusion.summary(),
                "overall": self.overall_confusion.summary(),
            },
            "valid_building_pixels": int(self.valid_building_pixels),
            "tau": tau_stats,
            "instance_aux": instance_aux_stats,
        }

