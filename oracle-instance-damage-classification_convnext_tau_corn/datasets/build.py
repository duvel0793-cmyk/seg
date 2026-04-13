"""Dataset and dataloader builders."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets.label_mapping import LABEL_TO_DAMAGE
from datasets.xbd_oracle_instance_dataset import XBDOracleInstanceDataset


def build_dataset(config: Dict[str, Any], split: str) -> XBDOracleInstanceDataset:
    """Build one xBD oracle instance dataset."""
    data_cfg = config["data"]
    return XBDOracleInstanceDataset(
        data_root=data_cfg["data_root"],
        split=split,
        train_split_file=data_cfg.get("train_split_file", ""),
        val_split_file=data_cfg.get("val_split_file", ""),
        image_size=data_cfg["image_size"],
        crop_context_ratio=data_cfg["crop_context_ratio"],
        min_crop_size=data_cfg["min_crop_size"],
        use_vertical_flip=data_cfg.get("use_vertical_flip", False),
        use_color_jitter=data_cfg.get("use_color_jitter", True),
    )


def oracle_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate tensors while keeping metadata as a list of dicts."""
    return {
        "pre_image": torch.stack([item["pre_image"] for item in batch], dim=0),
        "post_image": torch.stack([item["post_image"] for item in batch], dim=0),
        "mask": torch.stack([item["mask"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
        "meta": [item["meta"] for item in batch],
    }


def _log_loader_stats(message: str, logger=None) -> None:
    if logger is not None:
        logger.info(message)
    else:
        print(message)


def build_dataloader(
    config: Dict[str, Any],
    split: str,
    shuffle: bool | None = None,
    logger=None,
) -> DataLoader:
    """Build a dataloader for the given split."""
    dataset = build_dataset(config, split)
    runtime_cfg = config["runtime"]
    data_cfg = config["data"]
    label_counts = Counter(int(item["label"]) for item in dataset.records)

    use_weighted_sampler = bool(runtime_cfg.get("use_weighted_sampler", False)) and split == "train"
    sampler = None
    if shuffle is None:
        shuffle = split == "train"
    if use_weighted_sampler:
        sampler_power = float(runtime_cfg.get("sampler_power", 0.5))
        sample_weights = [
            1.0 / (float(label_counts[int(item["label"])]) ** sampler_power)
            for item in dataset.records
        ]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    _log_loader_stats(
        f"[DataLoader] split={split} num_samples={len(dataset)} "
        f"class_counts={{"
        + ", ".join(
            f"{LABEL_TO_DAMAGE.get(label, label)}: {count}"
            for label, count in sorted(label_counts.items())
        )
        + "} "
        f"weighted_sampler={use_weighted_sampler} "
        f"sampler_power={runtime_cfg.get('sampler_power', 0.5) if use_weighted_sampler else 'n/a'}",
        logger=logger,
    )

    return DataLoader(
        dataset,
        batch_size=runtime_cfg["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        collate_fn=oracle_collate_fn,
        drop_last=(split == "train"),
    )
