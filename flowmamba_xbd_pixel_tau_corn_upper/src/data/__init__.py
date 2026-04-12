"""Data builders."""

from __future__ import annotations

from torch.utils.data import DataLoader

from .collate import xbd_collate_fn
from .xbd_pixel_dataset import XBDPixelDataset


def build_dataset(cfg, mode: str):
    data_cfg = cfg["data"]
    return XBDPixelDataset(
        data_root=data_cfg["data_root"],
        mode=mode,
        list_path=data_cfg["train_list"] if mode == "train" else data_cfg["val_list"],
        preferred_split=data_cfg["train_split"] if mode == "train" else data_cfg["val_split"],
        crop_size=tuple(data_cfg["crop_size"]),
        image_size=tuple(data_cfg["image_size"]),
        ignore_index=int(data_cfg["ignore_index"]),
        mean=tuple(data_cfg["mean"]),
        std=tuple(data_cfg["std"]),
        augment_cfg=data_cfg.get("augment", {}),
        validation_cfg=cfg.get("validation", {}),
        min_polygon_area=float(data_cfg.get("min_polygon_area", 4.0)),
        strict=bool(cfg.get("strict_data_check", False)),
    )


def build_dataloader(cfg, mode: str):
    dataset = build_dataset(cfg, mode=mode)
    train_cfg = cfg["train"]
    batch_size = int(train_cfg["batch_size"] if mode == "train" else train_cfg["val_batch_size"])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=True,
        collate_fn=xbd_collate_fn,
        drop_last=(mode == "train"),
    )
    return dataset, loader


__all__ = ["XBDPixelDataset", "build_dataset", "build_dataloader"]
