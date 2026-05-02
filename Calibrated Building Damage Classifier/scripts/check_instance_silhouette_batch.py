from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_CONFIG_PATH, load_config
from datasets.xbd_oracle_instance_damage import XBDOracleInstanceDamageDataset
from utils.io import append_jsonl, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect instance silhouette neighborhood tensors.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--save_dir", type=str, default="outputs/debug_instance_silhouette")
    return parser.parse_args()


def _unnormalize_image(image: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    restored = image.transpose(1, 2, 0)
    restored = (restored * np.asarray(std, dtype=np.float32)) + np.asarray(mean, dtype=np.float32)
    return np.clip(restored * 255.0, 0.0, 255.0).astype(np.uint8)


def _single_channel_to_rgb(tensor: np.ndarray, *, signed: bool = False) -> np.ndarray:
    values = tensor.squeeze(0).astype(np.float32)
    if signed:
        values = np.clip((values + 1.0) * 0.5, 0.0, 1.0)
    else:
        values = np.clip(values, 0.0, 1.0)
    image = (values * 255.0).astype(np.uint8)
    return np.repeat(image[:, :, None], 3, axis=2)


def _build_panel(sample: dict, *, mean: list[float], std: list[float]) -> Image.Image:
    tiles = [
        ("neighborhood_rgb", _unnormalize_image(sample["post_neighborhood"].numpy(), mean, std)),
        ("target_mask", _single_channel_to_rgb(sample["target_mask_neighborhood"].numpy())),
        ("neighbor_union", _single_channel_to_rgb(sample["neighbor_union_mask_neighborhood"].numpy())),
        ("target_boundary", _single_channel_to_rgb(sample["target_boundary_neighborhood"].numpy())),
        ("neighbor_boundary", _single_channel_to_rgb(sample["neighbor_boundary_neighborhood"].numpy())),
        ("distance", _single_channel_to_rgb(sample["neighbor_distance_to_target_map"].numpy())),
        ("relative_x", _single_channel_to_rgb(sample["relative_x_map"].numpy(), signed=True)),
        ("relative_y", _single_channel_to_rgb(sample["relative_y_map"].numpy(), signed=True)),
    ]
    for neighbor_index in range(sample["neighbor_masks"].shape[0]):
        tiles.append((f"neighbor_{neighbor_index}", _single_channel_to_rgb(sample["neighbor_masks"][neighbor_index].numpy()[None, ...])))

    tile_size = max(image.shape[0] for _, image in tiles)
    cols = 3
    rows = int(np.ceil(len(tiles) / cols))
    panel = Image.new("RGB", (cols * tile_size, rows * tile_size), color=(18, 20, 24))
    draw = ImageDraw.Draw(panel)
    for index, (label, array) in enumerate(tiles):
        tile = Image.fromarray(array).resize((tile_size, tile_size))
        x = (index % cols) * tile_size
        y = (index // cols) * tile_size
        panel.paste(tile, (x, y))
        draw.text((x + 8, y + 8), label, fill=(255, 255, 255))
    return panel


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    save_dir = ensure_dir(args.save_dir)
    summary_path = save_dir / "summary.jsonl"
    if summary_path.exists():
        summary_path.unlink()

    dataset = XBDOracleInstanceDamageDataset(
        config=config,
        split_name="train",
        list_path=config["dataset"]["train_list"],
        is_train=False,
    )
    mean = list(config["augmentation"]["normalize_mean"])
    std = list(config["augmentation"]["normalize_std"])

    count = min(int(args.count), len(dataset))
    for index in range(count):
        sample = dataset[index]
        meta = sample["meta"]
        panel = _build_panel(sample, mean=mean, std=std)
        panel.save(save_dir / f"sample_{index:03d}_{meta['tile_id']}_{meta['building_idx']}.png")

        valid_mask = sample["neighbor_valid_mask"].numpy().astype(bool)
        geometry = sample["neighbor_geometry"].numpy()
        record = {
            "sample_index": int(index),
            "tile_id": meta["tile_id"],
            "building_idx": int(meta["building_idx"]),
            "target_area": float(meta["polygon_area"]),
            "valid_neighbor_count": int(valid_mask.sum()),
            "neighbor_distance_list": [float(row[7]) for row, valid in zip(geometry, valid_mask) if valid],
            "neighbor_area_list": [float(row[0]) for row, valid in zip(geometry, valid_mask) if valid],
            "crop_coordinate": meta["neighborhood_bbox_xyxy"],
            "mask_ratio": float(sample["target_mask_neighborhood"].float().mean().item()),
        }
        append_jsonl(summary_path, record)
        print(
            f"[{index:03d}] tile={record['tile_id']} building={record['building_idx']} "
            f"target_area={record['target_area']:.2f} valid_neighbors={record['valid_neighbor_count']} "
            f"neighbor_distances={record['neighbor_distance_list']} "
            f"neighbor_areas={record['neighbor_area_list']} "
            f"crop={record['crop_coordinate']} mask_ratio={record['mask_ratio']:.4f}"
        )

    print(f"Saved {count} instance silhouette panels to {save_dir}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
