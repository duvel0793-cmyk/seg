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
from datasets.transforms import SCALE_NAMES
from datasets.xbd_oracle_instance_damage import XBDOracleInstanceDamageDataset
from utils.io import append_jsonl, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect multi-context train samples and masks.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--save_dir", type=str, default="outputs/debug_multicontext_samples")
    return parser.parse_args()


def _unnormalize_image(image: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    restored = image.transpose(1, 2, 0)
    restored = (restored * np.asarray(std, dtype=np.float32)) + np.asarray(mean, dtype=np.float32)
    restored = np.clip(restored * 255.0, 0.0, 255.0).astype(np.uint8)
    return restored


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    mask_image = (mask.squeeze(0) > 0.5).astype(np.uint8) * 255
    return np.repeat(mask_image[:, :, None], 3, axis=2)


def _build_sample_panel(sample: dict, *, mean: list[float], std: list[float]) -> Image.Image:
    tiles = []
    for scale_name in SCALE_NAMES:
        pre_image = _unnormalize_image(sample[f"pre_{scale_name}"].numpy(), mean, std)
        post_image = _unnormalize_image(sample[f"post_{scale_name}"].numpy(), mean, std)
        mask_image = _mask_to_rgb(sample[f"mask_{scale_name}"].numpy())
        row = [Image.fromarray(pre_image), Image.fromarray(post_image), Image.fromarray(mask_image)]
        tiles.append(row)

    width = max(image.width for row in tiles for image in row)
    height = max(image.height for row in tiles for image in row)
    panel = Image.new("RGB", (width * 3, height * len(SCALE_NAMES)), color=(16, 18, 24))
    draw = ImageDraw.Draw(panel)
    for row_index, scale_name in enumerate(SCALE_NAMES):
        draw.text((8, (row_index * height) + 6), scale_name, fill=(255, 255, 255))
        for col_index, image in enumerate(tiles[row_index]):
            panel.paste(image.resize((width, height)), (col_index * width, row_index * height))
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
    normalize_mean = list(config["augmentation"]["normalize_mean"])
    normalize_std = list(config["augmentation"]["normalize_std"])

    count = min(int(args.count), len(dataset))
    for index in range(count):
        sample = dataset[index]
        meta = sample["meta"]
        panel = _build_sample_panel(sample, mean=normalize_mean, std=normalize_std)
        panel.save(save_dir / f"sample_{index:03d}_{meta['tile_id']}_{meta['building_idx']}.png")

        record = {
            "sample_index": int(index),
            "tile_id": meta["tile_id"],
            "building_idx": int(meta["building_idx"]),
            "crop_original_bbox_xyxy": meta["bbox_xyxy"],
            "tight_bbox_xyxy": meta["tight_bbox_xyxy"],
            "context_bbox_xyxy": meta["context_bbox_xyxy"],
            "neighborhood_bbox_xyxy": meta["neighborhood_bbox_xyxy"],
            "centroid_xy": meta["centroid_xy"],
            "mask_tight_pixel_ratio": float(sample["mask_tight"].float().mean().item()),
            "mask_context_pixel_ratio": float(sample["mask_context"].float().mean().item()),
            "mask_neighborhood_pixel_ratio": float(sample["mask_neighborhood"].float().mean().item()),
        }
        append_jsonl(summary_path, record)
        print(
            f"[{index:03d}] tile={meta['tile_id']} building={meta['building_idx']} "
            f"orig_bbox={record['crop_original_bbox_xyxy']} "
            f"tight={record['tight_bbox_xyxy']} "
            f"context={record['context_bbox_xyxy']} "
            f"neighborhood={record['neighborhood_bbox_xyxy']} "
            f"mask_ratio(t/c/n)="
            f"{record['mask_tight_pixel_ratio']:.4f}/"
            f"{record['mask_context_pixel_ratio']:.4f}/"
            f"{record['mask_neighborhood_pixel_ratio']:.4f}"
        )

    print(f"Saved {count} sample panels to {save_dir}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
