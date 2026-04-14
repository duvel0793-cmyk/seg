from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import CLASS_NAMES, XBDInstanceDamageDataset
from datasets.xbd_json import polygon_to_mask
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split", type=str, default="")
    parser.add_argument("--num-vis", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="")
    return parser.parse_args()


def save_visualizations(dataset: XBDInstanceDamageDataset, save_dir: Path, num_vis: int) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    for idx, sample in enumerate(dataset.samples[:num_vis]):
        pre_image = Image.open(sample["pre_image"]).convert("RGB")
        post_image = Image.open(sample["post_image"]).convert("RGB")
        x1, y1, x2, y2 = sample["crop_bbox_xyxy"]
        pre_crop = pre_image.crop((x1, y1, x2, y2))
        post_crop = post_image.crop((x1, y1, x2, y2))
        mask_np = polygon_to_mask(sample["polygon_xy"], y2 - y1, x2 - x1, offset=(x1, y1))
        mask_rgb = np.repeat((mask_np[..., None] * 255), 3, axis=2)
        panel = np.concatenate([np.asarray(pre_crop), np.asarray(post_crop), mask_rgb], axis=1)
        output_path = save_dir / f"{idx:03d}_{sample['tile_id']}_{sample['original_subtype']}.png"
        Image.fromarray(panel.astype(np.uint8)).save(output_path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    split_name = args.split or str(config["data"]["train_split"])
    dataset = XBDInstanceDamageDataset(config, split_name=split_name, is_train=False)

    print(f"split: {split_name}")
    print(f"num_samples: {len(dataset)}")
    print(f"class_counts: {dict(zip(CLASS_NAMES, dataset.class_counts))}")

    if args.save_dir:
        save_visualizations(dataset, Path(args.save_dir), args.num_vis)
        print(f"saved_visualizations: {Path(args.save_dir).resolve()}")


if __name__ == "__main__":
    main()
