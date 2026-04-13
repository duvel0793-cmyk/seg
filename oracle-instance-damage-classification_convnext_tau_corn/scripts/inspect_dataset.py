"""Inspect dataset size, class balance, and a few sample tensors."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.build import build_dataset
from datasets.label_mapping import LABEL_TO_DAMAGE
from utils.config import load_config
from utils.misc import ensure_dir


def tensor_to_uint8_image(tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    image = tensor.permute(1, 2, 0).numpy()
    image = (image * std + mean).clip(0.0, 1.0)
    return (image * 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect parsed oracle dataset.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--save-visuals", action="store_true")
    parser.add_argument("--num-samples", type=int, default=3)
    args = parser.parse_args()

    config = load_config(args.config)
    train_dataset = build_dataset(config, split="train")
    val_dataset = build_dataset(config, split="val")

    print(f"train samples: {len(train_dataset)}")
    print(f"val samples: {len(val_dataset)}")

    train_distribution = Counter(int(item["label"]) for item in train_dataset.records)
    val_distribution = Counter(int(item["label"]) for item in val_dataset.records)
    train_post_polygon_fallback = sum(1 for item in train_dataset.records if item.get("polygon_source") == "post_fallback")
    val_post_polygon_fallback = sum(1 for item in val_dataset.records if item.get("polygon_source") == "post_fallback")
    print("train class distribution:", {LABEL_TO_DAMAGE[k]: v for k, v in sorted(train_distribution.items())})
    print("val class distribution:", {LABEL_TO_DAMAGE[k]: v for k, v in sorted(val_distribution.items())})
    print(
        "train parser stats:",
        {
            **train_dataset.parser_stats,
            "num_post_polygon_fallback_kept": train_post_polygon_fallback,
        },
    )
    print(
        "val parser stats:",
        {
            **val_dataset.parser_stats,
            "num_post_polygon_fallback_kept": val_post_polygon_fallback,
        },
    )

    output_dir = Path(config["runtime"]["output_dir"]) / "debug_samples"
    if args.save_visuals:
        ensure_dir(output_dir)

    inspect_count = min(args.num_samples, len(val_dataset))
    for idx in range(inspect_count):
        sample = val_dataset[idx]
        print(f"sample[{idx}] pre={tuple(sample['pre_image'].shape)} post={tuple(sample['post_image'].shape)} mask={tuple(sample['mask'].shape)} label={sample['label'].item()}")
        print(sample["meta"])

        if args.save_visuals:
            pre_image = tensor_to_uint8_image(sample["pre_image"])
            post_image = tensor_to_uint8_image(sample["post_image"])
            mask = (sample["mask"].squeeze(0).numpy() * 255).astype(np.uint8)
            canvas = np.concatenate(
                [
                    pre_image,
                    post_image,
                    np.repeat(mask[:, :, None], 3, axis=2),
                ],
                axis=1,
            )
            Image.fromarray(canvas).save(output_dir / f"sample_{idx:03d}.png")


if __name__ == "__main__":
    main()
