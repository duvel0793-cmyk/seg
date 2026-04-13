"""Export split-level class count statistics."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.build import build_dataset
from datasets.label_mapping import LABEL_TO_DAMAGE
from utils.config import load_config
from utils.misc import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Export train/val/test split statistics.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    stats = {}
    for split in ("train", "val", "test"):
        dataset = build_dataset(config, split=split)
        counts = Counter(int(item["label"]) for item in dataset.records)
        stats[split] = {
            "num_samples": len(dataset),
            "class_counts": {LABEL_TO_DAMAGE[key]: counts.get(key, 0) for key in sorted(LABEL_TO_DAMAGE)},
        }

    output_path = Path(config["runtime"]["output_dir"]) / "split_stats.json"
    save_json(stats, output_path)
    print(f"Saved split stats to {output_path}")


if __name__ == "__main__":
    main()
