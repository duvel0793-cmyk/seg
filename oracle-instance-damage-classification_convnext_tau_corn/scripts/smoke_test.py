"""Smoke test covering dataset -> model forward -> tau/CORN outputs."""

from __future__ import annotations

import argparse
from pprint import pprint
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.build import build_dataloader
from models.build import build_model
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight dataset/model smoke test.")
    parser.add_argument("--config", default="configs/default.yaml", help="Config path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    loader = build_dataloader(config, split="val", shuffle=False)
    batch = next(iter(loader))

    model = build_model(config)
    model.eval()

    with torch.no_grad():
        outputs = model(
            pre_image=batch["pre_image"],
            post_image=batch["post_image"],
            mask=batch["mask"],
        )

    print("pre_image:", tuple(batch["pre_image"].shape))
    print("post_image:", tuple(batch["post_image"].shape))
    print("mask:", tuple(batch["mask"].shape))
    print("label:", tuple(batch["label"].shape))
    print("base_logits:", tuple(outputs["base_logits"].shape))
    print("logits:", tuple(outputs["logits"].shape))
    print("features_pre:", tuple(outputs["features_pre"].shape))
    print("features_post:", tuple(outputs["features_post"].shape))
    print("fused_features:", tuple(outputs["fused_features"].shape))
    print("decode_mode:", config["model"].get("decode_mode"))
    print("tau:", outputs["tau"])
    print("tau_raw:", outputs["tau_raw"])
    print("tau_positive:", outputs["tau_positive"])
    print("pred_labels:", outputs["pred_labels"])
    print("meta_sample:")
    pprint(batch["meta"][0])


if __name__ == "__main__":
    main()
