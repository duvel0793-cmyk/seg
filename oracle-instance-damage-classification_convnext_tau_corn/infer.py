"""Run checkpoint inference on a small number of dataset samples."""

from __future__ import annotations

import argparse

import torch

from datasets.build import build_dataloader
from datasets.label_mapping import LABEL_TO_DAMAGE
from models.build import build_model
from utils.checkpoint import load_checkpoint
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer oracle damage labels on a split.")
    parser.add_argument("--config", default="configs/default.yaml", help="Config path.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Split to sample from.")
    parser.add_argument("--num-batches", type=int, default=1, help="Number of batches to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() and config["runtime"].get("device", "cuda") == "cuda" else "cpu")

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()

    dataloader = build_dataloader(config, split=args.split, shuffle=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= args.num_batches:
                break
            pre_image = batch["pre_image"].to(device)
            post_image = batch["post_image"].to(device)
            mask = batch["mask"].to(device)
            outputs = model(pre_image=pre_image, post_image=post_image, mask=mask)
            preds = outputs["pred_labels"].cpu().tolist()
            for meta, pred in zip(batch["meta"], preds):
                print(
                    {
                        "sample_id": meta["sample_id"],
                        "image_id": meta["image_id"],
                        "building_id": meta["building_id"],
                        "pred_label": pred,
                        "pred_name": LABEL_TO_DAMAGE[pred],
                    }
                )


if __name__ == "__main__":
    main()

