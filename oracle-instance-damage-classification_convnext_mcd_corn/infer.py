from __future__ import annotations

import argparse

import torch

from datasets import XBDInstanceDamageDataset
from datasets.label_mapping import CLASS_NAMES
from losses.corn_loss import decode_threshold_count
from metrics.corn_decode import decode_corn_logits_with_thresholds
from models import build_model
from utils import load_checkpoint
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    dataset = XBDInstanceDamageDataset(config, split_name=args.split, is_train=False)
    sample = dataset[args.index]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    with torch.no_grad():
        outputs = model(
            sample["pre_image"].unsqueeze(0).to(device),
            sample["post_image"].unsqueeze(0).to(device),
            sample["instance_mask"].unsqueeze(0).to(device),
        )
    logits = outputs["tau_adjusted_logits"].detach().cpu()
    raw_pred = int(decode_threshold_count(logits)[0].item())
    best_thresholds = checkpoint.get("best_thresholds")
    calibrated_pred = raw_pred
    if best_thresholds is not None:
        calibrated_pred = int(decode_corn_logits_with_thresholds(logits, best_thresholds)[0].item())

    print(f"Meta: {sample['meta']}")
    print(f"GT label: {sample['label']} ({CLASS_NAMES[sample['label']]})")
    print(f"Raw pred: {raw_pred} ({CLASS_NAMES[raw_pred]})")
    print(f"Calibrated pred: {calibrated_pred} ({CLASS_NAMES[calibrated_pred]})")
    print(f"Tau: {outputs['tau'].detach().cpu().tolist()}")
    print(f"CE logits shape: {tuple(outputs['ce_logits'].shape)}")
    print(f"CORN logits shape: {tuple(outputs['corn_logits'].shape)}")


if __name__ == "__main__":
    main()
