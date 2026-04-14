from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import XBDInstanceDamageDataset, xbd_instance_collate_fn
from engine.trainer import build_ce_class_weights
from losses import build_loss
from models import build_model
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = XBDInstanceDamageDataset(config, split_name=str(config["data"]["train_split"]), is_train=True)
    loader = DataLoader(
        dataset,
        batch_size=min(2, int(config["train"]["batch_size"])),
        shuffle=False,
        num_workers=0,
        collate_fn=xbd_instance_collate_fn,
    )
    batch = next(iter(loader))
    model = build_model(config).to(device)
    ce_class_weights = build_ce_class_weights(dataset).to(device)
    criterion = build_loss(config, ce_class_weights=ce_class_weights)

    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    outputs = model(batch["pre_image"], batch["post_image"], batch["instance_mask"])
    loss_dict = criterion(outputs, batch["label"])

    print(f"ce_logits shape: {tuple(outputs['ce_logits'].shape)}")
    print(f"corn_logits shape: {tuple(outputs['corn_logits'].shape)}")
    print(f"tau: {outputs['tau'].detach().cpu().tolist()}")
    print(f"pred_labels: {outputs['pred_labels'].detach().cpu().tolist()}")
    print(
        "loss components:",
        {
            key: round(float(value.detach().cpu().item()), 6)
            for key, value in loss_dict.items()
            if torch.is_tensor(value) and value.ndim == 0
        },
    )


if __name__ == "__main__":
    main()
