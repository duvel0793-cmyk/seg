from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import CLASS_NAMES, XBDInstanceDamageDataset, xbd_instance_collate_fn
from engine.trainer import build_ce_class_weights, build_optimizer, build_weighted_sampler, fit
from losses import build_loss
from models import build_model
from utils import save_json, seed_everything, setup_logger
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["train"].get("seed", 42)))

    output_root = Path(args.output_dir) if args.output_dir else Path(config["output"]["root"]) / str(config["output"]["exp_name"])
    logger = setup_logger(output_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    train_dataset = XBDInstanceDamageDataset(config, split_name=str(config["data"]["train_split"]), is_train=True)
    val_dataset = XBDInstanceDamageDataset(config, split_name=str(config["data"]["val_split"]), is_train=False)
    sampler, sampler_weights = build_weighted_sampler(train_dataset, config)
    ce_class_weights = build_ce_class_weights(train_dataset)

    logger.info("Train dataset size: %d", len(train_dataset))
    logger.info("Val dataset size: %d", len(val_dataset))
    logger.info("Train class counts: %s", {name: count for name, count in zip(CLASS_NAMES, train_dataset.class_counts)})
    logger.info("Val class counts: %s", {name: count for name, count in zip(CLASS_NAMES, val_dataset.class_counts)})
    logger.info("Sampler class weights: %s", sampler_weights)
    logger.info("CE class weights: %s", {name: round(float(ce_class_weights[idx].item()), 4) for idx, name in enumerate(CLASS_NAMES)})

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        collate_fn=xbd_instance_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        collate_fn=xbd_instance_collate_fn,
    )

    model = build_model(config).to(device)
    criterion = build_loss(config, ce_class_weights=ce_class_weights.to(device))
    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["train"]["epochs"]))

    pretrained_report = model.get_pretrained_report()
    logger.info("Backbone loaded report: %s", pretrained_report)
    logger.info(
        "Loss components: ce_weight=%.3f corn_weight=%.3f tau_reg_weight=%.3f freeze_tau_epochs=%d",
        float(config["model"]["ce_weight"]),
        float(config["model"]["corn_weight"]),
        float(config["model"]["tau_reg_weight"]),
        int(config["model"]["freeze_tau_epochs"]),
    )

    result = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        output_dir=output_root,
        logger=logger,
    )
    save_json(result, output_root / "train_summary.json")
    logger.info("Training finished. Summary: %s", result)


if __name__ == "__main__":
    main()

