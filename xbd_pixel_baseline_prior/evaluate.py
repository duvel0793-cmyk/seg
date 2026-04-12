from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from src.datasets.xbd_dataset import XBDDataset, format_data_check
from src.engine.evaluator import evaluate
from src.losses.damage_losses import WeightedFocalCrossEntropyLoss
from src.losses.segmentation_losses import BCEWithDiceLoss
from src.models.resnet18_fpn_bda import build_resnet18_fpn_bda
from src.utils.io import (
    ensure_dir,
    load_checkpoint,
    load_yaml_config,
    pretty_yaml,
    resolve_device,
    save_json,
    strip_meta,
)
from src.utils.logger import setup_logger
from src.utils.metrics import CLASS_NAMES, format_confusion_matrix
from src.utils.seed import seed_worker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the lightweight xBD pixel baseline.")
    parser.add_argument("--config", type=str, default="configs/baseline_no_prior.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--list-path", type=str, default=None, help="Optional custom split file.")
    parser.add_argument("--image-dir", type=str, default=None, help="Optional custom image dir.")
    parser.add_argument("--target-dir", type=str, default=None, help="Optional custom target dir/root.")
    parser.add_argument("--prior-dir", type=str, default=None, help="Optional prior dir override.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-pred-masks", action="store_true")
    parser.add_argument("--save-overlay", action="store_true")
    return parser.parse_args()


def build_dataloader(dataset, batch_size: int, num_workers: int, pin_memory: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
    )


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    if (args.list_path is None) ^ (args.image_dir is None):
        raise ValueError("Please provide both --list-path and --image-dir together for custom evaluation.")

    if args.prior_dir is not None:
        config["data"]["prior_dir"] = args.prior_dir
    if args.target_dir is not None:
        config["data"]["target_dir"] = args.target_dir

    requested_device = str(config["device"])
    device = resolve_device(requested_device)

    default_output_dir = Path(args.checkpoint).expanduser().resolve().parent / f"eval_{args.split}"
    output_dir = ensure_dir(args.output_dir or default_output_dir)
    logger = setup_logger(log_dir=output_dir)
    logger.info("Starting evaluation")
    logger.info("Resolved config:\n%s", pretty_yaml(strip_meta(config)))

    if args.list_path is not None and args.image_dir is not None:
        list_path = args.list_path
        image_dir = args.image_dir
    elif args.split == "train":
        list_path = config["data"]["train_list"]
        image_dir = config["data"]["train_image_dir"]
    else:
        list_path = config["data"]["val_list"]
        image_dir = config["data"]["val_image_dir"]

    dataset = XBDDataset(
        list_path=list_path,
        image_dir=image_dir,
        target_dir=config["data"]["target_dir"],
        prior_dir=config["data"]["prior_dir"],
        prior_mode=config["data"]["prior_mode"],
        crop_size=None,
        is_train=False,
        strict_data_check=bool(config["data"]["strict_data_check"]),
        prior_filename_pattern=config["data"].get("prior_filename_pattern", "{sample_id}.png"),
    )
    logger.info(
        "Evaluation data check:\n%s",
        format_data_check(dataset.self_check(config["data"].get("label_stats_files", 128))),
    )

    dataloader = build_dataloader(
        dataset,
        batch_size=int(config["eval"].get("batch_size", 1)),
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=bool(config["data"]["pin_memory"]),
    )

    model = build_resnet18_fpn_bda(config["model"]).to(device)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    damage_criterion = WeightedFocalCrossEntropyLoss(
        class_weights=config["loss"]["class_weights"],
        gamma=float(config["loss"]["focal_gamma"]),
    )
    localization_criterion = BCEWithDiceLoss()

    metrics = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        damage_criterion=damage_criterion,
        localization_criterion=localization_criterion,
        lambda_loc=float(config["loss"]["lambda_loc"]),
        prior_mode=config["data"]["prior_mode"],
        prior_alpha=float(config["loss"]["prior_alpha"]),
        ignore_background_in_macro=bool(config["eval"]["ignore_background_in_macro"]),
        amp=bool(config["train"]["amp"]),
        save_pred_masks=bool(args.save_pred_masks or config["eval"].get("save_pred_masks", False)),
        save_overlay_images=bool(args.save_overlay or config["eval"].get("save_overlay", False)),
        output_dir=output_dir,
        desc=f"Evaluate-{args.split}",
    )

    logger.info("Confusion matrix:\n%s", format_confusion_matrix(metrics["confusion_matrix"]))
    for class_name in CLASS_NAMES:
        class_metrics = metrics["per_class"][class_name]
        logger.info(
            "%s | precision %.4f | recall %.4f | F1 %.4f | IoU %.4f | support %d",
            class_name,
            class_metrics["precision"],
            class_metrics["recall"],
            class_metrics["f1"],
            class_metrics["iou"],
            class_metrics["support"],
        )

    logger.info("No Damage F1: %.4f", metrics["no_damage_f1"])
    logger.info("Minor Damage F1: %.4f", metrics["minor_damage_f1"])
    logger.info("Major Damage F1: %.4f", metrics["major_damage_f1"])
    logger.info("Destroyed F1: %.4f", metrics["destroyed_f1"])
    logger.info("Damage Macro F1: %.4f", metrics["damage_macro_f1"])
    logger.info("Building localization F1: %.4f", metrics["building_localization_f1"])

    save_json(metrics, output_dir / "evaluation_metrics.json")


if __name__ == "__main__":
    main()
