import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

PACKAGE_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = PACKAGE_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from building_segmentation.datasets.xbd import make_dataloader, read_data_list
from building_segmentation.models.builder import build_model
from building_segmentation.utils.common import AverageMeter, ensure_dir, load_checkpoint, save_checkpoint, save_json, set_seed
from building_segmentation.utils.eval import evaluate_model, save_top_visualizations
from building_segmentation.utils.losses import lovasz_softmax


def parse_args():
    default_data_root = Path(os.environ.get("DATA_ROOT", "/home/lky/data/xBD"))
    default_output_dir = PACKAGE_ROOT / "outputs" / "train_run"
    default_pretrained = PACKAGE_ROOT / "pretrained_weight" / "vssm_small_0229_ckpt_epoch_222.pth"
    parser = argparse.ArgumentParser(description="Train FlowMamba building segmentation on xBD")
    parser.add_argument("--cfg", type=str, default=str(PACKAGE_ROOT / "configs/vssm_small_224.yaml"))
    parser.add_argument("--pretrained_weight_path", type=str, default=str(default_pretrained))
    parser.add_argument("--train_dataset_path", type=str, default=str(default_data_root / "train"))
    parser.add_argument("--train_data_list_path", type=str, default=str(default_data_root / "xBD_list/train_all.txt"))
    parser.add_argument("--val_dataset_path", type=str, default=str(default_data_root / "test"))
    parser.add_argument("--val_data_list_path", type=str, default=str(default_data_root / "xBD_list/val_all.txt"))
    parser.add_argument("--save_dir", type=str, default=str(default_output_dir))
    parser.add_argument("--resume", type=str)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--lr_decay_step", type=int, default=20)
    parser.add_argument("--lr_decay_gamma", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--vis_samples", type=int, default=4)
    return parser.parse_args()


def train_one_epoch(model, data_loader, optimizer, device, log_interval):
    model.train()
    loss_meter = AverageMeter()

    for step, (images, masks, _) in enumerate(data_loader, start=1):
        images = images.to(device=device, dtype=torch.float32)
        masks = masks.to(device=device, dtype=torch.long)

        logits = model(images)
        probs = F.softmax(logits, dim=1)
        loss = F.cross_entropy(logits, masks, ignore_index=255)
        loss = loss + 0.5 * lovasz_softmax(probs, masks, ignore=255)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))
        if step % log_interval == 0 or step == len(data_loader):
            print(f"train step {step}/{len(data_loader)} | loss {loss_meter.avg:.4f}")

    return loss_meter.avg


def main():
    args = parse_args()
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "FlowMamba building segmentation requires CUDA. "
            "The current vmamba selective-scan kernels do not support CPU inference/training."
        )
    device = torch.device("cuda")
    save_dir = ensure_dir(args.save_dir)
    vis_dir = ensure_dir(save_dir / "visualizations")

    train_loader = make_dataloader(
        dataset_path=args.train_dataset_path,
        data_list=read_data_list(args.train_data_list_path),
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        split="train",
        num_workers=args.num_workers,
    )
    val_loader = make_dataloader(
        dataset_path=args.val_dataset_path,
        data_list=read_data_list(args.val_data_list_path),
        crop_size=1024,
        batch_size=1,
        split="val",
        num_workers=args.num_workers,
        shuffle=False,
    )

    model = build_model(args.cfg, args.pretrained_weight_path).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)

    start_epoch = 0
    best_miou = -1.0
    if args.resume:
        start_epoch, checkpoint = load_checkpoint(model, args.resume, optimizer=optimizer, map_location="cpu")
        metrics = checkpoint.get("metrics", {}) if isinstance(checkpoint, dict) else {}
        best_miou = float(metrics.get("miou", -1.0))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    save_json(vars(args), save_dir / "train_args.json")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        print(f"epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.log_interval)
        if args.vis_samples > 0:
            val_metrics, top_visualizations = evaluate_model(
                model=model,
                data_loader=val_loader,
                device=device,
                num_visualizations=args.vis_samples,
                collect_visualizations=True,
            )
        else:
            val_metrics = evaluate_model(
                model=model,
                data_loader=val_loader,
                device=device,
                num_visualizations=0,
            )
            top_visualizations = []
        val_metrics["train_loss"] = round(train_loss, 6)
        print(
            "val | "
            f"miou {val_metrics['miou']:.4f} | "
            f"iou_building {val_metrics['iou_building']:.4f} | "
            f"f1 {val_metrics['f1']:.4f} | "
            f"pixel_acc {val_metrics['pixel_accuracy']:.4f}"
        )

        save_checkpoint(save_dir / "last_model.pth", epoch, model, optimizer, val_metrics)
        save_json(val_metrics, save_dir / f"metrics_epoch_{epoch:03d}.json")

        if val_metrics["miou"] > best_miou:
            best_miou = float(val_metrics["miou"])
            save_checkpoint(save_dir / "best_model.pth", epoch, model, optimizer, val_metrics)
            save_json(val_metrics, save_dir / "best_metrics.json")
            if top_visualizations:
                save_top_visualizations(top_visualizations, save_dir=vis_dir, clear_existing=True)

        scheduler.step()


if __name__ == "__main__":
    main()
