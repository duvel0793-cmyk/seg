import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from changedetection.datasets.make_data_loader import make_data_loader, read_data_list
from changedetection.utils_func import lovasz_loss as L
from changedetection.utils_func.build_model import build_model
from changedetection.utils_func.logger import get_logger, log_validation_metrics
from changedetection.utils_func.metrics import Evaluator
from changedetection.utils_func.param_calculate import param_calculate


def category_distance_aware_loss(logits, target, alpha=0.3, ignore_index=255, background_index=0):
    """
    Apply an ordinal soft-label loss on damage classes only.

    logits: [B, C, H, W] with class 0 reserved for background.
    target: [B, H, W] with valid damage labels in [1, C-1].
    """
    valid_mask = target != ignore_index
    if not valid_mask.any():
        return logits.new_tensor(0.0)

    valid_logits = logits.permute(0, 2, 3, 1)[valid_mask]
    valid_target = target[valid_mask]

    damage_logits = valid_logits[:, background_index + 1 :]
    damage_target = valid_target - (background_index + 1)

    if damage_logits.numel() == 0:
        return logits.new_tensor(0.0)

    class_ids = torch.arange(damage_logits.shape[1], device=logits.device)
    distance_weights = alpha ** torch.abs(class_ids.unsqueeze(0) - damage_target.unsqueeze(1))
    distance_weights = distance_weights / distance_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

    log_probs = F.log_softmax(damage_logits, dim=1)
    return -(distance_weights * log_probs).sum(dim=1).mean()

class Trainer:
    def __init__(self, args):
        self.args = args
        self.cda_alpha = getattr(args, "cda_alpha", 0.3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(args).to(self.device)

        self.train_loader = make_data_loader(
            dataset_path=args.train_dataset_path,
            data_list=read_data_list(args.train_data_list_path),
            crop_size=args.crop_size,
            batch_size=args.batch_size,
            split="train",
            max_iters=args.max_iters,
            num_workers=args.num_workers,
        )
        self.val_names = read_data_list(args.test_data_list_path)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)

        self.save_dir = Path(args.model_param_path)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(args.log_file, logging.INFO)
        self.evaluator_loc = Evaluator(num_class=2)
        self.evaluator_clf = Evaluator(num_class=5)

        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu")
            self.model.load_state_dict(checkpoint, strict=False)

        if args.with_param_calculate:
            param_calculate(self.model, args.batch_size, args.crop_size, self.logger)

    def train(self):
        best_score = -1.0
        best_metrics = None
        self.model.train()
        self.logger.info("-" * 10 + "Training configs" + "-" * 10)
        self.logger.info(
            "Training option: FlowMamba + loc(ce + 0.5*lovasz) + "
            "damage(ce + 0.75*lovasz + cda) + align "
            f"(align_loss_weight={self.args.align_loss_weight}, cda_alpha={self.cda_alpha})"
        )
        self.logger.info(f"Pretrained weight: {self.args.pretrained_weight_path or 'None'}")
        self.logger.info("-" * 10 + "Start training" + "-" * 10)

        start = time.time()
        for iteration, batch in enumerate(self.train_loader, start=1):
            pre_img, post_img, labels_loc, labels_clf, _ = batch
            pre_img = pre_img.to(self.device).float()
            post_img = post_img.to(self.device).float()
            labels_loc = labels_loc.to(self.device).long()
            labels_clf = labels_clf.to(self.device).long()

            if not (labels_clf != 255).any():
                continue

            if self.args.align_loss_weight > 0:
                output_loc, output_clf, align_loss = self.model(pre_img, post_img, labels_loc)
            else:
                output_loc, output_clf = self.model(pre_img, post_img)
                align_loss = torch.zeros((), device=self.device)

            loc_loss = F.cross_entropy(output_loc, labels_loc, ignore_index=255)
            loc_loss = loc_loss + 0.5 * L.lovasz_softmax(F.softmax(output_loc, dim=1), labels_loc, ignore=255)

            clf_ce = F.cross_entropy(output_clf, labels_clf, ignore_index=255)
            clf_lovasz = 0.75 * L.lovasz_softmax(F.softmax(output_clf, dim=1), labels_clf, ignore=255)
            clf_cda = category_distance_aware_loss(
                output_clf,
                labels_clf,
                alpha=self.cda_alpha,
                ignore_index=255,
                )

            clf_loss = clf_ce + clf_lovasz + clf_cda

            loss = loc_loss + clf_loss + self.args.align_loss_weight * align_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if iteration % self.args.log_interval == 0:
                elapsed = max(time.time() - start, 1e-6)
                eta_seconds = (len(self.train_loader) - iteration) * elapsed / self.args.log_interval
                eta = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                self.logger.info(
                    f"Training | iter: {iteration}/{len(self.train_loader)} | "
                    f"loss: {loss.item():.4f} | loc_loss: {loc_loss.item():.4f} | "
                    f"bda_loss: {clf_loss.item():.4f} | ce: {clf_ce.item():.4f} | "
                    f"lovasz: {clf_lovasz.item():.4f} | cda: {clf_cda.item():.4f} | "
                    f"align_loss: {align_loss.item():.4f} | "
                    f"eta: {eta}"
                )
                start = time.time()

            if iteration % self.args.val_interval == 0:
                metrics = self.validate()
                if metrics["F1_oa"] > best_score:
                    best_score = metrics["F1_oa"]
                    best_metrics = metrics
                    torch.save(self.model.state_dict(), self.save_dir / "best_model.pth")
                self.model.train()

        torch.save(self.model.state_dict(), self.save_dir / "last_model.pth")
        if best_metrics is not None:
            log_validation_metrics(self.logger, best_metrics, final=True)
        else:
            self.logger.info("No validation metrics were produced during this run.")

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        self.evaluator_loc.reset()
        self.evaluator_clf.reset()

        val_loader = make_data_loader(
            dataset_path=self.args.test_dataset_path,
            data_list=self.val_names,
            crop_size=1024,
            batch_size=1,
            split="test",
            num_workers=self.args.num_workers,
            shuffle=False,
        )

        for batch in val_loader:
            pre_img, post_img, labels_loc, labels_clf, _ = batch
            pre_img = pre_img.to(self.device).float()
            post_img = post_img.to(self.device).float()
            labels_loc = labels_loc.to(self.device).long()
            labels_clf = labels_clf.to(self.device).long()

            output_loc, output_clf = self.model(pre_img, post_img)
            output_loc = output_loc.argmax(dim=1).cpu().numpy()
            output_clf = output_clf.argmax(dim=1).cpu().numpy()
            labels_loc = labels_loc.cpu().numpy()
            labels_clf = labels_clf.cpu().numpy()

            self.evaluator_loc.add_batch(labels_loc, output_loc)
            valid_pred = output_clf[labels_loc > 0]
            valid_label = labels_clf[labels_loc > 0]
            if valid_pred.size > 0:
                self.evaluator_clf.add_batch(valid_label, valid_pred)

        damage_f1 = self.evaluator_clf.Damage_F1_score()
        harmonic_mean_f1 = len(damage_f1) / np.sum(1.0 / (damage_f1 + 1e-7))
        metrics = dict(
            F1_oa=0.3 * self.evaluator_loc.Pixel_F1_score() + 0.7 * harmonic_mean_f1,
            F1_loc=self.evaluator_loc.Pixel_F1_score(),
            F1_bda=harmonic_mean_f1,
            F1_subcls=damage_f1,
        )
        log_validation_metrics(self.logger, metrics)
        return metrics


def parse_args():
    default_data_root = Path(os.environ.get("DATA_ROOT", "/home/lky/data/xBD"))
    default_save_dir = PROJECT_ROOT / "saved_models" / "flowmamba"
    default_pretrained = PROJECT_ROOT / "pretrained_weight" / "vssm_small_0229_ckpt_epoch_222.pth"
    parser = argparse.ArgumentParser(description="Train FlowMamba on xBD")
    parser.add_argument("--cfg", type=str, default=str(PROJECT_ROOT / "changedetection/configs/vssm1/vssm_small_224.yaml"))
    parser.add_argument("--opts", default=None, nargs="+")
    parser.add_argument(
        "--pretrained_weight_path",
        type=str,
        default=str(default_pretrained) if default_pretrained.exists() else None,
    )
    parser.add_argument("--train_dataset_path", type=str, default=str(default_data_root / "train"))
    parser.add_argument("--train_data_list_path", type=str, default=str(default_data_root / "xBD_list/train_all.txt"))
    parser.add_argument("--test_dataset_path", type=str, default=str(default_data_root / "test"))
    parser.add_argument("--test_data_list_path", type=str, default=str(default_data_root / "xBD_list/val_all.txt"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--max_iters", type=int, default=80000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--lr_decay_step", type=int, default=20000)
    parser.add_argument("--lr_decay_gamma", type=float, default=0.5)
    parser.add_argument("--val_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--model_param_path", type=str, default=str(default_save_dir))
    parser.add_argument("--log_file", type=str, default=str(default_save_dir / "train.log"))
    parser.add_argument("--resume", type=str)
    parser.add_argument("--align_loss_weight", type=float, default=1.0)
    parser.add_argument("--with_param_calculate", action="store_true", default=False)
    parser.add_argument("--cda_alpha", type=float, default=0.3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
    Trainer(args).train()
