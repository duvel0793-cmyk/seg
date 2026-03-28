import argparse
import logging
import os
import time
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from changedetection.datasets.make_data_loader import make_data_loader, read_data_list
from changedetection.utils_func.build_model import build_model
from changedetection.utils_func.logger import get_logger, log_test_metrics
from changedetection.utils_func.metrics import Evaluator


COLOR_MAP = {
    0: [0, 0, 0],
    1: [0, 255, 0],
    2: [255, 128, 0],
    3: [153, 51, 255],
    4: [255, 0, 0],
}


class Inference:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(args).to(self.device)
        checkpoint = torch.load(args.resume, map_location="cpu")
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.eval()

        self.logger = get_logger(args.log_file, logging.INFO, test=True)
        self.evaluator_loc = Evaluator(num_class=2)
        self.evaluator_clf = Evaluator(num_class=5)
        self.test_names = read_data_list(args.test_data_list_path)
        self.output_dir = Path(args.result_saved_path)
        if args.save_predictions:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def run(self):
        start = time.time()
        data_loader = make_data_loader(
            dataset_path=self.args.test_dataset_path,
            data_list=self.test_names,
            crop_size=1024,
            batch_size=1,
            split="test",
            num_workers=self.args.num_workers,
            shuffle=False,
        )

        self.evaluator_loc.reset()
        self.evaluator_clf.reset()
        infer_time = 0.0

        for pre_img, post_img, labels_loc, labels_clf, names in data_loader:
            pre_img = pre_img.to(self.device).float()
            post_img = post_img.to(self.device).float()
            labels_loc = labels_loc.to(self.device).long()
            labels_clf = labels_clf.to(self.device).long()

            tick = time.time()
            output_loc, output_clf = self.model(pre_img, post_img)
            infer_time += time.time() - tick

            pred_loc = output_loc.argmax(dim=1).cpu().numpy()
            pred_clf = output_clf.argmax(dim=1).cpu().numpy()
            gt_loc = labels_loc.cpu().numpy()
            gt_clf = labels_clf.cpu().numpy()

            self.evaluator_loc.add_batch(gt_loc, pred_loc)
            valid_pred = pred_clf[gt_loc > 0]
            valid_gt = gt_clf[gt_loc > 0]
            if valid_pred.size > 0:
                self.evaluator_clf.add_batch(valid_gt, valid_pred)

            if self.args.save_predictions:
                self.save_predictions(names[0], pred_loc.squeeze(), pred_clf.squeeze())

        damage_f1 = self.evaluator_clf.Damage_F1_score()
        harmonic_mean_f1 = len(damage_f1) / np.sum(1.0 / (damage_f1 + 1e-7))
        metrics = dict(
            acc=self.evaluator_clf.Pixel_Accuracy(),
            uoc=self.evaluator_clf.uoc_index(),
            F1_oa=0.3 * self.evaluator_loc.Pixel_F1_score() + 0.7 * harmonic_mean_f1,
            F1_loc=self.evaluator_loc.Pixel_F1_score(),
            F1_bda=harmonic_mean_f1,
            F1_subcls=damage_f1,
        )
        log_test_metrics(self.logger, metrics)
        self.logger.info(f"Inference time per image: {infer_time / max(len(data_loader), 1):.4f}s")
        self.logger.info(f"Test time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")

    def save_predictions(self, name, pred_loc, pred_clf):
        loc_map = pred_loc.copy()
        loc_map[loc_map == 1] = 255

        color_map = np.zeros(pred_clf.shape + (3,), dtype=np.uint8)
        pred_clf = pred_clf.copy()
        pred_clf[loc_map == 0] = 0
        for label, color in COLOR_MAP.items():
            color_map[pred_clf == label] = color
        color_map[(loc_map == 255) & (pred_clf == 0)] = [255, 255, 255]

        imageio.imwrite(self.output_dir / f"{name}_pre_disaster.png", loc_map.astype(np.uint8))
        imageio.imwrite(self.output_dir / f"{name}_post_disaster.png", color_map)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FlowMamba on xBD")
    parser.add_argument("--cfg", type=str, default="changedetection/configs/vssm1/vssm_small_224.yaml")
    parser.add_argument("--opts", default=None, nargs="+")
    parser.add_argument("--pretrained_weight_path", type=str, default="pretrained_weight/vssm_small_0229_ckpt_epoch_222.pth")
    parser.add_argument("--test_dataset_path", type=str, required=True)
    parser.add_argument("--test_data_list_path", type=str, required=True)
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--result_saved_path", type=str, default="saved_models/flowmamba/test_results")
    parser.add_argument("--log_file", type=str, default="test.log")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_predictions", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
    Inference(args).run()
