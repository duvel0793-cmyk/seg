import heapq
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from building_segmentation.utils.common import AverageMeter, ensure_dir
from building_segmentation.utils.losses import lovasz_softmax
from building_segmentation.utils.metrics import SegmentationMetrics
from building_segmentation.utils.visualization import save_prediction_visualization


def compute_building_iou(gt_mask, pred_mask, ignore_index=255):
    valid = gt_mask != ignore_index
    gt_building = gt_mask[valid] == 1
    pred_building = pred_mask[valid] == 1
    union = np.logical_or(gt_building, pred_building).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(gt_building, pred_building).sum()
    return float(intersection / union)


def save_top_visualizations(candidates, save_dir, clear_existing=False):
    save_dir = Path(save_dir)
    if clear_existing and save_dir.exists():
        shutil.rmtree(save_dir)
    vis_dir = ensure_dir(save_dir)

    ranked_candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)
    for rank, candidate in enumerate(ranked_candidates, start=1):
        score = candidate["score"]
        name = candidate["name"]
        save_prediction_visualization(
            output_path=vis_dir / f"{rank:02d}_{name}_iou_{score:.4f}.png",
            image=candidate["image"],
            pred_mask=candidate["pred_mask"],
            gt_mask=candidate["gt_mask"],
            title=f"{name} | building IoU {score:.4f}",
        )


@torch.no_grad()
def evaluate_model(model, data_loader, device, save_dir=None, num_visualizations=0, collect_visualizations=False):
    model.eval()
    metrics = SegmentationMetrics(num_classes=2)
    loss_meter = AverageMeter()

    collect_candidates = num_visualizations > 0 and (save_dir is not None or collect_visualizations)
    top_visualizations = []
    sample_order = 0

    for images, masks, names in data_loader:
        images = images.to(device=device, dtype=torch.float32)
        masks = masks.to(device=device, dtype=torch.long)

        logits = model(images)
        probs = F.softmax(logits, dim=1)
        loss = F.cross_entropy(logits, masks, ignore_index=255)
        loss = loss + 0.5 * lovasz_softmax(probs, masks, ignore=255)
        loss_meter.update(loss.item(), images.size(0))

        preds = logits.argmax(dim=1).cpu().numpy()
        gt = masks.cpu().numpy()

        for index in range(preds.shape[0]):
            metrics.update(gt[index], preds[index])
            if collect_candidates:
                score = compute_building_iou(gt[index], preds[index])
                candidate = {
                    "score": score,
                    "name": names[index],
                    "image": images[index].cpu().numpy().copy(),
                    "pred_mask": preds[index].astype(np.uint8, copy=True),
                    "gt_mask": gt[index].astype(np.uint8, copy=True),
                }
                item = (score, sample_order, candidate)
                sample_order += 1
                if len(top_visualizations) < num_visualizations:
                    heapq.heappush(top_visualizations, item)
                elif score > top_visualizations[0][0]:
                    heapq.heapreplace(top_visualizations, item)

    results = metrics.summary()
    results["loss"] = round(loss_meter.avg, 6)
    candidates = [item[2] for item in sorted(top_visualizations, key=lambda item: item[0], reverse=True)]

    if save_dir is not None and candidates:
        save_top_visualizations(candidates, save_dir=save_dir, clear_existing=True)

    if collect_visualizations:
        return results, candidates
    return results
