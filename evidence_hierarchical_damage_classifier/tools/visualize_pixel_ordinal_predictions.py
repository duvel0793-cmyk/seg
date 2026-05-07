from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import XBDInstanceDataset, xbd_instance_collate_fn
from models import build_model
from utils.checkpoint import load_checkpoint
from utils.misc import CLASS_NAMES, ensure_dir, get_split_list, load_config, merge_config, resolve_device, set_seed


CLASS_COLORS = np.asarray(
    [
        [78, 121, 167],
        [242, 142, 43],
        [225, 87, 89],
        [118, 183, 178],
    ],
    dtype=np.float32,
) / 255.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    return parser.parse_args()


def _resolve_state_dict(checkpoint: dict[str, Any], *, use_ema: bool) -> dict[str, Any]:
    if use_ema:
        state_dict = checkpoint.get("ema_state_dict") or checkpoint.get("model_state_dict") or checkpoint.get("raw_model_state_dict")
    else:
        state_dict = checkpoint.get("raw_model_state_dict") or checkpoint.get("model_state_dict") or checkpoint.get("ema_state_dict")
    if state_dict is None:
        raise RuntimeError("Checkpoint does not contain a usable model state dict.")
    return state_dict


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def _unnormalize(image: torch.Tensor, config: dict[str, Any]) -> np.ndarray:
    mean = torch.as_tensor(config["augmentation"]["normalize_mean"], dtype=image.dtype, device=image.device).view(3, 1, 1)
    std = torch.as_tensor(config["augmentation"]["normalize_std"], dtype=image.dtype, device=image.device).view(3, 1, 1)
    restored = (image[:3] * std) + mean
    restored = restored.clamp(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
    return restored


def _render_argmax_map(pred_labels: np.ndarray) -> np.ndarray:
    return CLASS_COLORS[np.clip(pred_labels.astype(np.int64), 0, len(CLASS_COLORS) - 1)]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    if isinstance(checkpoint.get("config"), dict):
        config = merge_config(checkpoint["config"])
    set_seed(int(config["training"]["seed"]))
    save_dir = ensure_dir(
        Path(args.save_dir)
        if args.save_dir
        else Path(config["project"]["output_dir"]) / "visualizations"
    )
    dataset = XBDInstanceDataset(config=config, split=args.split, list_path=get_split_list(config, args.split), is_train=False)
    device = resolve_device()
    model = build_model(config).to(device)
    incompatible = model.load_state_dict(_resolve_state_dict(checkpoint, use_ema=args.use_ema), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "checkpoint_load_warning "
            f"missing={list(incompatible.missing_keys)} unexpected={list(incompatible.unexpected_keys)}"
        )
    model.eval()

    if args.sample_index is not None:
        indices = [int(args.sample_index)]
    else:
        population = list(range(len(dataset)))
        random.shuffle(population)
        indices = population[: max(1, int(args.num_samples))]

    feature_source = str(config["model"].get("pixel_line_feature_source", "tight")).lower()
    visual_scale = feature_source if feature_source in {"tight", "context", "neighborhood"} else "tight"

    for sample_index in indices:
        sample = dataset[sample_index]
        batch = xbd_instance_collate_fn([sample])
        batch = _move_batch_to_device(batch, device)
        with torch.inference_mode():
            outputs = model(batch)
        if outputs.get("pixel_class_probabilities") is None or outputs.get("pixel_pred_labels") is None:
            raise RuntimeError("Pixel ordinal line is not enabled in the loaded config/checkpoint.")

        image_key_pre = f"pre_{visual_scale}"
        image_key_post = f"post_{visual_scale}"
        mask_key = f"mask_{visual_scale}"
        pre_image = _unnormalize(batch[image_key_pre][0].detach().cpu(), config)
        post_image = _unnormalize(batch[image_key_post][0].detach().cpu(), config)
        target_mask = batch[mask_key][0:1].float()
        resized_mask = F.interpolate(target_mask, size=outputs["pixel_pred_labels"].shape[-2:], mode="nearest")[0, 0].detach().cpu().numpy() > 0.5
        pixel_pred = outputs["pixel_pred_labels"][0].detach().cpu().numpy()
        pixel_probs = outputs["pixel_class_probabilities"][0].detach().cpu()
        pred_rgb = _render_argmax_map(pixel_pred)
        pred_rgb[~resized_mask] = 0.5

        figure, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.reshape(2, 4)
        axes[0, 0].imshow(pre_image)
        axes[0, 0].set_title("Pre Crop")
        axes[0, 1].imshow(post_image)
        axes[0, 1].set_title("Post Crop")
        axes[0, 2].imshow(batch[mask_key][0, 0].detach().cpu().numpy(), cmap="gray")
        axes[0, 2].set_title("Oracle Mask")
        axes[0, 3].imshow(pred_rgb)
        axes[0, 3].set_title("Pixel Argmax")

        heatmap_titles = ["No Damage", "Minor", "Major", "Destroyed"]
        for class_index in range(4):
            heatmap = pixel_probs[class_index].numpy()
            heatmap = np.where(resized_mask, heatmap, np.nan)
            axes[1, class_index].imshow(heatmap, cmap="magma", vmin=0.0, vmax=1.0)
            axes[1, class_index].set_title(heatmap_titles[class_index])

        gt_label = int(batch["label"][0].item())
        instance_pred = int(outputs.get("pred_labels", outputs.get("pred_label"))[0].item())
        pixel_agg_pred = int(outputs["pixel_instance_pred_labels"][0].item())
        for axis in axes.flat:
            axis.set_xticks([])
            axis.set_yticks([])
        figure.suptitle(
            f"sample={sample_index} gt={CLASS_NAMES[gt_label]} "
            f"instance={CLASS_NAMES[instance_pred]} pixel_agg={CLASS_NAMES[pixel_agg_pred]}",
            fontsize=12,
        )
        figure.tight_layout()
        output_path = save_dir / f"sample_{sample_index:06d}.png"
        figure.savefig(output_path, dpi=160)
        plt.close(figure)
        print(f"saved {output_path}")


if __name__ == "__main__":
    main()
