"""Load one batch and export crop visualizations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.xbd_dataset import XBDInstanceDataset
from utils.common import ensure_dir
from utils.config import load_config
from utils.visualization import overlay_mask, save_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a Clean-MODG training batch.")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path.")
    return parser.parse_args()


def _tensor_to_vis(tensor, normalized: bool = True) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    array = np.transpose(array[:3], (1, 2, 0))
    if normalized:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        array = array * std + mean
    array = np.clip(array, 0.0, 1.0)
    return (array * 255).astype(np.uint8)


def _print_tensor_stats(name: str, tensor) -> None:
    array = tensor.detach().cpu().float()
    print(
        f"{name} stats | min={array.min().item():.4f} "
        f"max={array.max().item():.4f} mean={array.mean().item():.4f} std={array.std(unbiased=False).item():.4f}"
    )


def _print_mask_stats(name: str, tensor) -> None:
    array = tensor.detach().cpu().float()
    unique_values = sorted({float(x) for x in array.unique().tolist()})
    foreground_ratio = float((array > 0.5).float().mean().item())
    print(f"{name} stats | unique={unique_values} foreground_ratio={foreground_ratio:.4f}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset = XBDInstanceDataset(config["data"], split=config["data"].get("split_train", "train"))
    loader = DataLoader(dataset, batch_size=min(4, int(config["data"].get("batch_size", 4))), shuffle=False, num_workers=0)
    batch = next(iter(loader))
    for key, value in batch.items():
        if hasattr(value, "shape"):
            print(key, tuple(value.shape))
        else:
            print(key, type(value), len(value))
    for key in ["pre_tight", "post_tight", "pre_context", "post_context"]:
        _print_tensor_stats(key, batch[key])
    for key in ["mask_tight", "mask_context"]:
        _print_mask_stats(key, batch[key])

    output_dir = ensure_dir(Path("outputs/debug_batch"))
    images = []
    titles = []
    for idx in range(min(4, batch["label"].shape[0])):
        for prefix in ["pre_tight", "post_tight", "pre_context", "post_context"]:
            image = _tensor_to_vis(batch[prefix][idx], normalized=config["data"].get("normalize", "imagenet") == "imagenet")
            mask_key = "mask_tight" if "tight" in prefix else "mask_context"
            mask = batch[mask_key][idx].detach().cpu().numpy()[0]
            images.append(overlay_mask(image, mask))
            titles.append(f"{prefix} | y={int(batch['label'][idx])}")
    save_image_grid(images, titles, output_dir / "batch_grid.png", cols=4, figsize=(14, 12))
    print(f"Saved visualization to {output_dir / 'batch_grid.png'}")


if __name__ == "__main__":
    main()
