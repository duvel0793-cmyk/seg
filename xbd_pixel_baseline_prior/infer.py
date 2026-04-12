from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as TF

from src.datasets.xbd_dataset import IMAGENET_MEAN, IMAGENET_STD, resolve_prior_path
from src.engine.trainer import get_autocast_context
from src.models.resnet18_fpn_bda import build_resnet18_fpn_bda
from src.utils.io import ensure_dir, load_checkpoint, load_yaml_config, pretty_yaml, resolve_device, strip_meta
from src.utils.logger import setup_logger
from src.utils.visualize import save_overlay, save_prediction_mask


@dataclass(frozen=True)
class InferenceRecord:
    sample_id: str
    pre_image_path: Path
    post_image_path: Path
    prior_path: Optional[Path] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for the lightweight xBD pixel baseline.")
    parser.add_argument("--config", type=str, default="configs/baseline_no_prior.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Single xBD image path or directory.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--prior", type=str, default=None, help="Single prior file for single-image inference.")
    parser.add_argument("--prior-dir", type=str, default=None, help="Prior dir override for directory inference.")
    return parser.parse_args()


def _sample_id_from_path(path: Path) -> tuple[str, str]:
    if path.name.endswith("_post_disaster.png"):
        return path.name.replace("_post_disaster.png", ""), "post"
    if path.name.endswith("_pre_disaster.png"):
        return path.name.replace("_pre_disaster.png", ""), "pre"
    raise ValueError(
        f"Expected xBD-style file name ending with _pre_disaster.png or _post_disaster.png, got {path.name}"
    )


def discover_inference_records(
    input_path: Path,
    prior_mode: str,
    prior_dir: Optional[Path],
    prior_file: Optional[Path],
    prior_filename_pattern: str,
) -> list[InferenceRecord]:
    records: list[InferenceRecord] = []

    if input_path.is_file():
        sample_id, role = _sample_id_from_path(input_path)
        pre_image_path = input_path if role == "pre" else input_path.with_name(f"{sample_id}_pre_disaster.png")
        post_image_path = input_path if role == "post" else input_path.with_name(f"{sample_id}_post_disaster.png")

        if not pre_image_path.exists() or not post_image_path.exists():
            raise FileNotFoundError(
                f"Could not pair pre/post images for {input_path}. "
                f"Expected {pre_image_path} and {post_image_path}."
            )

        resolved_prior = None
        if prior_mode == "input_channel":
            if prior_file is not None:
                resolved_prior = prior_file
            elif prior_dir is not None:
                resolved_prior = resolve_prior_path(sample_id, prior_dir, prior_filename_pattern)
            if resolved_prior is None or not resolved_prior.exists():
                raise FileNotFoundError(
                    f"prior_mode=input_channel requires a prior mask for {sample_id}, but none was found."
                )

        records.append(
            InferenceRecord(
                sample_id=sample_id,
                pre_image_path=pre_image_path.resolve(),
                post_image_path=post_image_path.resolve(),
                prior_path=resolved_prior.resolve() if resolved_prior is not None else None,
            )
        )
        return records

    if not input_path.is_dir():
        raise FileNotFoundError(f"Inference input not found: {input_path}")

    post_images = sorted(input_path.glob("*_post_disaster.png"))
    if not post_images:
        raise FileNotFoundError(f"No *_post_disaster.png files found under {input_path}")

    for post_image_path in post_images:
        sample_id, _ = _sample_id_from_path(post_image_path)
        pre_image_path = input_path / f"{sample_id}_pre_disaster.png"
        if not pre_image_path.exists():
            raise FileNotFoundError(f"Missing pre image for {sample_id}: {pre_image_path}")

        resolved_prior = None
        if prior_mode == "input_channel":
            if prior_dir is None:
                raise ValueError("prior_mode=input_channel requires --prior-dir or data.prior_dir for directory inference.")
            resolved_prior = resolve_prior_path(sample_id, prior_dir, prior_filename_pattern)
            if resolved_prior is None:
                raise FileNotFoundError(f"Missing prior mask for {sample_id} in {prior_dir}")

        records.append(
            InferenceRecord(
                sample_id=sample_id,
                pre_image_path=pre_image_path.resolve(),
                post_image_path=post_image_path.resolve(),
                prior_path=resolved_prior.resolve() if resolved_prior is not None else None,
            )
        )
    return records


def preprocess_record(record: InferenceRecord, prior_mode: str) -> tuple[torch.Tensor, tuple[int, int], Image.Image]:
    pre_image = Image.open(record.pre_image_path).convert("RGB")
    post_image = Image.open(record.post_image_path).convert("RGB")

    if pre_image.size != post_image.size:
        raise ValueError(
            f"Pre/Post image size mismatch for {record.sample_id}: {pre_image.size} vs {post_image.size}"
        )

    pre_tensor = TF.normalize(TF.to_tensor(pre_image), mean=IMAGENET_MEAN, std=IMAGENET_STD)
    post_tensor = TF.normalize(TF.to_tensor(post_image), mean=IMAGENET_MEAN, std=IMAGENET_STD)
    image_tensor = torch.cat([pre_tensor, post_tensor], dim=0)

    if prior_mode == "input_channel":
        assert record.prior_path is not None
        prior_image = Image.open(record.prior_path).convert("L")
        if prior_image.size != post_image.size:
            prior_image = prior_image.resize(post_image.size, resample=Image.NEAREST)
        prior_array = (np.array(prior_image, dtype=np.uint8) > 0).astype(np.float32)
        prior_tensor = torch.from_numpy(prior_array).unsqueeze(0)
        image_tensor = torch.cat([image_tensor, prior_tensor], dim=0)

    return image_tensor.unsqueeze(0).float(), post_image.size, post_image


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    if args.prior_dir is not None:
        config["data"]["prior_dir"] = args.prior_dir

    requested_device = str(config["device"])
    device = resolve_device(requested_device)
    output_dir = ensure_dir(args.output_dir)
    logger = setup_logger(log_dir=output_dir)
    logger.info("Starting inference")
    logger.info("Resolved config:\n%s", pretty_yaml(strip_meta(config)))

    model = build_resnet18_fpn_bda(config["model"]).to(device)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    prior_dir = Path(config["data"]["prior_dir"]).expanduser().resolve() if config["data"]["prior_dir"] else None
    prior_file = Path(args.prior).expanduser().resolve() if args.prior else None
    records = discover_inference_records(
        input_path=Path(args.input).expanduser().resolve(),
        prior_mode=config["data"]["prior_mode"],
        prior_dir=prior_dir,
        prior_file=prior_file,
        prior_filename_pattern=config["data"].get("prior_filename_pattern", "{sample_id}.png"),
    )

    use_amp = bool(config["train"].get("amp", False) and device.type == "cuda")
    mask_dir = ensure_dir(output_dir / "pred_masks")
    overlay_dir = ensure_dir(output_dir / "overlays")

    progress_bar = tqdm(records, desc="Infer", dynamic_ncols=True)
    with torch.no_grad():
        for record in progress_bar:
            image_tensor, input_size, post_image = preprocess_record(record, config["data"]["prior_mode"])
            image_tensor = image_tensor.to(device, non_blocking=device.type == "cuda")

            with get_autocast_context(device, enabled=use_amp):
                outputs = model(image_tensor)

            prediction = outputs["damage_logits"].argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
            output_size = tuple(int(value) for value in outputs["damage_logits"].shape[-2:])
            unique_values, counts = np.unique(prediction, return_counts=True)
            pixel_stats = {int(value): int(count) for value, count in zip(unique_values.tolist(), counts.tolist())}

            save_prediction_mask(prediction, mask_dir / f"{record.sample_id}_pred.png")
            save_overlay(post_image, prediction, overlay_dir / f"{record.sample_id}_overlay.png")

            logger.info(
                "Sample %s | input_size=%s | output_size=%s | class_pixel_stats=%s",
                record.sample_id,
                input_size,
                output_size,
                pixel_stats,
            )


if __name__ == "__main__":
    main()
