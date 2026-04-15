"""xBD dataset loading for pre-disaster building segmentation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
SAM3_REQUIRED_IMAGE_SIZE = 1008


@dataclass(frozen=True)
class XBDSample:
    image_id: str
    split: str
    image_path: Path
    target_path: Path
    label_path: Path
    building_count: int


def _is_pre_disaster_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS and path.stem.endswith(
        "_pre_disaster"
    )


def _load_split_list(list_file: Path) -> list[str]:
    with list_file.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def scan_xbd_split(
    xbd_root: str | Path,
    split: str,
    *,
    use_list_file: bool = False,
    list_file: Optional[str | Path] = None,
) -> list[XBDSample]:
    """Scan a split and return matched pre-disaster image/mask/label triplets."""
    split_root = Path(xbd_root).expanduser().resolve() / split
    image_dir = split_root / "images"
    target_dir = split_root / "targets"
    label_dir = split_root / "labels"

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory not found: {target_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    image_ids: list[str]
    if use_list_file:
        if list_file is None:
            default_list = Path(xbd_root).expanduser().resolve() / "xBD_list" / f"{split}_all.txt"
            list_file = default_list
        list_path = Path(list_file).expanduser().resolve()
        if not list_path.exists():
            raise FileNotFoundError(f"Split list file not found: {list_path}")
        image_ids = [f"{name}_pre_disaster" for name in _load_split_list(list_path)]
    else:
        image_ids = sorted(path.stem for path in image_dir.iterdir() if _is_pre_disaster_image(path))

    samples: list[XBDSample] = []
    for image_id in image_ids:
        image_path = next(
            (candidate for candidate in sorted(image_dir.glob(f"{image_id}.*")) if candidate.suffix.lower() in IMAGE_EXTENSIONS),
            None,
        )
        target_path = target_dir / f"{image_id}_target.png"
        label_path = label_dir / f"{image_id}.json"
        if image_path is None:
            raise FileNotFoundError(f"Image file missing for {image_id} in {image_dir}")
        if not target_path.exists():
            raise FileNotFoundError(f"Target mask missing for {image_id}: {target_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Label JSON missing for {image_id}: {label_path}")
        samples.append(
            XBDSample(
                image_id=image_id,
                split=split,
                image_path=image_path,
                target_path=target_path,
                label_path=label_path,
                building_count=_load_building_count(label_path),
            )
        )
    return samples


def load_binary_mask(mask_path: str | Path) -> np.ndarray:
    """Load a mask and normalize it to {0, 1}."""
    mask = Image.open(mask_path).convert("L")
    mask_np = np.asarray(mask, dtype=np.uint8)
    return (mask_np > 0).astype(np.uint8)


def load_image_rgb(image_path: str | Path) -> Image.Image:
    """Load an RGB image."""
    return Image.open(image_path).convert("RGB")


def compute_sample_foreground_ratios(
    sample: XBDSample,
    *,
    image_size: int,
) -> tuple[float, float]:
    """Return original/resized foreground ratios for a sample's target mask."""
    mask_original_np = load_binary_mask(sample.target_path)
    mask_resized_np = _resize_mask(mask_original_np, image_size)
    return float(mask_original_np.mean()), float(mask_resized_np.mean())


def _resize_image(image: Image.Image, image_size: int) -> np.ndarray:
    resized = image.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def _resize_mask(mask: np.ndarray, image_size: int) -> np.ndarray:
    mask_image = Image.fromarray(mask.astype(np.uint8))
    resized = mask_image.resize((image_size, image_size), resample=Image.Resampling.NEAREST)
    return (np.asarray(resized, dtype=np.uint8) > 0).astype(np.float32)


def _normalize_image(image_np: np.ndarray) -> torch.Tensor:
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    return (image_tensor - 0.5) / 0.5


def _apply_train_augmentations(
    image_np: np.ndarray,
    mask_np: np.ndarray,
    *,
    aug_hflip: bool,
    aug_vflip: bool,
    aug_rot90: bool,
    aug_brightness_contrast: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply lightweight spatial augmentations that keep image/mask aligned."""
    if aug_hflip and torch.rand(1).item() < 0.5:
        image_np = np.flip(image_np, axis=1).copy()
        mask_np = np.flip(mask_np, axis=1).copy()
    if aug_vflip and torch.rand(1).item() < 0.5:
        image_np = np.flip(image_np, axis=0).copy()
        mask_np = np.flip(mask_np, axis=0).copy()
    if aug_rot90:
        rotations = int(torch.randint(low=0, high=4, size=(1,)).item())
        if rotations:
            image_np = np.rot90(image_np, k=rotations, axes=(0, 1)).copy()
            mask_np = np.rot90(mask_np, k=rotations, axes=(0, 1)).copy()
    if aug_brightness_contrast:
        brightness_delta = (torch.rand(1).item() * 2.0 - 1.0) * 0.08 * 255.0
        contrast_scale = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * 0.08
        image_np = np.clip(image_np * contrast_scale + brightness_delta, 0.0, 255.0)
    return image_np, mask_np


def _load_building_count(label_path: Path) -> int:
    with label_path.open("r", encoding="utf-8") as handle:
        label_data = json.load(handle)
    features = label_data.get("features", {})
    xy_features = features.get("xy", [])
    return sum(
        1
        for feature in xy_features
        if feature.get("properties", {}).get("feature_type") == "building"
    )


def split_train_val_samples(
    samples: Sequence[XBDSample],
    *,
    val_ratio: float,
    split_seed: int,
) -> tuple[list[XBDSample], list[XBDSample]]:
    """Create a deterministic train/val split from the xBD train split only."""
    samples = list(samples)
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if not samples:
        return [], []
    if val_ratio <= 0.0 or len(samples) < 2:
        return samples, []

    indices = list(range(len(samples)))
    rng = random.Random(split_seed)
    rng.shuffle(indices)

    val_size = max(1, int(round(len(samples) * val_ratio)))
    val_size = min(val_size, len(samples) - 1)
    val_indices = set(indices[:val_size])

    train_samples = [sample for index, sample in enumerate(samples) if index not in val_indices]
    val_samples = [sample for index, sample in enumerate(samples) if index in val_indices]
    return train_samples, val_samples


class XBDPreDisasterDataset(Dataset[dict[str, Any]]):
    """Dataset returning resized tensors plus original-resolution masks."""

    def __init__(
        self,
        xbd_root: str | Path,
        split: str,
        *,
        image_size: int = 1008,
        use_list_files: bool = False,
        list_file: Optional[str | Path] = None,
        samples: Optional[Sequence[XBDSample]] = None,
        enable_augment: bool = False,
        aug_hflip: bool = True,
        aug_vflip: bool = True,
        aug_rot90: bool = True,
        aug_brightness_contrast: bool = False,
    ) -> None:
        if image_size != SAM3_REQUIRED_IMAGE_SIZE:
            raise ValueError(
                f"The current SAM3 adapter expects image_size={SAM3_REQUIRED_IMAGE_SIZE}, got {image_size}."
            )
        self.xbd_root = Path(xbd_root).expanduser().resolve()
        self.split = split
        self.image_size = image_size
        self.enable_augment = enable_augment
        self.aug_hflip = aug_hflip
        self.aug_vflip = aug_vflip
        self.aug_rot90 = aug_rot90
        self.aug_brightness_contrast = aug_brightness_contrast
        self.samples = list(samples) if samples is not None else scan_xbd_split(
            self.xbd_root,
            split,
            use_list_file=use_list_files,
            list_file=list_file,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = load_image_rgb(sample.image_path)
        original_width, original_height = image.size
        mask_original_np = load_binary_mask(sample.target_path)
        mask_resized_np = _resize_mask(mask_original_np, self.image_size)
        image_resized_np = _resize_image(image, self.image_size)

        if self.enable_augment:
            image_resized_np, mask_resized_np = _apply_train_augmentations(
                image_resized_np,
                mask_resized_np,
                aug_hflip=self.aug_hflip,
                aug_vflip=self.aug_vflip,
                aug_rot90=self.aug_rot90,
                aug_brightness_contrast=self.aug_brightness_contrast,
            )

        image_tensor = _normalize_image(image_resized_np)
        mask_tensor = torch.from_numpy(mask_resized_np).unsqueeze(0).float()
        mask_original = torch.from_numpy(mask_original_np).unsqueeze(0).float()
        foreground_ratio = float(mask_original_np.mean())
        foreground_ratio_resized = float(mask_resized_np.mean())

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "mask_original": mask_original,
            "image_id": sample.image_id,
            "split": sample.split,
            "image_path": str(sample.image_path),
            "target_path": str(sample.target_path),
            "label_path": str(sample.label_path),
            "original_height": original_height,
            "original_width": original_width,
            "building_count": sample.building_count,
            "foreground_ratio": foreground_ratio,
            "foreground_ratio_resized": foreground_ratio_resized,
        }
