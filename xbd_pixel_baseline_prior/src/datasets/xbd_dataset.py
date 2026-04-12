from __future__ import annotations

import random
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PRIOR_MODES = {"none", "input_channel", "loss_weight"}


@dataclass(frozen=True)
class XBDSampleRecord:
    sample_id: str
    pre_image_path: Path
    post_image_path: Path
    post_target_path: Path
    prior_path: Optional[Path] = None


def _deduplicate(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    return unique_paths


def resolve_target_path(sample_id: str, image_dir: Path, target_dir: Optional[Path]) -> Optional[Path]:
    """Resolve sample_id -> target path using local xBD layout fallbacks."""
    target_name = f"{sample_id}_post_disaster_target.png"
    split_name = image_dir.parent.name

    candidates: list[Path] = []
    if target_dir is not None:
        candidates.extend(
            [
                target_dir / target_name,
                target_dir / "targets" / target_name,
                target_dir / split_name / "targets" / target_name,
            ]
        )

    candidates.append(image_dir.parent / "targets" / target_name)

    for candidate in _deduplicate(candidates):
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_prior_path(
    sample_id: str,
    prior_dir: Path,
    prior_filename_pattern: str = "{sample_id}.png",
) -> Optional[Path]:
    try:
        configured_name = prior_filename_pattern.format(sample_id=sample_id)
    except KeyError as exc:
        raise ValueError(
            "prior_filename_pattern must support {sample_id}, "
            f"got: {prior_filename_pattern}"
        ) from exc

    candidates = _deduplicate(
        [
            prior_dir / configured_name,
            prior_dir / f"{sample_id}.png",
            prior_dir / f"{sample_id}_post_disaster.png",
            prior_dir / f"{sample_id}_post_disaster_mask.png",
            prior_dir / f"{sample_id}_building_mask.png",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _read_sample_ids(list_path: Path) -> list[str]:
    if not list_path.exists():
        raise FileNotFoundError(f"Split list not found: {list_path}")
    sample_ids = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not sample_ids:
        raise ValueError(f"Split list is empty: {list_path}")
    return sample_ids


def format_data_check(summary: dict[str, Any]) -> str:
    lines = [
        f"list_samples: {summary['list_samples']}",
        f"valid_samples: {summary['valid_samples']}",
        f"missing_samples: {summary['missing_samples']}",
        f"mapping_rule: {summary['mapping_rule']}",
        f"label_stats_files: {summary['label_stats_files']}",
        f"label_value_histogram: {summary['label_value_histogram']}",
    ]
    if summary["missing_examples"]:
        lines.append(f"missing_examples: {summary['missing_examples']}")
    return "\n".join(lines)


class XBDDataset(Dataset[dict[str, Any]]):
    """
    Sample mapping rule:
    sample_id -> {image_dir}/{sample_id}_pre_disaster.png
              -> {image_dir}/{sample_id}_post_disaster.png
              -> {target_dir}/{sample_id}_post_disaster_target.png
    with fallbacks that also support the local split-specific layout:
    {root}/{split}/targets/{sample_id}_post_disaster_target.png
    """

    def __init__(
        self,
        list_path: str | Path,
        image_dir: str | Path,
        target_dir: Optional[str | Path] = None,
        prior_dir: Optional[str | Path] = None,
        prior_mode: str = "none",
        crop_size: Optional[int] = 256,
        is_train: bool = False,
        strict_data_check: bool = False,
        prior_filename_pattern: str = "{sample_id}.png",
    ) -> None:
        super().__init__()

        self.list_path = Path(list_path).expanduser().resolve()
        self.image_dir = Path(image_dir).expanduser().resolve()
        self.target_dir = Path(target_dir).expanduser().resolve() if target_dir else None
        self.prior_dir = Path(prior_dir).expanduser().resolve() if prior_dir else None
        self.prior_mode = prior_mode
        self.crop_size = int(crop_size) if crop_size else None
        self.is_train = is_train
        self.strict_data_check = strict_data_check
        self.prior_filename_pattern = prior_filename_pattern

        if self.prior_mode not in PRIOR_MODES:
            raise ValueError(f"Unsupported prior_mode: {self.prior_mode}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if self.prior_mode != "none" and self.prior_dir is None:
            raise ValueError(
                "prior_mode is not 'none', but data.prior_dir is missing. "
                "Please set data.prior_dir in the yaml config."
            )
        if self.prior_dir is not None and not self.prior_dir.exists():
            raise FileNotFoundError(f"Prior directory not found: {self.prior_dir}")

        self.sample_ids = _read_sample_ids(self.list_path)
        self.records: list[XBDSampleRecord] = []
        self.missing_records: list[dict[str, Any]] = []
        self.mapping_rule = (
            f"{self.image_dir}/{{sample_id}}_[pre|post]_disaster.png + "
            "resolved post target via configured target_dir or sibling split targets directory"
        )
        self._build_records()

    def _build_records(self) -> None:
        for sample_id in self.sample_ids:
            pre_image_path = self.image_dir / f"{sample_id}_pre_disaster.png"
            post_image_path = self.image_dir / f"{sample_id}_post_disaster.png"
            post_target_path = resolve_target_path(
                sample_id=sample_id,
                image_dir=self.image_dir,
                target_dir=self.target_dir,
            )

            missing_parts: list[str] = []
            if not pre_image_path.exists():
                missing_parts.append("pre_image")
            if not post_image_path.exists():
                missing_parts.append("post_image")
            if post_target_path is None:
                missing_parts.append("post_target")

            prior_path: Optional[Path] = None
            if self.prior_mode != "none":
                assert self.prior_dir is not None
                prior_path = resolve_prior_path(
                    sample_id=sample_id,
                    prior_dir=self.prior_dir,
                    prior_filename_pattern=self.prior_filename_pattern,
                )
                if prior_path is None:
                    missing_parts.append("prior")

            if missing_parts:
                self.missing_records.append(
                    {
                        "sample_id": sample_id,
                        "missing": missing_parts,
                    }
                )
                continue

            assert post_target_path is not None
            self.records.append(
                XBDSampleRecord(
                    sample_id=sample_id,
                    pre_image_path=pre_image_path.resolve(),
                    post_image_path=post_image_path.resolve(),
                    post_target_path=post_target_path.resolve(),
                    prior_path=prior_path,
                )
            )

        if self.missing_records:
            examples = self.missing_records[:5]
            message = (
                f"{len(self.missing_records)} samples are missing files. "
                f"Examples: {examples}"
            )
            if self.strict_data_check:
                raise FileNotFoundError(message)
            warnings.warn(message, stacklevel=2)

    def __len__(self) -> int:
        return len(self.records)

    def self_check(self, max_label_files: Optional[int] = 128) -> dict[str, Any]:
        inspected_records = self.records
        if max_label_files is not None and max_label_files >= 0:
            inspected_records = self.records[:max_label_files]

        label_counter: Counter[int] = Counter()
        for record in inspected_records:
            label_array = np.array(Image.open(record.post_target_path), dtype=np.uint8)
            values, counts = np.unique(label_array, return_counts=True)
            for value, count in zip(values.tolist(), counts.tolist()):
                label_counter[int(value)] += int(count)

        return {
            "list_samples": len(self.sample_ids),
            "valid_samples": len(self.records),
            "missing_samples": len(self.missing_records),
            "missing_examples": self.missing_records[:5],
            "mapping_rule": self.mapping_rule,
            "label_stats_files": len(inspected_records),
            "label_value_histogram": dict(sorted(label_counter.items())),
        }

    def _load_rgb(self, path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def _load_mask(self, path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("L")

    def _random_crop(
        self,
        pre_image: Image.Image,
        post_image: Image.Image,
        target: Image.Image,
        prior: Optional[Image.Image],
    ) -> tuple[Image.Image, Image.Image, Image.Image, Optional[Image.Image]]:
        if self.crop_size is None:
            return pre_image, post_image, target, prior

        width, height = pre_image.size
        if width < self.crop_size or height < self.crop_size:
            raise ValueError(
                f"Crop size {self.crop_size} is larger than image size {(width, height)} "
                f"for sample from {self.image_dir}"
            )

        top = random.randint(0, height - self.crop_size)
        left = random.randint(0, width - self.crop_size)

        pre_image = TF.crop(pre_image, top, left, self.crop_size, self.crop_size)
        post_image = TF.crop(post_image, top, left, self.crop_size, self.crop_size)
        target = TF.crop(target, top, left, self.crop_size, self.crop_size)
        if prior is not None:
            prior = TF.crop(prior, top, left, self.crop_size, self.crop_size)
        return pre_image, post_image, target, prior

    def _random_flips(
        self,
        pre_image: Image.Image,
        post_image: Image.Image,
        target: Image.Image,
        prior: Optional[Image.Image],
    ) -> tuple[Image.Image, Image.Image, Image.Image, Optional[Image.Image]]:
        if random.random() < 0.5:
            pre_image = TF.hflip(pre_image)
            post_image = TF.hflip(post_image)
            target = TF.hflip(target)
            if prior is not None:
                prior = TF.hflip(prior)
        if random.random() < 0.5:
            pre_image = TF.vflip(pre_image)
            post_image = TF.vflip(post_image)
            target = TF.vflip(target)
            if prior is not None:
                prior = TF.vflip(prior)
        return pre_image, post_image, target, prior

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        pre_image = self._load_rgb(record.pre_image_path)
        post_image = self._load_rgb(record.post_image_path)
        target = self._load_mask(record.post_target_path)
        prior = self._load_mask(record.prior_path) if record.prior_path is not None else None

        if pre_image.size != post_image.size:
            raise ValueError(
                f"Pre/Post image size mismatch for {record.sample_id}: "
                f"{pre_image.size} vs {post_image.size}"
            )
        if target.size != post_image.size:
            raise ValueError(
                f"Target size mismatch for {record.sample_id}: {target.size} vs {post_image.size}"
            )
        if prior is not None and prior.size != post_image.size:
            prior = prior.resize(post_image.size, resample=Image.NEAREST)

        if self.is_train:
            pre_image, post_image, target, prior = self._random_crop(pre_image, post_image, target, prior)
            pre_image, post_image, target, prior = self._random_flips(pre_image, post_image, target, prior)

        pre_tensor = TF.normalize(TF.to_tensor(pre_image), mean=IMAGENET_MEAN, std=IMAGENET_STD)
        post_tensor = TF.normalize(TF.to_tensor(post_image), mean=IMAGENET_MEAN, std=IMAGENET_STD)
        image_tensor = torch.cat([pre_tensor, post_tensor], dim=0)

        target_array = np.array(target, dtype=np.int64)
        target_tensor = torch.from_numpy(target_array.copy()).long()
        loc_target_tensor = (target_tensor > 0).float().unsqueeze(0)

        if prior is not None:
            prior_array = (np.array(prior, dtype=np.uint8) > 0).astype(np.float32)
            prior_tensor = torch.from_numpy(prior_array).unsqueeze(0)
        else:
            prior_tensor = torch.zeros((1, target_tensor.shape[0], target_tensor.shape[1]), dtype=torch.float32)

        if self.prior_mode == "input_channel":
            image_tensor = torch.cat([image_tensor, prior_tensor], dim=0)

        return {
            "image": image_tensor.float(),
            "target": target_tensor,
            "loc_target": loc_target_tensor,
            "prior_mask": prior_tensor.float(),
            "sample_id": record.sample_id,
            "pre_path": str(record.pre_image_path),
            "post_path": str(record.post_image_path),
            "target_path": str(record.post_target_path),
        }
