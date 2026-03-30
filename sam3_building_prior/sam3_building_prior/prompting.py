"""Prompt extraction helpers for text-only and coarse-mask-guided runs."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from skimage import measure

from .config import IMAGE_EXTENSIONS
from .io_utils import resolve_matching_image_path
from .types import PromptRegion


def resolve_mask_path(image_path: Path, mask_dir: Path) -> Optional[Path]:
    """Resolve the coarse mask that matches an input image name."""
    return resolve_matching_image_path(
        search_root=mask_dir,
        candidate_names=[image_path.name],
        candidate_stems=[image_path.stem],
        suffixes=IMAGE_EXTENSIONS,
    )


def read_binary_mask(path: Path) -> np.ndarray:
    """Read a single-channel binary mask as a boolean array."""
    from PIL import Image

    mask = Image.open(path).convert("L")
    return np.array(mask) > 0


def _component_center_point(component_mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(component_mask)
    if len(xs) == 0:
        return (0, 0)

    center_x = xs.mean()
    center_y = ys.mean()
    distances = (xs - center_x) ** 2 + (ys - center_y) ** 2
    idx = int(np.argmin(distances))
    return (int(xs[idx]), int(ys[idx]))


def extract_prompt_regions(mask: np.ndarray, min_area: int = 64) -> List[PromptRegion]:
    """Extract coarse connected components as prompt regions."""
    labels = measure.label(mask.astype(np.uint8), connectivity=1)
    props = measure.regionprops(labels)

    prompts: List[PromptRegion] = []
    prompt_id = 1
    for prop in props:
        area = int(prop.area)
        if area < min_area:
            continue

        minr, minc, maxr, maxc = prop.bbox
        component_mask = labels == prop.label
        prompts.append(
            PromptRegion(
                prompt_id=prompt_id,
                bbox_xyxy=(int(minc), int(minr), int(maxc), int(maxr)),
                center_point_xy=_component_center_point(component_mask),
                area=area,
            )
        )
        prompt_id += 1

    return prompts


def prompt_regions_to_json(
    image_path: Path,
    prompt_mode: str,
    text_prompt: str,
    coarse_mask_path: Optional[Path],
    prompt_min_area: int,
    prompt_regions: List[PromptRegion],
) -> dict:
    """Serialize generated prompts into a JSON-friendly payload."""
    return {
        "image": str(image_path),
        "image_name": image_path.name,
        "prompt_mode": prompt_mode,
        "text_prompt": text_prompt,
        "coarse_mask": str(coarse_mask_path) if coarse_mask_path is not None else None,
        "prompt_min_area": prompt_min_area,
        "num_prompts": len(prompt_regions),
        "prompts": [
            {
                "prompt_id": prompt.prompt_id,
                "bbox_xyxy": list(prompt.bbox_xyxy),
                "center_point_xy": list(prompt.center_point_xy),
                "area": prompt.area,
            }
            for prompt in prompt_regions
        ],
    }
