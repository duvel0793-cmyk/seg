from typing import List, Optional
import numpy as np
from skimage import measure
from .types import InstancePrediction
import logging


def extract_instances_from_masks(
    masks: np.ndarray,
    min_area: int = 64,
    prompt_type: str = "text",
    scores: Optional[np.ndarray] = None,
) -> List[InstancePrediction]:
    """Split CxHxW masks into instance predictions via connected components."""
    instances = []
    if masks.size == 0:
        return instances

    inst_id = 1
    C = masks.shape[0]
    H = masks.shape[1]
    W = masks.shape[2]
    for c in range(C):
        mask = masks[c].astype(np.uint8)
        if mask.sum() == 0:
            continue
        # connected components to split multiple parts
        labels = measure.label(mask, connectivity=1)
        props = measure.regionprops(labels)
        for prop in props:
            area = int(prop.area)
            if area < min_area:
                continue
            minr, minc, maxr, maxc = prop.bbox
            bbox = (int(minc), int(minr), int(maxc), int(maxr))
            inst = InstancePrediction(
                instance_id=inst_id,
                bbox_xyxy=bbox,
                area=area,
                score=(float(scores[c]) if scores is not None and len(scores) > c else None),
                prompt_type=prompt_type,
                mask=(labels == prop.label),
            )
            instances.append(inst)
            inst_id += 1
    logging.debug(
        "Extracted %d instances after filtering (min_area=%d)",
        len(instances),
        min_area,
    )
    return instances


def extract_largest_instance_from_mask(
    mask: np.ndarray,
    instance_id: int,
    min_area: int = 64,
    score: Optional[float] = None,
    prompt_type: str = "box",
    prompt_id: Optional[int] = None,
    prompt_bbox_xyxy: Optional[tuple[int, int, int, int]] = None,
    prompt_point_xy: Optional[tuple[int, int]] = None,
    coarse_area: Optional[int] = None,
) -> Optional[InstancePrediction]:
    """Keep the largest connected component from a refined instance mask."""
    labels = measure.label(mask.astype(np.uint8), connectivity=1)
    props = [prop for prop in measure.regionprops(labels) if int(prop.area) >= min_area]
    if not props:
        return None

    prop = max(props, key=lambda item: int(item.area))
    minr, minc, maxr, maxc = prop.bbox
    bbox = (int(minc), int(minr), int(maxc), int(maxr))
    return InstancePrediction(
        instance_id=instance_id,
        bbox_xyxy=bbox,
        area=int(prop.area),
        score=score,
        prompt_type=prompt_type,
        mask=(labels == prop.label),
        prompt_id=prompt_id,
        prompt_bbox_xyxy=prompt_bbox_xyxy,
        prompt_point_xy=prompt_point_xy,
        coarse_area=coarse_area,
    )


def merge_instances_to_mask(instances: List[InstancePrediction], H: int, W: int):
    """Merge instance masks back into a single binary mask."""
    merged = np.zeros((H, W), dtype=np.uint8)
    for inst in instances:
        merged |= (inst.mask.astype(np.uint8))
    return merged
