from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from building_segmentation.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD


BUILDING_COLOR = np.array([0, 255, 0], dtype=np.uint8)


def denormalize_image(image):
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    image = image.astype(np.float32)
    output = np.empty_like(image, dtype=np.float32)
    for channel in range(image.shape[-1]):
        output[..., channel] = image[..., channel] * IMAGENET_STD[channel] + IMAGENET_MEAN[channel]
    return np.clip(output, 0, 255).astype(np.uint8)


def mask_to_rgb(mask):
    output = np.zeros(mask.shape + (3,), dtype=np.uint8)
    output[mask > 0] = 255
    return output


def overlay_mask(image, mask, color=BUILDING_COLOR, alpha=0.45):
    image = image.astype(np.float32).copy()
    valid = mask > 0
    image[valid] = (1.0 - alpha) * image[valid] + alpha * color.astype(np.float32)
    return np.clip(image, 0, 255).astype(np.uint8)


def save_prediction_visualization(
    output_path,
    image,
    pred_mask,
    gt_mask=None,
    title=None,
    panel_size=384,
    image_is_normalized=True,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = denormalize_image(image) if image_is_normalized else image.astype(np.uint8)
    pred_overlay = overlay_mask(image, pred_mask)
    pred_mask_rgb = mask_to_rgb(pred_mask)
    panels = [
        ("image", image),
        ("pred overlay", pred_overlay),
        ("pred mask", pred_mask_rgb),
    ]

    if gt_mask is not None:
        gt_overlay = overlay_mask(image, gt_mask)
        gt_mask_rgb = mask_to_rgb(gt_mask)
        panels.extend(
            [
                ("gt overlay", gt_overlay),
                ("gt mask", gt_mask_rgb),
            ]
        )

    title_height = 28
    padding = 12
    canvas = Image.new(
        "RGB",
        (padding * (len(panels) + 1) + panel_size * len(panels), padding * 3 + title_height * 2 + panel_size),
        color=(18, 18, 18),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    if title:
        draw.text((padding, padding), title, fill=(255, 255, 255), font=font)

    for index, (panel_title, panel) in enumerate(panels):
        x = padding + index * (panel_size + padding)
        y = padding * 2 + title_height
        draw.text((x, y - title_height), panel_title, fill=(230, 230, 230), font=font)
        panel_image = Image.fromarray(panel).resize((panel_size, panel_size), Image.Resampling.BILINEAR)
        canvas.paste(panel_image, (x, y))

    canvas.save(output_path)
