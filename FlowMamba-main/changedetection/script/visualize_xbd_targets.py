import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


BUILDING_COLOR = np.array([0, 255, 0], dtype=np.uint8)
DAMAGE_COLORS = {
    0: np.array([0, 0, 0], dtype=np.uint8),
    1: np.array([0, 255, 0], dtype=np.uint8),
    2: np.array([255, 128, 0], dtype=np.uint8),
    3: np.array([153, 51, 255], dtype=np.uint8),
    4: np.array([255, 0, 0], dtype=np.uint8),
}


def load_image(path: Path) -> np.ndarray:
    image = np.asarray(imageio.imread(path))
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    return image.astype(np.uint8)


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(imageio.imread(path), dtype=np.uint8)


def blend_single_mask(image: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> np.ndarray:
    output = image.astype(np.float32).copy()
    valid = mask > 0
    output[valid] = (1.0 - alpha) * output[valid] + alpha * color.astype(np.float32)
    return np.clip(output, 0, 255).astype(np.uint8)


def blend_damage_mask(image: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    output = image.astype(np.float32).copy()
    for label, color in DAMAGE_COLORS.items():
        if label == 0:
            continue
        valid = mask == label
        output[valid] = (1.0 - alpha) * output[valid] + alpha * color.astype(np.float32)
    return np.clip(output, 0, 255).astype(np.uint8)


def render_binary_mask(mask: np.ndarray) -> np.ndarray:
    output = np.zeros(mask.shape + (3,), dtype=np.uint8)
    output[mask > 0] = 255
    return output


def resize_panel(image: np.ndarray, panel_size: int) -> Image.Image:
    return Image.fromarray(image).resize((panel_size, panel_size), Image.Resampling.BILINEAR)


def compose_visualization(dataset_path: Path, basename: str, panel_size: int) -> Image.Image:
    pre_image = load_image(dataset_path / "images" / f"{basename}_pre_disaster.png")
    post_image = load_image(dataset_path / "images" / f"{basename}_post_disaster.png")
    pre_target = load_mask(dataset_path / "targets" / f"{basename}_pre_disaster_target.png")
    post_target = load_mask(dataset_path / "targets" / f"{basename}_post_disaster_target.png")

    pre_overlay = blend_single_mask(pre_image, pre_target, BUILDING_COLOR, alpha=0.45)
    post_overlay = blend_damage_mask(post_image, post_target, alpha=0.45)
    binary_mask = render_binary_mask(pre_target)

    panels = [
        ("pre image", pre_image),
        ("pre + building", pre_overlay),
        ("building mask", binary_mask),
        ("post + damage", post_overlay),
    ]

    title_height = 28
    padding = 12
    canvas = Image.new(
        "RGB",
        (padding * (len(panels) + 1) + panel_size * len(panels), padding * 3 + title_height * 2 + panel_size),
        color=(18, 18, 18),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((padding, padding), basename, fill=(255, 255, 255), font=font)

    for index, (title, panel) in enumerate(panels):
        x = padding + index * (panel_size + padding)
        y = padding * 2 + title_height
        draw.text((x, y - title_height), title, fill=(230, 230, 230), font=font)
        canvas.paste(resize_panel(panel, panel_size), (x, y))

    return canvas


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize xBD building and damage targets")
    parser.add_argument("--dataset_path", type=str, required=True, help="Split directory, e.g. /home/lky/data/xBD/train")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--basenames", nargs="+", required=True, help="Sample basenames without _pre/_post suffix")
    parser.add_argument("--panel_size", type=int, default=384)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for basename in args.basenames:
        vis = compose_visualization(dataset_path, basename, args.panel_size)
        out_path = output_dir / f"{basename}_targets_vis.png"
        vis.save(out_path)
        print(out_path)


if __name__ == "__main__":
    main()
