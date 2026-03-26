import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = PACKAGE_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from building_segmentation.utils.visualization import BUILDING_COLOR, overlay_mask, save_prediction_visualization


def load_image(path):
    image = np.asarray(imageio.imread(path))
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    return image.astype(np.uint8)


def load_mask(path):
    return np.asarray(imageio.imread(path), dtype=np.uint8)


def resolve_basenames(args):
    if args.basenames:
        return args.basenames
    if not args.data_list_path:
        raise ValueError("Either --basenames or --data_list_path must be provided")
    with open(args.data_list_path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()][:args.num_samples]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize xBD building masks")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_list_path", type=str)
    parser.add_argument("--basenames", nargs="*")
    parser.add_argument("--num_samples", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for basename in resolve_basenames(args):
        image = load_image(dataset_path / "images" / f"{basename}_pre_disaster.png")
        mask = load_mask(dataset_path / "targets" / f"{basename}_pre_disaster_target.png")
        overlay = overlay_mask(image, mask, color=BUILDING_COLOR, alpha=0.45)

        save_prediction_visualization(
            output_path=output_dir / f"{basename}_gt.png",
            image=image,
            pred_mask=mask,
            gt_mask=mask,
            title=basename,
            image_is_normalized=False,
        )

        imageio.imwrite(output_dir / f"{basename}_overlay.png", overlay)
        print(output_dir / f"{basename}_gt.png")


if __name__ == "__main__":
    main()
