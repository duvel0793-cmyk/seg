import argparse
import os
import sys
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = PACKAGE_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from building_segmentation.datasets.xbd import make_dataloader, read_data_list
from building_segmentation.models.builder import build_model
from building_segmentation.utils.common import ensure_dir, load_checkpoint, save_json
from building_segmentation.utils.eval import evaluate_model


def parse_args():
    default_data_root = Path(os.environ.get("DATA_ROOT", "/home/lky/data/xBD"))
    default_output_dir = PACKAGE_ROOT / "outputs" / "eval_run"
    default_pretrained = PACKAGE_ROOT / "pretrained_weight" / "vssm_small_0229_ckpt_epoch_222.pth"
    parser = argparse.ArgumentParser(description="Evaluate FlowMamba building segmentation on xBD")
    parser.add_argument("--cfg", type=str, default=str(PACKAGE_ROOT / "configs/vssm_small_224.yaml"))
    parser.add_argument("--pretrained_weight_path", type=str, default=str(default_pretrained))
    parser.add_argument("--dataset_path", type=str, default=str(default_data_root / "test"))
    parser.add_argument("--data_list_path", type=str, default=str(default_data_root / "xBD_list/val_all.txt"))
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=str(default_output_dir))
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--vis_samples", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "FlowMamba building segmentation requires CUDA. "
            "The current vmamba selective-scan kernels do not support CPU inference/training."
        )
    device = torch.device("cuda")
    output_dir = ensure_dir(args.output_dir)

    data_loader = make_dataloader(
        dataset_path=args.dataset_path,
        data_list=read_data_list(args.data_list_path),
        crop_size=1024,
        batch_size=1,
        split="val",
        num_workers=args.num_workers,
        shuffle=False,
    )

    model = build_model(args.cfg, args.pretrained_weight_path).to(device)
    load_checkpoint(model, args.resume, optimizer=None, map_location="cpu")

    metrics = evaluate_model(
        model=model,
        data_loader=data_loader,
        device=device,
        save_dir=output_dir / "visualizations",
        num_visualizations=args.vis_samples,
    )
    save_json(metrics, output_dir / "metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
