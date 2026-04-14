from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


OLD_PROJECT = Path("/home/lky/code/oracle-instance-damage-classification_corn")
NEW_PROJECT = Path("/home/lky/code/oracle-instance-damage-classification_convnext_mcd_corn")
OLD_CONFIG = OLD_PROJECT / "configs" / "corn_safe_full_recipe.yaml"
NEW_CONFIG = NEW_PROJECT / "configs" / "default.yaml"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    old_cfg = load_yaml(OLD_CONFIG)
    new_cfg = load_yaml(NEW_CONFIG)

    rows = [
        ("backbone", old_cfg["model"]["backbone"], new_cfg["model"]["backbone"]),
        ("input mode", "RGB+mask 4ch stem", "RGB stem + mask gating (default), optional 4ch stem"),
        ("loss mode", old_cfg["training"]["loss_mode"], new_cfg["model"]["loss_mode"]),
        ("mask usage", "early concat + masked pooling", "stage mask gating + masked multiscale pooling"),
        ("fusion/pooling", "oracle_mcd bidirectional + masked pooling", "simple_bidirectional + masked_multiscale"),
    ]
    print(f"old_project: {OLD_PROJECT}")
    print(f"new_project: {NEW_PROJECT}")
    for key, old_value, new_value in rows:
        print(f"{key}:")
        print(f"  old: {old_value}")
        print(f"  new: {new_value}")


if __name__ == "__main__":
    main()
