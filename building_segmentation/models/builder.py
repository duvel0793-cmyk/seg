from copy import deepcopy
from pathlib import Path

import yaml

from building_segmentation.models.network import FlowMambaBuilding


DEFAULT_MODEL_CONFIG = {
    "MODEL": {
        "DROP_PATH_RATE": 0.1,
        "NUM_CLASSES": 1000,
        "VSSM": {
            "PATCH_SIZE": 4,
            "IN_CHANS": 3,
            "DEPTHS": [2, 2, 9, 2],
            "EMBED_DIM": 96,
            "SSM_D_STATE": 16,
            "SSM_RATIO": 2.0,
            "SSM_RANK_RATIO": 2.0,
            "SSM_DT_RANK": "auto",
            "SSM_ACT_LAYER": "silu",
            "SSM_CONV": 3,
            "SSM_CONV_BIAS": True,
            "SSM_DROP_RATE": 0.0,
            "SSM_INIT": "v0",
            "SSM_FORWARDTYPE": "v2",
            "MLP_RATIO": 4.0,
            "MLP_ACT_LAYER": "gelu",
            "MLP_DROP_RATE": 0.0,
            "PATCH_NORM": True,
            "NORM_LAYER": "ln",
            "DOWNSAMPLE": "v2",
            "PATCHEMBED": "v2",
            "GMLP": False,
        },
    },
    "TRAIN": {
        "USE_CHECKPOINT": False,
    },
}


def deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_model_config(cfg_path):
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    if cfg_path is not None:
        with open(cfg_path, "r", encoding="utf-8") as handle:
            user_config = yaml.safe_load(handle) or {}
        deep_update(config, user_config)
    return config


def get_model_kwargs(cfg_path):
    config = load_model_config(cfg_path)
    vssm = config["MODEL"]["VSSM"]
    return {
        "patch_size": vssm["PATCH_SIZE"],
        "in_chans": vssm["IN_CHANS"],
        "num_classes": config["MODEL"]["NUM_CLASSES"],
        "depths": vssm["DEPTHS"],
        "dims": vssm["EMBED_DIM"],
        "ssm_d_state": vssm["SSM_D_STATE"],
        "ssm_ratio": vssm["SSM_RATIO"],
        "ssm_rank_ratio": vssm["SSM_RANK_RATIO"],
        "ssm_dt_rank": "auto" if vssm["SSM_DT_RANK"] == "auto" else int(vssm["SSM_DT_RANK"]),
        "ssm_act_layer": vssm["SSM_ACT_LAYER"],
        "ssm_conv": vssm["SSM_CONV"],
        "ssm_conv_bias": vssm["SSM_CONV_BIAS"],
        "ssm_drop_rate": vssm["SSM_DROP_RATE"],
        "ssm_init": vssm["SSM_INIT"],
        "forward_type": vssm["SSM_FORWARDTYPE"],
        "mlp_ratio": vssm["MLP_RATIO"],
        "mlp_act_layer": vssm["MLP_ACT_LAYER"],
        "mlp_drop_rate": vssm["MLP_DROP_RATE"],
        "drop_path_rate": config["MODEL"]["DROP_PATH_RATE"],
        "patch_norm": vssm["PATCH_NORM"],
        "norm_layer": vssm["NORM_LAYER"],
        "downsample_version": vssm["DOWNSAMPLE"],
        "patchembed_version": vssm["PATCHEMBED"],
        "gmlp": vssm["GMLP"],
        "use_checkpoint": config["TRAIN"]["USE_CHECKPOINT"],
        "use_ossm": False,
    }


def build_model(cfg_path, pretrained_weight_path, num_classes=2):
    cfg_path = Path(cfg_path) if cfg_path is not None else None
    return FlowMambaBuilding(
        pretrained=pretrained_weight_path,
        output_classes=num_classes,
        **get_model_kwargs(cfg_path),
    )
