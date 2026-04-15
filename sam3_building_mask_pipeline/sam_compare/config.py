"""Configuration loading for the standalone SAM3 comparison project."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .paths import CONFIGS_DIR, DEFAULT_XBD_ROOT, OUTPUTS_DIR, UPSTREAM_SAM3_DIR


try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ModuleNotFoundError(
        "PyYAML is required. Install dependencies from requirements.txt."
    ) from exc


@dataclass
class PathsConfig:
    xbd_root: Path = DEFAULT_XBD_ROOT
    sam3_repo: Path = UPSTREAM_SAM3_DIR
    checkpoint: Optional[Path] = None
    output_root: Path = OUTPUTS_DIR


@dataclass
class DataConfig:
    train_split: str = "train"
    test_split: str = "test"
    image_size: int = 1008
    use_list_files: bool = False
    enable_train_aug: bool = True
    aug_hflip: bool = True
    aug_vflip: bool = True
    aug_rot90: bool = True
    aug_brightness_contrast: bool = False


@dataclass
class ModelConfig:
    decoder_channels: int = 128
    freeze_backbone: bool = False
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    decoder_dropout: float = 0.1
    use_resaspp: bool = True
    use_attention_fusion: bool = True
    use_presence_head: bool = True
    use_boundary_head: bool = True


@dataclass
class SystemConfig:
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    val_ratio: float = 0.1
    split_seed: int = 42
    pos_weight: float = 2.0
    tversky_alpha: float = 0.35
    tversky_beta: float = 0.65
    small_fg_ratio_thr: float = 0.01
    empty_fg_boost: float = 1.35
    small_fg_boost: float = 1.25
    boundary_weight: float = 1.0
    boundary_kernel_size: int = 5
    boundary_aux_weight: float = 0.20
    presence_aux_weight: float = 0.05
    presence_empty_weight: float = 1.50
    presence_small_weight: float = 1.15
    stage1_freeze_backbone_epochs: int = 3
    backbone_lr_scale: float = 0.1
    enable_scheduler: bool = True
    scheduler_type: str = "plateau"
    scheduler_monitor: str = "val_iou"
    scheduler_mode: str = "max"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    scheduler_threshold: float = 0.001
    scheduler_min_lr_backbone: float = 1e-6
    scheduler_min_lr_decoder: float = 1e-5
    enable_early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    early_stopping_start_epoch: int = 8
    enable_weighted_sampler: bool = True
    sample_weight_empty: float = 1.50
    sample_weight_small: float = 1.25
    sample_weight_medium: float = 1.00
    sample_weight_large: float = 0.90
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    log_interval: int = 20
    resume: Optional[Path] = None


@dataclass
class EvalConfig:
    batch_size: int = 1
    threshold: float = 0.5
    save_pred_masks: bool = True
    save_visualizations: bool = True
    eval_split: str = "test"
    enable_tta: bool = False
    tta_hflip: bool = True
    tta_vflip: bool = True
    tta_rot90: bool = True
    enable_threshold_sweep: bool = False
    threshold_candidates: list[float] = field(
        default_factory=lambda: [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    )
    threshold_json: Optional[Path] = None
    postprocess_json: Optional[Path] = None
    enable_postprocess: bool = True
    min_component_area: int = 16
    max_hole_area: int = 16


@dataclass
class ExperimentConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def _coerce_optional_path(value: Any) -> Optional[Path]:
    if value in (None, "", "null"):
        return None
    return Path(value).expanduser().resolve()


def _coerce_path(value: Any) -> Path:
    return Path(value).expanduser().resolve()


def _config_from_dict(data: Mapping[str, Any]) -> ExperimentConfig:
    paths_data = dict(data.get("paths", {}))
    data_data = dict(data.get("data", {}))
    model_data = dict(data.get("model", {}))
    system_data = dict(data.get("system", {}))
    train_data = dict(data.get("train", {}))
    eval_data = dict(data.get("eval", {}))

    paths_cfg = PathsConfig(
        xbd_root=_coerce_path(paths_data.get("xbd_root", DEFAULT_XBD_ROOT)),
        sam3_repo=_coerce_path(paths_data.get("sam3_repo", UPSTREAM_SAM3_DIR)),
        checkpoint=_coerce_optional_path(paths_data.get("checkpoint")),
        output_root=_coerce_path(paths_data.get("output_root", OUTPUTS_DIR)),
    )
    train_resume = _coerce_optional_path(train_data.get("resume"))
    eval_threshold_json = _coerce_optional_path(eval_data.get("threshold_json"))
    eval_postprocess_json = _coerce_optional_path(eval_data.get("postprocess_json"))

    return ExperimentConfig(
        paths=paths_cfg,
        data=DataConfig(**data_data),
        model=ModelConfig(**model_data),
        system=SystemConfig(**system_data),
        train=TrainConfig(resume=train_resume, **{k: v for k, v in train_data.items() if k != "resume"}),
        eval=EvalConfig(
            threshold_json=eval_threshold_json,
            postprocess_json=eval_postprocess_json,
            **{
                k: v
                for k, v in eval_data.items()
                if k not in {"threshold_json", "postprocess_json"}
            },
        ),
    )


def load_experiment_config(
    config_paths: Optional[Sequence[str | Path]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> ExperimentConfig:
    """Load and merge YAML config files, then apply an optional override mapping."""
    merged: dict[str, Any] = {}
    if config_paths is None:
        config_paths = (CONFIGS_DIR / "default.yaml",)
    for raw_path in config_paths:
        path = Path(raw_path).expanduser().resolve()
        _deep_update(merged, _load_yaml_file(path))
    if overrides:
        _deep_update(merged, dict(overrides))
    return _config_from_dict(merged)
