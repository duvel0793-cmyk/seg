from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from datasets import XBDInstanceDataset
from engine import build_epoch_loss, run_epoch
from evaluate import build_dataloader, write_prediction_exports
from models import build_model
from utils.checkpoint import load_checkpoint, resolve_checkpoint_model_state
from utils.misc import (
    get_split_list,
    load_config,
    merge_config,
    set_seed,
    write_json,
    write_yaml,
)
from utils.oracle_bridge_diagnosis import (
    build_area_bin_class_report,
    build_current_model_instances_by_tile,
    build_dataset_gt_instances_by_tile,
    build_prediction_records_by_tile,
    build_target_component_oracle,
    build_tile_info_map,
    collect_full_json_diagnostics,
    diagnose_skipped_buildings,
    export_skipped_buildings_report,
    export_tile_coverage_report,
    format_summary_table,
    group_dataset_samples_by_tile,
    read_split_tile_ids,
    run_bridge_experiment,
    run_direct_map_experiment,
    summary_row_from_experiment,
    write_area_and_minor_reports,
    write_component_match_report,
    write_dilation_summary,
    write_per_tile_comparison,
    write_prediction_records_jsonl,
    write_summary_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")
    parser.add_argument("--list-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--dilation-list", type=str, default="0,1,2,3")
    parser.add_argument("--save-maps", action="store_true")
    parser.add_argument("--save-visuals", action="store_true")
    parser.add_argument("--use-official-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-dir", type=str, default=None)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-num-workers", type=int, default=None)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    return parser.parse_args()


def _parse_dilation_list(raw: str) -> list[int]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    parsed = sorted({int(item) for item in values})
    if not parsed:
        raise ValueError("dilation-list must contain at least one integer.")
    if any(value < 0 for value in parsed):
        raise ValueError("dilation-list values must be >= 0.")
    return parsed


def _choose_device(device_name: str) -> torch.device:
    requested = str(device_name).lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested via --device cuda, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cpu")


def _write_effective_split_list(path: Path, tile_ids: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(tile_ids) + "\n", encoding="utf-8")
    return path


def _prepare_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    if args.target_dir:
        config.setdefault("bridge", {})
        config["bridge"]["target_dir"] = str(Path(args.target_dir))
    config.setdefault("data", {})
    config.setdefault("eval", {})
    if args.eval_batch_size is not None:
        config["eval"]["batch_size"] = int(args.eval_batch_size)
    if args.eval_num_workers is not None:
        config["eval"]["num_workers"] = int(args.eval_num_workers)
        config["data"]["num_workers"] = int(args.eval_num_workers)
    if args.pin_memory is not None:
        config["data"]["pin_memory"] = bool(args.pin_memory)
    if args.persistent_workers is not None:
        config["data"]["persistent_workers"] = bool(args.persistent_workers)
    if args.prefetch_factor is not None:
        config["data"]["prefetch_factor"] = int(args.prefetch_factor)
    return config


def _load_dataset(
    *,
    config: dict[str, Any],
    split: str,
    effective_list_path: Path,
) -> XBDInstanceDataset:
    dataset = XBDInstanceDataset(
        config=config,
        split=split,
        list_path=effective_list_path,
        is_train=False,
    )
    return dataset


def _load_current_model_predictions(
    *,
    config: dict[str, Any],
    checkpoint_path: str,
    split: str,
    effective_list_path: Path,
    output_dir: Path,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model_config = config
    checkpoint_config = checkpoint.get("config")
    if isinstance(checkpoint_config, dict):
        model_config = merge_config(checkpoint_config)
        model_config["data"] = dict(config["data"])
        model_config["dataset"] = dict(config["dataset"])
        model_config["eval"] = dict(config["eval"])
        model_config["bridge"] = dict(config.get("bridge", {}))
        model_config["project"] = dict(config["project"])
    model_config["model"] = dict(model_config["model"])
    model_config["model"]["pretrained"] = False
    set_seed(int(model_config["training"]["seed"]))
    dataset = _load_dataset(config=model_config, split=split, effective_list_path=effective_list_path)
    loader = build_dataloader(
        dataset,
        batch_size=int(model_config["eval"]["batch_size"]),
        num_workers=int(model_config["eval"]["num_workers"]),
        seed=int(model_config["training"]["seed"]),
    )
    model = build_model(model_config).to(device)
    if bool(model_config["training"].get("channels_last", False)):
        model = model.to(memory_format=torch.channels_last)
    state_dict = resolve_checkpoint_model_state(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "checkpoint_load_warning "
            f"missing={list(incompatible.missing_keys)} unexpected={list(incompatible.unexpected_keys)}"
        )
    criterion = build_epoch_loss(model_config).to(device)
    result = run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        config=model_config,
        optimizer=None,
        scaler=None,
        epoch_index=max(int(checkpoint.get("epoch", 1)) - 1, 0),
        collect_predictions=True,
        apply_best_damage_threshold=str(model_config["eval"].get("apply_best_damage_threshold", "none")),
        max_batches=None,
    )
    write_json(output_dir / "instance_metrics.json", result["metrics"])
    if result.get("diagnostics") is not None:
        write_json(output_dir / "instance_diagnostics.json", result["diagnostics"])
    if result.get("threshold_sweep") is not None:
        write_json(output_dir / "threshold_sweep.json", result["threshold_sweep"])
    (output_dir / "classification_report_instance.txt").write_text(result["report"] + "\n", encoding="utf-8")
    write_prediction_exports(output_dir, result["prediction_records"])
    write_prediction_records_jsonl(output_dir / "prediction_records.jsonl", result["prediction_records"])
    return result["prediction_records"], result["metrics"]


def _collect_target_maps(
    tile_ids: list[str],
    tile_info_map: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    from utils.oracle_bridge_diagnosis import _load_official_target_map  # local import to keep public surface small

    gt_maps_by_tile: dict[str, Any] = {}
    for tile_id in tile_ids:
        tile_info = tile_info_map.get(str(tile_id))
        if tile_info is None:
            continue
        gt_map, _ = _load_official_target_map(tile_info, config)
        if gt_map is not None:
            gt_maps_by_tile[str(tile_id)] = gt_map
    return gt_maps_by_tile


def _append_per_tile_rows(experiment_result: dict[str, Any], accumulator: list[dict[str, Any]]) -> None:
    for row in experiment_result.get("per_tile_metrics", []):
        accumulator.append(
            {
                "experiment": row.get("experiment", experiment_result["experiment"]),
                "tile_id": row["tile_id"],
                "gt_source": row.get("gt_source", ""),
                "num_pred_instances": int(row.get("num_pred_instances", 0)),
                "num_gt_instances": int(row.get("num_gt_instances", 0)),
                "pred_building_pixels": int(row.get("pred_building_pixels", 0)),
                "gt_building_pixels": int(row.get("gt_building_pixels", 0)),
                "localization_precision": float(row.get("localization_precision", 0.0)),
                "localization_recall": float(row.get("localization_recall", 0.0)),
                "localization_f1": float(row.get("localization_f1", 0.0)),
                "damage_f1_hmean": float(row.get("damage_f1_hmean", 0.0)),
                "xview2_overall_score": float(row.get("xview2_overall_score", 0.0)),
                "localization_fp": int(row.get("localization_fp", 0)),
                "localization_fn": int(row.get("localization_fn", 0)),
            }
        )


def main() -> None:
    args = parse_args()
    config = _prepare_config(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(output_dir / "diagnosis_config.yaml", config)
    write_json(output_dir / "diagnosis_args.json", vars(args))

    list_path = Path(args.list_path) if args.list_path else Path(get_split_list(config, args.split))
    tile_ids = read_split_tile_ids(list_path, max_tiles=args.max_tiles)
    effective_list_path = _write_effective_split_list(output_dir / "_effective_split.txt", tile_ids)
    dilation_values = _parse_dilation_list(args.dilation_list)
    overlap_policy = str(config.get("bridge", {}).get("overlap_policy", "max_label"))

    print(f"Processing split={args.split} tiles={len(tile_ids)} list_path={list_path}")
    print(f"Dilation sweep values: {dilation_values}")
    dataset = _load_dataset(config=config, split=args.split, effective_list_path=effective_list_path)
    dataset_samples_by_tile = group_dataset_samples_by_tile(dataset.samples)
    dataset_instances_by_tile = build_dataset_gt_instances_by_tile(dataset_samples_by_tile)
    print(f"Indexed dataset.samples={len(dataset.samples)} across {len(dataset_samples_by_tile)} tiles")

    tile_info_map, missing_tile_path_rows = build_tile_info_map(tile_ids, config)
    full_json_instances_by_tile, full_json_reports_by_tile, full_json_stats = collect_full_json_diagnostics(tile_ids, config)
    print(
        "Full JSON valid buildings="
        f"{full_json_stats['num_full_json_valid_instances']} "
        f"invalid_wkt={full_json_stats['invalid_wkt_count']} "
        f"unknown_subtype={full_json_stats['unknown_subtype_count']}"
    )

    skipped_report = diagnose_skipped_buildings(
        tile_ids=tile_ids,
        config=config,
        dataset_samples_by_tile=dataset_samples_by_tile,
        full_json_reports_by_tile=full_json_reports_by_tile,
    )
    export_skipped_buildings_report(skipped_report, output_dir)
    export_tile_coverage_report(
        output_dir=output_dir,
        tile_ids=tile_ids,
        tile_info_map=tile_info_map,
        dataset_instances_by_tile=dataset_instances_by_tile,
        full_json_instances_by_tile=full_json_instances_by_tile,
        config=config,
    )
    print(
        "Skipped/missing valid buildings from dataset.samples="
        f"{skipped_report['summary']['num_missing_from_dataset']} "
        f"missing_ratio={skipped_report['summary']['missing_ratio']:.4f}"
    )

    summary_rows: list[dict[str, Any]] = []
    per_tile_rows: list[dict[str, Any]] = []
    area_rows_all: list[dict[str, Any]] = []
    minor_rows_all: list[dict[str, Any]] = []
    warnings: list[str] = []
    if missing_tile_path_rows:
        warnings.append(
            f"WARNING: {len(missing_tile_path_rows)} tiles could not resolve tile paths and were skipped."
        )
    if skipped_report["summary"]["num_missing_from_dataset"] > 0:
        warnings.append(
            "WARNING: dataset.samples does not cover all GT buildings; current oracle bridge is not a true full oracle pixel upper bound."
        )

    target_maps_by_tile = _collect_target_maps(tile_ids, tile_info_map, config) if args.use_official_target else {}

    if args.use_official_target:
        print("Running experiment: target_png_self_check")
        target_self_check = run_direct_map_experiment(
            experiment_name="target_png_self_check",
            output_dir=output_dir / "target_png_self_check",
            tile_ids=tile_ids,
            tile_info_map=tile_info_map,
            pred_maps_by_tile={tile_id: gt_map.copy() for tile_id, gt_map in target_maps_by_tile.items()},
            gt_maps_by_tile=target_maps_by_tile,
            save_maps=args.save_maps,
            save_visuals=args.save_visuals,
        )
        summary_rows.append(summary_row_from_experiment("target_png_self_check", target_self_check["global_metrics"]))
        _append_per_tile_rows(target_self_check, per_tile_rows)
        self_check_metrics = target_self_check["global_metrics"]
        if any(
            abs(float(self_check_metrics[key]) - 1.0) > 1.0e-8
            for key in ["localization_f1", "damage_f1_hmean", "xview2_overall_score"]
        ):
            warnings.append("CRITICAL_METRIC_BUG: target_png_self_check is not exactly 1.0.")
        print(
            "  score="
            f"{self_check_metrics['xview2_overall_score']:.4f} "
            f"loc_f1={self_check_metrics['localization_f1']:.4f} "
            f"damage_hmean={self_check_metrics['damage_f1_hmean']:.4f}"
        )
    else:
        warnings.append("WARNING: target_png_self_check was skipped because --no-use-official-target was set.")

    print("Running experiment: dataset_gt_polygon_bridge")
    dataset_gt_experiment = run_bridge_experiment(
        experiment_name="dataset_gt_polygon_bridge",
        output_dir=output_dir / "dataset_gt_polygon_bridge",
        tile_ids=tile_ids,
        tile_info_map=tile_info_map,
        pred_instances_by_tile=dataset_instances_by_tile,
        gt_instances_by_tile=dataset_instances_by_tile,
        config=config,
        overlap_policy=overlap_policy,
        use_official_target=args.use_official_target,
        save_maps=args.save_maps,
        save_visuals=args.save_visuals,
        extra_global_fields={
            "num_dataset_samples": int(len(dataset.samples)),
            "num_tiles": int(len(tile_ids)),
        },
    )
    summary_rows.append(summary_row_from_experiment("dataset_gt_polygon_bridge", dataset_gt_experiment["global_metrics"]))
    _append_per_tile_rows(dataset_gt_experiment, per_tile_rows)
    print(
        "  score="
        f"{dataset_gt_experiment['global_metrics']['xview2_overall_score']:.4f} "
        f"loc_f1={dataset_gt_experiment['global_metrics']['localization_f1']:.4f} "
        f"loc_r={dataset_gt_experiment['global_metrics']['localization_recall']:.4f}"
    )

    dataset_area_rows, dataset_minor_rows = build_area_bin_class_report(
        experiment_name="dataset_gt_polygon_bridge",
        tile_ids=tile_ids,
        tile_info_map=tile_info_map,
        instances_by_tile=dataset_instances_by_tile,
        pred_instances_by_tile=dataset_instances_by_tile,
        config=config,
        use_component_masks=True,
    )
    area_rows_all.extend(dataset_area_rows)
    minor_rows_all.extend(dataset_minor_rows)

    print("Running experiment: full_json_gt_polygon_bridge")
    full_json_gt_experiment = run_bridge_experiment(
        experiment_name="full_json_gt_polygon_bridge",
        output_dir=output_dir / "full_json_gt_polygon_bridge",
        tile_ids=tile_ids,
        tile_info_map=tile_info_map,
        pred_instances_by_tile={
            tile_id: [{**instance, "pred_label": int(instance["gt_label"])} for instance in instances]
            for tile_id, instances in full_json_instances_by_tile.items()
        },
        gt_instances_by_tile=full_json_instances_by_tile,
        config=config,
        overlap_policy=overlap_policy,
        use_official_target=args.use_official_target,
        save_maps=args.save_maps,
        save_visuals=args.save_visuals,
        extra_global_fields={
            "invalid_polygon_count": int(full_json_stats["invalid_polygon_count"]),
            "invalid_wkt_count": int(full_json_stats["invalid_wkt_count"]),
            "unknown_subtype_count": int(full_json_stats["unknown_subtype_count"]),
            "num_full_json_valid_instances": int(full_json_stats["num_full_json_valid_instances"]),
        },
    )
    summary_rows.append(summary_row_from_experiment("full_json_gt_polygon_bridge", full_json_gt_experiment["global_metrics"]))
    _append_per_tile_rows(full_json_gt_experiment, per_tile_rows)
    print(
        "  score="
        f"{full_json_gt_experiment['global_metrics']['xview2_overall_score']:.4f} "
        f"loc_f1={full_json_gt_experiment['global_metrics']['localization_f1']:.4f} "
        f"loc_r={full_json_gt_experiment['global_metrics']['localization_recall']:.4f}"
    )

    print("Running experiment: dataset polygon dilation sweep")
    dataset_dilation_rows: list[dict[str, Any]] = []
    for dilation_value in dilation_values:
        dilation_name = f"dataset_dilation_{dilation_value}"
        experiment_result = run_bridge_experiment(
            experiment_name=dilation_name,
            output_dir=output_dir / "dilation_sweep_dataset" / f"dilation_{dilation_value}",
            tile_ids=tile_ids,
            tile_info_map=tile_info_map,
            pred_instances_by_tile=dataset_instances_by_tile,
            gt_instances_by_tile=dataset_instances_by_tile,
            config=config,
            overlap_policy=overlap_policy,
            use_official_target=args.use_official_target,
            save_maps=args.save_maps,
            save_visuals=args.save_visuals,
            pred_dilation_radius=int(dilation_value),
        )
        row = summary_row_from_experiment(dilation_name, experiment_result["global_metrics"])
        row["dilation"] = int(dilation_value)
        dataset_dilation_rows.append(
            {
                "dilation": int(dilation_value),
                "localization_precision": row["localization_precision"],
                "localization_recall": row["localization_recall"],
                "localization_f1": row["localization_f1"],
                "damage_f1_hmean": row["damage_f1_hmean"],
                "xview2_overall_score": row["xview2_overall_score"],
                "minor_f1": row["minor_f1"],
                "major_f1": row["major_f1"],
                "destroyed_f1": row["destroyed_f1"],
                "no_damage_f1": row["no_damage_f1"],
                "localization_fp": row["localization_fp"],
                "localization_fn": row["localization_fn"],
            }
        )
        summary_rows.append(row)
        _append_per_tile_rows(experiment_result, per_tile_rows)
        print(
            f"  dilation={dilation_value} "
            f"score={row['xview2_overall_score']:.4f} "
            f"loc_r={row['localization_recall']:.4f}"
        )
    write_dilation_summary(output_dir / "dilation_sweep_dataset", dataset_dilation_rows)

    print("Running experiment: full JSON polygon dilation sweep")
    full_json_dilation_rows: list[dict[str, Any]] = []
    for dilation_value in dilation_values:
        dilation_name = f"full_json_dilation_{dilation_value}"
        experiment_result = run_bridge_experiment(
            experiment_name=dilation_name,
            output_dir=output_dir / "dilation_sweep_full_json" / f"dilation_{dilation_value}",
            tile_ids=tile_ids,
            tile_info_map=tile_info_map,
            pred_instances_by_tile={
                tile_id: [
                    {**instance, "pred_label": int(instance["gt_label"])}
                    for instance in instances
                ]
                for tile_id, instances in full_json_instances_by_tile.items()
            },
            gt_instances_by_tile=full_json_instances_by_tile,
            config=config,
            overlap_policy=overlap_policy,
            use_official_target=args.use_official_target,
            save_maps=args.save_maps,
            save_visuals=args.save_visuals,
            pred_dilation_radius=int(dilation_value),
        )
        row = summary_row_from_experiment(dilation_name, experiment_result["global_metrics"])
        row["dilation"] = int(dilation_value)
        full_json_dilation_rows.append(
            {
                "dilation": int(dilation_value),
                "localization_precision": row["localization_precision"],
                "localization_recall": row["localization_recall"],
                "localization_f1": row["localization_f1"],
                "damage_f1_hmean": row["damage_f1_hmean"],
                "xview2_overall_score": row["xview2_overall_score"],
                "minor_f1": row["minor_f1"],
                "major_f1": row["major_f1"],
                "destroyed_f1": row["destroyed_f1"],
                "no_damage_f1": row["no_damage_f1"],
                "localization_fp": row["localization_fp"],
                "localization_fn": row["localization_fn"],
            }
        )
        summary_rows.append(row)
        _append_per_tile_rows(experiment_result, per_tile_rows)
        print(
            f"  dilation={dilation_value} "
            f"score={row['xview2_overall_score']:.4f} "
            f"loc_r={row['localization_recall']:.4f}"
        )
    write_dilation_summary(output_dir / "dilation_sweep_full_json", full_json_dilation_rows)

    if args.use_official_target:
        print("Running experiment: target_component_oracle_bridge")
        component_pred_by_tile, component_report_rows, component_metrics_extra = build_target_component_oracle(
            dataset_instances_by_tile=dataset_instances_by_tile,
            tile_info_map=tile_info_map,
            config=config,
        )
        component_experiment = run_bridge_experiment(
            experiment_name="target_component_oracle_bridge",
            output_dir=output_dir / "target_component_oracle_bridge",
            tile_ids=tile_ids,
            tile_info_map=tile_info_map,
            pred_instances_by_tile=component_pred_by_tile,
            gt_instances_by_tile=dataset_instances_by_tile,
            config=config,
            overlap_policy=overlap_policy,
            use_official_target=True,
            save_maps=args.save_maps,
            save_visuals=args.save_visuals,
            extra_global_fields=component_metrics_extra,
        )
        write_component_match_report(output_dir / "target_component_oracle_bridge", component_report_rows)
        summary_rows.append(summary_row_from_experiment("target_component_oracle_bridge", component_experiment["global_metrics"]))
        _append_per_tile_rows(component_experiment, per_tile_rows)
        print(
            "  score="
            f"{component_experiment['global_metrics']['xview2_overall_score']:.4f} "
            f"loc_f1={component_experiment['global_metrics']['localization_f1']:.4f}"
        )
        del component_pred_by_tile
        del component_report_rows
        del component_experiment
        gc.collect()
    else:
        warnings.append("WARNING: target_component_oracle_bridge was skipped because --no-use-official-target was set.")

    if args.checkpoint:
        print("Running experiment: current_model_prediction_bridge")
        try:
            device = _choose_device(args.device)
            current_model_output_dir = output_dir / "current_model_prediction_bridge"
            prediction_records, instance_metrics = _load_current_model_predictions(
                config=config,
                checkpoint_path=args.checkpoint,
                split=args.split,
                effective_list_path=effective_list_path,
                output_dir=current_model_output_dir,
                device=device,
            )
            prediction_records_by_tile = build_prediction_records_by_tile(prediction_records)
            current_model_instances_by_tile = build_current_model_instances_by_tile(
                dataset_samples_by_tile=dataset_samples_by_tile,
                prediction_records_by_tile=prediction_records_by_tile,
            )
            current_model_experiment = run_bridge_experiment(
                experiment_name="current_model_prediction_bridge",
                output_dir=current_model_output_dir,
                tile_ids=tile_ids,
                tile_info_map=tile_info_map,
                pred_instances_by_tile=current_model_instances_by_tile,
                gt_instances_by_tile=dataset_instances_by_tile,
                config=config,
                overlap_policy=overlap_policy,
                use_official_target=args.use_official_target,
                save_maps=args.save_maps,
                save_visuals=args.save_visuals,
                extra_global_fields={"instance_metrics": instance_metrics},
            )
            summary_rows.append(summary_row_from_experiment("current_model_prediction_bridge", current_model_experiment["global_metrics"]))
            _append_per_tile_rows(current_model_experiment, per_tile_rows)
            current_area_rows, current_minor_rows = build_area_bin_class_report(
                experiment_name="current_model_prediction_bridge",
                tile_ids=tile_ids,
                tile_info_map=tile_info_map,
                instances_by_tile=dataset_instances_by_tile,
                pred_instances_by_tile=current_model_instances_by_tile,
                config=config,
                use_component_masks=True,
            )
            area_rows_all.extend(current_area_rows)
            minor_rows_all.extend(current_minor_rows)
            print(
                "  score="
                f"{current_model_experiment['global_metrics']['xview2_overall_score']:.4f} "
                f"loc_f1={current_model_experiment['global_metrics']['localization_f1']:.4f} "
                f"minor_f1={summary_rows[-1]['minor_f1']:.4f}"
            )
            del prediction_records
            del prediction_records_by_tile
            del current_model_instances_by_tile
            del current_model_experiment
            gc.collect()
        except Exception as exc:
            warnings.append(f"WARNING: current_model_prediction_bridge failed: {exc}")
            print(f"  failed: {exc}")

    write_area_and_minor_reports(output_dir, area_rows_all, minor_rows_all)
    write_summary_table(output_dir, summary_rows, warnings)
    write_per_tile_comparison(output_dir, per_tile_rows)

    summary_text = format_summary_table(summary_rows)
    print()
    print(summary_text)
    if warnings:
        print()
        print("Warnings:")
        for item in warnings:
            print(f"- {item}")


if __name__ == "__main__":
    main()
