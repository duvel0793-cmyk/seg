from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.misc import read_json, read_yaml, write_csv_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def warn(message: str) -> None:
    print(f"warning: {message}")


def best_history_row(history_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not history_rows:
        return None
    metric_key = "ema_macro_f1" if "ema_macro_f1" in history_rows[0] else "ema_val_macro_f1"
    return max(history_rows, key=lambda row: float(row.get(metric_key, float("-inf"))))


def flatten_gate_value(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows: list[dict[str, Any]] = []
    for config_path in sorted(root.rglob("run_config.yaml")):
        run_dir = config_path.parent
        config = read_yaml(config_path)
        history_path = run_dir / "logs" / "history.jsonl"
        diagnostics_best_path = run_dir / "diagnostics_best.json"
        optimizer_groups_path = run_dir / "optimizer_groups.json"

        history_rows = []
        if history_path.exists():
            history_rows = read_jsonl(history_path)
        else:
            warn(f"missing history.jsonl for {run_dir}")
        diagnostics_best = {}
        if diagnostics_best_path.exists():
            diagnostics_best = read_json(diagnostics_best_path)
        else:
            warn(f"missing diagnostics_best.json for {run_dir}")
        optimizer_groups = None
        if not optimizer_groups_path.exists():
            warn(f"missing optimizer_groups.json for {run_dir}")
        else:
            optimizer_groups = read_json(optimizer_groups_path)

        if not history_rows or not diagnostics_best or optimizer_groups is None:
            warn(f"skipping incomplete experiment directory: {run_dir}")
            continue

        best_row = best_history_row(history_rows) or {}
        per_class = diagnostics_best.get("per_class", {})
        project_cfg = config.get("project", {})
        model_cfg = config.get("model", {})
        loss_cfg = config.get("loss", {})

        rows.append(
            {
                "experiment_name": str(project_cfg.get("name", run_dir.name)),
                "best_epoch": best_row.get("epoch", ""),
                "best_metric": best_row.get("ema_macro_f1", best_row.get("ema_val_macro_f1", "")),
                "ema_macro_f1": best_row.get("ema_macro_f1", best_row.get("ema_val_macro_f1", diagnostics_best.get("macro_f1", ""))),
                "ema_qwk": best_row.get("ema_qwk", best_row.get("ema_val_qwk", diagnostics_best.get("qwk", ""))),
                "far_error": best_row.get("ema_far_error", diagnostics_best.get("far_error", "")),
                "accuracy": diagnostics_best.get("accuracy", ""),
                "no_f1": best_row.get("no_f1", per_class.get("no-damage", {}).get("f1", "")),
                "minor_f1": best_row.get("minor_f1", per_class.get("minor-damage", {}).get("f1", "")),
                "major_f1": best_row.get("major_f1", per_class.get("major-damage", {}).get("f1", "")),
                "destroyed_f1": best_row.get("destroyed_f1", per_class.get("destroyed", {}).get("f1", "")),
                "binary_damage_f1": best_row.get("binary_damage_f1", diagnostics_best.get("binary_damage_f1", "")),
                "damaged_severity_macro_f1": best_row.get(
                    "damaged_severity_macro_f1",
                    diagnostics_best.get("damaged_severity_macro_f1", ""),
                ),
                "gt_minor_pred_no_rate": best_row.get("gt_minor_pred_no_rate", diagnostics_best.get("gt_minor_pred_no_rate", "")),
                "gt_minor_pred_major_rate": best_row.get("gt_minor_pred_major_rate", diagnostics_best.get("gt_minor_pred_major_rate", "")),
                "gt_major_pred_minor_rate": best_row.get("gt_major_pred_minor_rate", diagnostics_best.get("gt_major_pred_minor_rate", "")),
                "final_ce_weight": loss_cfg.get("final_ce_weight", ""),
                "minor_no_aux_weight": loss_cfg.get("minor_no_aux_weight", ""),
                "minor_major_aux_weight": loss_cfg.get("minor_major_aux_weight", ""),
                "structural_binary_weight": loss_cfg.get("structural_binary_weight", ""),
                "low_stage_weight": loss_cfg.get("low_stage_weight", ""),
                "high_stage_weight": loss_cfg.get("high_stage_weight", ""),
                "scale_router_ce_weight": loss_cfg.get("scale_router_ce_weight", ""),
                "context_token_count": model_cfg.get("context_token_count", ""),
                "local_attention_layers_context": model_cfg.get("local_attention_layers_context", model_cfg.get("local_attention_layers", "")),
                "cross_scale_layers": model_cfg.get("cross_scale_layers", ""),
                "scale_router_gates": flatten_gate_value(best_row.get("scale_router_gates", diagnostics_best.get("scale_router_gates", ""))),
            }
        )

    output_path = root / "experiment_summary.csv"
    write_csv_rows(
        output_path,
        [
            "experiment_name",
            "best_epoch",
            "best_metric",
            "ema_macro_f1",
            "ema_qwk",
            "far_error",
            "accuracy",
            "no_f1",
            "minor_f1",
            "major_f1",
            "destroyed_f1",
            "binary_damage_f1",
            "damaged_severity_macro_f1",
            "gt_minor_pred_no_rate",
            "gt_minor_pred_major_rate",
            "gt_major_pred_minor_rate",
            "final_ce_weight",
            "minor_no_aux_weight",
            "minor_major_aux_weight",
            "structural_binary_weight",
            "low_stage_weight",
            "high_stage_weight",
            "scale_router_ce_weight",
            "context_token_count",
            "local_attention_layers_context",
            "cross_scale_layers",
            "scale_router_gates",
        ],
        rows,
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
