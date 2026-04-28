from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    phase_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Export one flat CSV row per run from a runs tree containing "
            "summary.json/config.json/history.csv/step_history.csv."
        )
    )
    parser.add_argument(
        "--runs-root",
        default=str(phase_root / "runs_refined"),
        help="Root directory containing run subdirectories.",
    )
    parser.add_argument(
        "--output-path",
        default=str(phase_root / "analysis_outputs" / "all_runs_export.csv"),
        help="Path for the exported CSV.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        row_count = sum(1 for _ in handle)
    return max(row_count - 1, 0)


def normalize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, sort_keys=True)


def flatten_mapping(
    data: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in sorted(data.items()):
        column = f"{prefix}{key}"
        if isinstance(value, dict):
            flattened.update(flatten_mapping(value, f"{column}__"))
        else:
            flattened[column] = normalize_value(value)
    return flattened


def infer_total_cores(summary: dict[str, Any], config: dict[str, Any], run_name: str) -> int | None:
    summary_shape = summary.get("ttlora_shape")
    if isinstance(summary_shape, list) and summary_shape:
        return len(summary_shape)

    model_shape = config.get("model", {}).get("ttlora_shape")
    if isinstance(model_shape, list) and model_shape:
        return len(model_shape)

    weight_configs = summary.get("ttlora_weight_configs")
    if isinstance(weight_configs, list) and weight_configs:
        tt_shape = weight_configs[0].get("tt_shape")
        if isinstance(tt_shape, list) and tt_shape:
            return len(tt_shape)

    match = re.search(r"cores(\d+)", run_name)
    return int(match.group(1)) if match else None


def infer_task_type(summary: dict[str, Any], config: dict[str, Any]) -> str:
    task_type = summary.get("task_type")
    if task_type:
        return str(task_type)

    model_name = str(summary.get("model_name_or_path") or config.get("model", {}).get("model_name_or_path") or "").lower()
    if "gpt2" in model_name:
        return "generation"
    return "classification"


def infer_primary_metric(summary: dict[str, Any], task_type: str) -> tuple[str | None, Any, str | None]:
    if task_type == "generation":
        if "best_validation_perplexity" in summary:
            return (
                "best_validation_perplexity",
                summary.get("best_validation_perplexity"),
                "lower_is_better",
            )
        if "best_validation_loss" in summary:
            return (
                "best_validation_loss",
                summary.get("best_validation_loss"),
                "lower_is_better",
            )
    else:
        if "best_validation_accuracy" in summary:
            return (
                "best_validation_accuracy",
                summary.get("best_validation_accuracy"),
                "higher_is_better",
            )
        if "best_validation_token_accuracy" in summary:
            return (
                "best_validation_token_accuracy",
                summary.get("best_validation_token_accuracy"),
                "higher_is_better",
            )
    return (None, None, None)


def build_row(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"
    history_path = run_dir / "history.csv"
    step_history_path = run_dir / "step_history.csv"

    summary = read_json(summary_path) if summary_path.exists() else {}
    config = read_json(config_path) if config_path.exists() else {}

    run_name = run_dir.name
    parent_group = run_dir.parent.name
    dataset_name = (
        summary.get("dataset_name")
        or config.get("data", {}).get("dataset_name")
        or parent_group.split("_")[0]
    )
    ttlora_variant = (
        summary.get("ttlora_variant")
        or config.get("model", {}).get("ttlora_variant")
        or parent_group.split("_", 1)[-1]
    )
    task_type = infer_task_type(summary, config)
    total_cores = infer_total_cores(summary, config, run_name)
    primary_metric_name, primary_metric_value, primary_metric_goal = infer_primary_metric(summary, task_type)

    row: dict[str, Any] = {
        "run_name": run_name,
        "dataset_name": normalize_value(dataset_name),
        "ttlora_variant": normalize_value(ttlora_variant),
        "task_type": task_type,
        "total_cores": total_cores,
        "learning_rate": summary.get("learning_rate", config.get("training", {}).get("learning_rate")),
        "ttlora_rank": summary.get("ttlora_rank", config.get("model", {}).get("ttlora_rank")),
        "ttlora_alpha": summary.get("ttlora_alpha", config.get("model", {}).get("ttlora_alpha")),
        "trainable_parameters": summary.get("trainable_parameters"),
        "total_parameters": summary.get("total_parameters"),
        "frozen_parameters": summary.get("frozen_parameters"),
        "best_epoch": summary.get("best_epoch"),
        "epochs_ran": summary.get("epochs_ran"),
        "seed": summary.get("seed", config.get("training", {}).get("seed")),
        "phase": summary.get("phase", config.get("metadata", {}).get("phase")),
        "parent_group": parent_group,
        "run_dir": str(run_dir.resolve()),
        "summary_path": str(summary_path.resolve()),
        "config_path": str(config_path.resolve()) if config_path.exists() else None,
        "history_path": str(history_path.resolve()) if history_path.exists() else None,
        "step_history_path": str(step_history_path.resolve()) if step_history_path.exists() else None,
        "history_rows": csv_row_count(history_path),
        "step_history_rows": csv_row_count(step_history_path),
        "primary_metric_name": primary_metric_name,
        "primary_metric_value": primary_metric_value,
        "primary_metric_goal": primary_metric_goal,
    }

    row.update(flatten_mapping(summary, "summary__"))
    if config:
        row.update(flatten_mapping(config, "config__"))

    return row


def collect_rows(runs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(runs_root.rglob("summary.json")):
        rows.append(build_row(summary_path.parent))
    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: normalize_value(row.get(key)) for key in fieldnames})


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root does not exist: {runs_root}")

    rows = collect_rows(runs_root)
    if not rows:
        raise RuntimeError(f"No runs with summary.json found under {runs_root}")

    write_csv(rows, output_path)
    print(f"Wrote {len(rows)} runs to {output_path}")


if __name__ == "__main__":
    main()
