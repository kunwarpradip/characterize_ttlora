from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return values
    averaged = []
    running_sum = 0.0
    for idx, value in enumerate(values):
        running_sum += value
        if idx >= window:
            running_sum -= values[idx - window]
        averaged.append(running_sum / min(idx + 1, window))
    return averaged


def derive_label(summary: dict[str, Any], run_dir: Path) -> str:
    method = summary.get("adaptation_method", "run")
    variant = summary.get("ttlora_variant")
    if method == "ttlora" and variant:
        return f"{method}-{variant}"
    return f"{method}-{run_dir.name}"


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: parse_value(value) for key, value in row.items()} for row in reader]


@dataclass
class RunArtifacts:
    run_dir: Path
    label: str
    summary: dict[str, Any]
    history: list[dict[str, Any]]
    step_history: list[dict[str, Any]]


def load_run_artifacts(run_dir: Path, label_override: str | None = None) -> RunArtifacts:
    summary_path = run_dir / "summary.json"
    history_path = run_dir / "history.csv"
    step_history_path = run_dir / "step_history.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in {run_dir}")
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history.csv in {run_dir}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    history = read_csv_records(history_path)
    step_history = read_csv_records(step_history_path) if step_history_path.exists() else []
    label = label_override or derive_label(summary, run_dir)

    return RunArtifacts(
        run_dir=run_dir,
        label=label,
        summary=summary,
        history=history,
        step_history=step_history,
    )


def plot_epoch_curves(runs: list[RunArtifacts], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epoch_metrics = [
        ("train_loss", "Train Loss"),
        ("validation_loss", "Validation Loss"),
        ("validation_accuracy", "Validation Accuracy"),
        ("learning_rate", "Learning Rate"),
    ]

    for ax, (metric, title) in zip(axes.flatten(), epoch_metrics):
        for run in runs:
            epochs = [row["epoch"] for row in run.history]
            values = [row[metric] for row in run.history]
            ax.plot(epochs, values, marker="o", linewidth=2, label=run.label)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "epoch_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_epoch_diagnostics(runs: list[RunArtifacts], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [
        ("avg_grad_norm", "Average Grad Norm"),
        ("max_grad_norm", "Max Grad Norm"),
        ("clipped_step_fraction", "Clipped Step Fraction"),
        ("peak_memory_gb", "Peak Memory (GB)"),
    ]

    for ax, (metric, title) in zip(axes.flatten(), metrics):
        for run in runs:
            epochs = [row["epoch"] for row in run.history]
            values = [row[metric] for row in run.history]
            ax.plot(epochs, values, marker="o", linewidth=2, label=run.label)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "epoch_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_step_curves(runs: list[RunArtifacts], output_dir: Path, smoothing_window: int) -> None:
    runs_with_steps = [run for run in runs if run.step_history]
    if not runs_with_steps:
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    metrics = [
        ("train_loss", "Step Train Loss"),
        ("grad_norm_pre_clip", "Step Grad Norm (Pre-Clip)"),
        ("clipping_triggered", "Rolling Clip Rate"),
    ]

    for ax, (metric, title) in zip(axes, metrics):
        for run in runs_with_steps:
            steps = [row["optimizer_step"] for row in run.step_history]
            raw_values = [float(row[metric]) for row in run.step_history]
            smoothed_values = moving_average(raw_values, smoothing_window)
            ax.plot(steps, smoothed_values, linewidth=1.8, label=run.label)
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Optimizer Step")
    fig.tight_layout()
    fig.savefig(output_dir / "step_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary_table(runs: list[RunArtifacts], output_dir: Path) -> None:
    rows = []
    for run in runs:
        rows.append(
            {
                "label": run.label,
                "adaptation_method": run.summary.get("adaptation_method"),
                "ttlora_variant": run.summary.get("ttlora_variant"),
                "best_epoch": run.summary.get("best_epoch"),
                "best_validation_accuracy": run.summary.get("best_validation_accuracy"),
                "final_validation_accuracy": run.summary.get("final_validation_accuracy"),
                "final_validation_loss": run.summary.get("final_validation_loss"),
                "max_peak_memory_gb": run.summary.get("max_peak_memory_gb"),
                "avg_grad_norm": run.summary.get("avg_grad_norm"),
                "max_grad_norm": run.summary.get("max_grad_norm"),
                "avg_clipped_step_fraction": run.summary.get("avg_clipped_step_fraction"),
                "total_clipped_steps": run.summary.get("total_clipped_steps"),
                "avg_epoch_seconds": run.summary.get("avg_epoch_seconds"),
                "trainable_parameters": run.summary.get("trainable_parameters"),
                "resolved_run_dir": run.summary.get("resolved_run_dir"),
            }
        )

    path = output_dir / "comparison_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary_bars(runs: list[RunArtifacts], output_dir: Path) -> None:
    labels = [run.label for run in runs]
    metrics = [
        ("best_validation_accuracy", "Best Val Accuracy"),
        ("final_validation_accuracy", "Final Val Accuracy"),
        ("max_peak_memory_gb", "Peak Memory (GB)"),
        ("avg_clipped_step_fraction", "Avg Clipped Step Fraction"),
        ("total_clipped_steps", "Total Clipped Steps"),
        ("avg_epoch_seconds", "Avg Epoch Seconds"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (metric, title) in zip(axes.flatten(), metrics):
        values = [run.summary.get(metric, 0.0) for run in runs]
        ax.bar(labels, values)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "summary_bars.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare training runs and plot diagnostics.")
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="Run directories containing summary.json/history.csv/step_history.csv",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels matching --run-dirs order.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/pkunwar/characterize_ttlora/plots/compare_runs",
        help="Directory to save plots and comparison tables.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=20,
        help="Moving-average window for step-level plots.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_dirs = [Path(path).expanduser().resolve() for path in args.run_dirs]
    labels = args.labels or []
    if labels and len(labels) != len(run_dirs):
        raise ValueError("--labels must match the number of --run-dirs.")

    runs = []
    for idx, run_dir in enumerate(run_dirs):
        label = labels[idx] if labels else None
        runs.append(load_run_artifacts(run_dir, label))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_epoch_curves(runs, output_dir)
    plot_epoch_diagnostics(runs, output_dir)
    plot_step_curves(runs, output_dir, smoothing_window=args.smoothing_window)
    plot_summary_bars(runs, output_dir)
    write_summary_table(runs, output_dir)

    print(f"Saved comparison plots to: {output_dir}")


if __name__ == "__main__":
    main()
