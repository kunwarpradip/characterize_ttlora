from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from suite_utils import RunSummaryBundle, load_run_bundle, save_json, write_csv


def safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def safe_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def collect_run_dirs(runs_root: Path, suite_name: str) -> list[Path]:
    prefix = f"suite_{suite_name}_"
    return sorted(
        path for path in runs_root.iterdir() if path.is_dir() and path.name.startswith(prefix) and (path / "summary.json").exists()
    )


def method_label(bundle: RunSummaryBundle) -> str:
    method = bundle.summary.get("adaptation_method", "run")
    variant = bundle.summary.get("ttlora_variant")
    if method == "ttlora" and variant:
        return f"{method}-{variant}"
    return method


def flattened_summary_row(bundle: RunSummaryBundle) -> dict[str, Any]:
    model_cfg = bundle.config.get("model", {})
    train_cfg = bundle.config.get("training", {})
    return {
        "run_dir": str(bundle.run_dir),
        "dataset_name": bundle.summary.get("dataset_name"),
        "method_label": method_label(bundle),
        "adaptation_method": bundle.summary.get("adaptation_method"),
        "ttlora_variant": bundle.summary.get("ttlora_variant"),
        "seed": bundle.summary.get("seed"),
        "learning_rate": train_cfg.get("learning_rate"),
        "lr_scheduler": bundle.summary.get("lr_scheduler"),
        "best_epoch": bundle.summary.get("best_epoch"),
        "best_validation_accuracy": bundle.summary.get("best_validation_accuracy"),
        "final_validation_accuracy": bundle.summary.get("final_validation_accuracy"),
        "final_validation_loss": bundle.summary.get("final_validation_loss"),
        "max_peak_memory_gb": bundle.summary.get("max_peak_memory_gb"),
        "avg_epoch_seconds": bundle.summary.get("avg_epoch_seconds"),
        "avg_grad_norm": bundle.summary.get("avg_grad_norm"),
        "max_grad_norm": bundle.summary.get("max_grad_norm"),
        "avg_clipped_step_fraction": bundle.summary.get("avg_clipped_step_fraction"),
        "total_clipped_steps": bundle.summary.get("total_clipped_steps"),
        "trainable_parameters": bundle.summary.get("trainable_parameters"),
        "target_modules": ",".join(model_cfg.get("target_modules", [])),
        "ttlora_rank": model_cfg.get("ttlora_rank"),
        "ttlora_alpha": model_cfg.get("ttlora_alpha"),
        "ttlora_shape": "x".join(str(x) for x in model_cfg.get("ttlora_shape", [])),
        "lora_rank": model_cfg.get("lora_rank"),
        "lora_alpha": model_cfg.get("lora_alpha"),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset_name"], row["method_label"], row["learning_rate"])].append(row)

    aggregated = []
    for (dataset_name, method, learning_rate), group in sorted(grouped.items()):
        aggregated.append(
            {
                "dataset_name": dataset_name,
                "method_label": method,
                "learning_rate": learning_rate,
                "num_seeds": len(group),
                "mean_best_validation_accuracy": safe_mean([row["best_validation_accuracy"] for row in group]),
                "std_best_validation_accuracy": safe_std([row["best_validation_accuracy"] for row in group]),
                "mean_final_validation_accuracy": safe_mean([row["final_validation_accuracy"] for row in group]),
                "mean_final_validation_loss": safe_mean([row["final_validation_loss"] for row in group]),
                "mean_peak_memory_gb": safe_mean([row["max_peak_memory_gb"] for row in group]),
                "mean_avg_epoch_seconds": safe_mean([row["avg_epoch_seconds"] for row in group]),
                "mean_avg_grad_norm": safe_mean([row["avg_grad_norm"] for row in group]),
                "mean_max_grad_norm": safe_mean([row["max_grad_norm"] for row in group]),
                "mean_clipped_step_fraction": safe_mean([row["avg_clipped_step_fraction"] for row in group]),
                "mean_total_clipped_steps": safe_mean([row["total_clipped_steps"] for row in group]),
                "trainable_parameters": group[0]["trainable_parameters"],
            }
        )
    return aggregated


def plot_accuracy_vs_lr(rows: list[dict[str, Any]], output_dir: Path) -> None:
    datasets = sorted({row["dataset_name"] for row in rows})
    for dataset_name in datasets:
        subset = [row for row in rows if row["dataset_name"] == dataset_name]
        methods = sorted({row["method_label"] for row in subset})
        fig, ax = plt.subplots(figsize=(10, 6))
        for method in methods:
            method_rows = sorted((row for row in subset if row["method_label"] == method), key=lambda row: row["learning_rate"])
            x = [row["learning_rate"] for row in method_rows]
            y = [row["mean_best_validation_accuracy"] for row in method_rows]
            yerr = [row["std_best_validation_accuracy"] or 0.0 for row in method_rows]
            ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=2, capsize=4, label=method)
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Mean Best Validation Accuracy")
        ax.set_title(f"Learning Rate Sweep: {dataset_name}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{dataset_name}_accuracy_vs_lr.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_memory_vs_accuracy(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    for row in rows:
        ax.scatter(row["mean_peak_memory_gb"], row["mean_best_validation_accuracy"], s=80)
        ax.annotate(
            f"{row['dataset_name']}\n{row['method_label']}\nlr={row['learning_rate']:.0e}",
            (row["mean_peak_memory_gb"], row["mean_best_validation_accuracy"]),
            fontsize=8,
            alpha=0.8,
        )
    ax.set_xlabel("Mean Peak Memory (GB)")
    ax.set_ylabel("Mean Best Validation Accuracy")
    ax.set_title("Memory vs Accuracy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "memory_vs_accuracy.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_stability_vs_accuracy(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    for row in rows:
        ax.scatter(row["mean_clipped_step_fraction"], row["mean_best_validation_accuracy"], s=80)
        ax.annotate(
            f"{row['dataset_name']}\n{row['method_label']}\nlr={row['learning_rate']:.0e}",
            (row["mean_clipped_step_fraction"], row["mean_best_validation_accuracy"]),
            fontsize=8,
            alpha=0.8,
        )
    ax.set_xlabel("Mean Clipped Step Fraction")
    ax.set_ylabel("Mean Best Validation Accuracy")
    ax.set_title("Stability vs Accuracy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "stability_vs_accuracy.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_report(aggregated_rows: list[dict[str, Any]], output_dir: Path) -> str:
    datasets = sorted({row["dataset_name"] for row in aggregated_rows})
    lines = ["# Experiment Suite Report", ""]

    for dataset_name in datasets:
        subset = [row for row in aggregated_rows if row["dataset_name"] == dataset_name]
        best_by_accuracy = max(subset, key=lambda row: row["mean_best_validation_accuracy"])
        best_by_stability = min(subset, key=lambda row: row["mean_clipped_step_fraction"])
        best_by_memory = min(subset, key=lambda row: row["mean_peak_memory_gb"])

        lines.append(f"## {dataset_name}")
        lines.append(
            f"- Best average validation accuracy: `{best_by_accuracy['method_label']}` at "
            f"`lr={best_by_accuracy['learning_rate']:.0e}` with "
            f"`{best_by_accuracy['mean_best_validation_accuracy']:.4f}`."
        )
        lines.append(
            f"- Most stable by clipping fraction: `{best_by_stability['method_label']}` at "
            f"`lr={best_by_stability['learning_rate']:.0e}` with "
            f"`{best_by_stability['mean_clipped_step_fraction']:.4f}`."
        )
        lines.append(
            f"- Lowest memory: `{best_by_memory['method_label']}` at "
            f"`lr={best_by_memory['learning_rate']:.0e}` with "
            f"`{best_by_memory['mean_peak_memory_gb']:.4f} GB`."
        )

        contraction_rows = [row for row in subset if row["method_label"] == "ttlora-contraction"]
        reconstruction_rows = [row for row in subset if row["method_label"] == "ttlora-reconstruction"]
        if contraction_rows and reconstruction_rows:
            best_contraction = max(contraction_rows, key=lambda row: row["mean_best_validation_accuracy"])
            best_reconstruction = max(reconstruction_rows, key=lambda row: row["mean_best_validation_accuracy"])
            delta_acc = best_contraction["mean_best_validation_accuracy"] - best_reconstruction["mean_best_validation_accuracy"]
            delta_clip = best_contraction["mean_clipped_step_fraction"] - best_reconstruction["mean_clipped_step_fraction"]
            lines.append(
                f"- TT-LoRA comparison: best contraction run achieved "
                f"`{best_contraction['mean_best_validation_accuracy']:.4f}` vs "
                f"`{best_reconstruction['mean_best_validation_accuracy']:.4f}` for reconstruction "
                f"(accuracy delta `{delta_acc:+.4f}`)."
            )
            lines.append(
                f"- TT-LoRA clipping comparison: contraction `{best_contraction['mean_clipped_step_fraction']:.4f}` "
                f"vs reconstruction `{best_reconstruction['mean_clipped_step_fraction']:.4f}` "
                f"(delta `{delta_clip:+.4f}`, negative favors contraction)."
            )
        lines.append("")

    report = "\n".join(lines)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate and analyze experiment suite results.")
    parser.add_argument("--suite-name", required=True)
    parser.add_argument("--runs-root", default="/home/pkunwar/characterize_ttlora/runs")
    parser.add_argument("--output-root", default="/home/pkunwar/characterize_ttlora/suites")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    runs_root = Path(args.runs_root).expanduser().resolve()
    output_dir = Path(args.output_root).expanduser().resolve() / args.suite_name / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = collect_run_dirs(runs_root, args.suite_name)
    if not run_dirs:
        raise FileNotFoundError(
            f"No run directories found under {runs_root} for suite prefix 'suite_{args.suite_name}_'."
        )

    bundles = [load_run_bundle(run_dir) for run_dir in run_dirs]
    flat_rows = [flattened_summary_row(bundle) for bundle in bundles]
    aggregated_rows = aggregate_rows(flat_rows)

    write_csv(output_dir / "all_runs_summary.csv", flat_rows)
    write_csv(output_dir / "aggregated_summary.csv", aggregated_rows)
    plot_accuracy_vs_lr(aggregated_rows, output_dir)
    plot_memory_vs_accuracy(aggregated_rows, output_dir)
    plot_stability_vs_accuracy(aggregated_rows, output_dir)
    report = generate_report(aggregated_rows, output_dir)
    save_json(
        output_dir / "analysis_manifest.json",
        {
            "suite_name": args.suite_name,
            "num_runs": len(bundles),
            "run_dirs": [str(path) for path in run_dirs],
        },
    )

    print(f"Saved suite analysis to: {output_dir}")
    print(report)


if __name__ == "__main__":
    main()
