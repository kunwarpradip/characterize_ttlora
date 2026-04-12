from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import json

import sys

PHASE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PHASE_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from suite_utils import write_csv


def collect_run_dirs(runs_root: Path, suite_name: str) -> list[Path]:
    prefix = f"ttcomp_{suite_name}_"
    return sorted(
        path for path in runs_root.iterdir() if path.is_dir() and path.name.startswith(prefix) and (path / "summary.json").exists()
    )


def load_summary(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def flatten_summary(run_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "dataset_name": summary.get("dataset_name"),
        "ttlora_variant": summary.get("ttlora_variant"),
        "ttlora_rank": summary.get("ttlora_rank"),
        "learning_rate": summary.get("learning_rate"),
        "seed": summary.get("seed"),
        "best_epoch": summary.get("best_epoch"),
        "best_validation_accuracy": summary.get("best_validation_accuracy"),
        "final_validation_accuracy": summary.get("final_validation_accuracy"),
        "final_validation_loss": summary.get("final_validation_loss"),
        "avg_clipped_step_fraction": summary.get("avg_clipped_step_fraction"),
        "total_clipped_steps": summary.get("total_clipped_steps"),
        "avg_grad_norm": summary.get("avg_grad_norm"),
        "max_grad_norm": summary.get("max_grad_norm"),
        "avg_epoch_seconds": summary.get("avg_epoch_seconds"),
        "max_peak_memory_gb": summary.get("max_peak_memory_gb"),
        "trainable_parameters": summary.get("trainable_parameters"),
        "summary_only": summary.get("summary_only"),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset_name"], row["ttlora_variant"], row["ttlora_rank"], row["learning_rate"])].append(row)

    aggregated = []
    for (dataset_name, variant, rank, learning_rate), group in sorted(grouped.items()):
        best_vals = [row["best_validation_accuracy"] for row in group]
        final_vals = [row["final_validation_accuracy"] for row in group]
        clip_vals = [row["avg_clipped_step_fraction"] for row in group]
        grad_vals = [row["avg_grad_norm"] for row in group]
        max_grad_vals = [row["max_grad_norm"] for row in group]
        time_vals = [row["avg_epoch_seconds"] for row in group]
        mem_vals = [row["max_peak_memory_gb"] for row in group]
        aggregated.append(
            {
                "dataset_name": dataset_name,
                "ttlora_variant": variant,
                "ttlora_rank": rank,
                "learning_rate": learning_rate,
                "num_seeds": len(group),
                "mean_best_validation_accuracy": float(pd.Series(best_vals).mean()),
                "std_best_validation_accuracy": float(pd.Series(best_vals).std(ddof=0)),
                "mean_final_validation_accuracy": float(pd.Series(final_vals).mean()),
                "std_final_validation_accuracy": float(pd.Series(final_vals).std(ddof=0)),
                "mean_final_validation_loss": float(pd.Series([row["final_validation_loss"] for row in group]).mean()),
                "mean_clipped_step_fraction": float(pd.Series(clip_vals).mean()),
                "mean_total_clipped_steps": float(pd.Series([row["total_clipped_steps"] for row in group]).mean()),
                "mean_avg_grad_norm": float(pd.Series(grad_vals).mean()),
                "mean_max_grad_norm": float(pd.Series(max_grad_vals).mean()),
                "mean_avg_epoch_seconds": float(pd.Series(time_vals).mean()),
                "mean_peak_memory_gb": float(pd.Series(mem_vals).mean()),
                "trainable_parameters": group[0]["trainable_parameters"],
            }
        )
    return aggregated


def plot_lr_curves(aggregated: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("mean_best_validation_accuracy", "Best Validation Accuracy"),
        ("mean_clipped_step_fraction", "Clipped Step Fraction"),
        ("mean_avg_grad_norm", "Average Grad Norm"),
    ]
    for dataset_name in sorted(aggregated["dataset_name"].unique()):
        for variant in sorted(aggregated["ttlora_variant"].unique()):
            subset = aggregated[(aggregated["dataset_name"] == dataset_name) & (aggregated["ttlora_variant"] == variant)]
            fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.5))
            for ax, (metric, title) in zip(axes, metrics):
                for rank in sorted(subset["ttlora_rank"].unique()):
                    rank_rows = subset[subset["ttlora_rank"] == rank].sort_values("learning_rate")
                    ax.plot(rank_rows["learning_rate"], rank_rows[metric], marker="o", linewidth=2, label=f"rank {rank}")
                ax.set_xscale("log")
                ax.set_xlabel("Learning Rate")
                ax.set_title(f"{variant}: {title}")
                ax.grid(True, alpha=0.3)
            axes[0].legend()
            fig.suptitle(f"{dataset_name} rank x learning-rate curves", y=1.02)
            fig.tight_layout()
            fig.savefig(output_dir / f"{dataset_name}_{variant}_rank_lr_curves.png", dpi=180, bbox_inches="tight")
            plt.close(fig)


def plot_heatmaps(aggregated: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("mean_best_validation_accuracy", "Best Val Accuracy"),
        ("mean_final_validation_accuracy", "Final Val Accuracy"),
        ("mean_clipped_step_fraction", "Clipped Fraction"),
    ]
    for dataset_name in sorted(aggregated["dataset_name"].unique()):
        for variant in sorted(aggregated["ttlora_variant"].unique()):
            subset = aggregated[(aggregated["dataset_name"] == dataset_name) & (aggregated["ttlora_variant"] == variant)]
            fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.8))
            for ax, (metric, title) in zip(axes, metrics):
                pivot = subset.pivot(index="ttlora_rank", columns="learning_rate", values=metric).sort_index().sort_index(axis=1)
                image = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
                ax.set_title(f"{variant}: {title}")
                ax.set_xlabel("Learning Rate")
                ax.set_ylabel("TT rank")
                ax.set_xticks(range(len(pivot.columns)), [f"{value:.0e}" for value in pivot.columns], rotation=45, ha="right")
                ax.set_yticks(range(len(pivot.index)), [str(value) for value in pivot.index])
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle(f"{dataset_name} rank x learning-rate heatmaps", y=1.02)
            fig.tight_layout()
            fig.savefig(output_dir / f"{dataset_name}_{variant}_rank_lr_heatmaps.png", dpi=180, bbox_inches="tight")
            plt.close(fig)


def generate_report(aggregated: pd.DataFrame, output_dir: Path) -> str:
    lines = ["# TT-LoRA Rank x Learning Rate Report", ""]
    for dataset_name in sorted(aggregated["dataset_name"].unique()):
        dataset_rows = aggregated[aggregated["dataset_name"] == dataset_name]
        lines.append(f"## {dataset_name}")
        for variant in sorted(dataset_rows["ttlora_variant"].unique()):
            variant_rows = dataset_rows[dataset_rows["ttlora_variant"] == variant]
            best = variant_rows.sort_values(
                ["mean_best_validation_accuracy", "mean_final_validation_accuracy"], ascending=False
            ).iloc[0]
            most_stable = variant_rows.sort_values(
                ["mean_clipped_step_fraction", "mean_avg_grad_norm"], ascending=True
            ).iloc[0]
            lines.append(
                f"- `{variant}` best setting: rank `{int(best['ttlora_rank'])}`, lr `{best['learning_rate']:.0e}`, "
                f"mean best acc `{best['mean_best_validation_accuracy']:.4f}`, mean final acc `{best['mean_final_validation_accuracy']:.4f}`."
            )
            lines.append(
                f"- `{variant}` most stable setting: rank `{int(most_stable['ttlora_rank'])}`, lr `{most_stable['learning_rate']:.0e}`, "
                f"clipped fraction `{most_stable['mean_clipped_step_fraction']:.4f}`, avg grad norm `{most_stable['mean_avg_grad_norm']:.4f}`."
            )
        lines.append("")
    report = "\n".join(lines)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze TT-LoRA rank x learning-rate experiments.")
    parser.add_argument("--suite-name", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    runs_root = Path(args.runs_root).expanduser().resolve()
    output_dir = Path(args.output_root).expanduser().resolve() / args.suite_name / "analysis" / "rank_lr"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = collect_run_dirs(runs_root, args.suite_name)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found for ttcomp suite '{args.suite_name}' under {runs_root}.")

    summaries = [(run_dir, load_summary(run_dir)) for run_dir in run_dirs]
    flat_rows = [flatten_summary(run_dir, summary) for run_dir, summary in summaries]
    aggregated_rows = aggregate_rows(flat_rows)

    write_csv(output_dir / "all_runs_summary.csv", flat_rows)
    write_csv(output_dir / "aggregated_summary.csv", aggregated_rows)

    aggregated_df = pd.DataFrame(aggregated_rows)
    plot_lr_curves(aggregated_df, output_dir)
    plot_heatmaps(aggregated_df, output_dir)
    report = generate_report(aggregated_df, output_dir)

    print(f"Saved rank x lr analysis to: {output_dir}")
    print(report)


if __name__ == "__main__":
    main()
