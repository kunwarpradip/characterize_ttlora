from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Focused analysis for TT-LoRA contraction vs reconstruction.")
    parser.add_argument("--suite-name", required=True)
    parser.add_argument("--dataset-name", default="mrpc")
    parser.add_argument("--runs-root", default="/home/pkunwar/characterize_ttlora/runs")
    parser.add_argument("--analysis-root", default="/home/pkunwar/characterize_ttlora/suites")
    return parser


def load_tables(analysis_root: Path, suite_name: str, dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    suite_analysis_dir = analysis_root / suite_name / "analysis"
    all_runs = pd.read_csv(suite_analysis_dir / "all_runs_summary.csv")
    aggregated = pd.read_csv(suite_analysis_dir / "aggregated_summary.csv")

    methods = ["ttlora-contraction", "ttlora-reconstruction"]
    all_runs = all_runs[(all_runs["dataset_name"] == dataset_name) & (all_runs["method_label"].isin(methods))].copy()
    aggregated = aggregated[
        (aggregated["dataset_name"] == dataset_name) & (aggregated["method_label"].isin(methods))
    ].copy()
    if all_runs.empty or aggregated.empty:
        raise FileNotFoundError(f"No TT-LoRA rows found for suite={suite_name} dataset={dataset_name}.")

    all_runs["best_final_gap"] = all_runs["best_validation_accuracy"] - all_runs["final_validation_accuracy"]
    return all_runs, aggregated


def save_tables(output_dir: Path, all_runs: pd.DataFrame, aggregated: pd.DataFrame) -> None:
    all_runs.to_csv(output_dir / "ttlora_all_runs_summary.csv", index=False)
    aggregated.to_csv(output_dir / "ttlora_aggregated_summary.csv", index=False)

    filtered_ops = all_runs[all_runs["max_peak_memory_gb"] > 0].copy()
    filtered_ops.to_csv(output_dir / "ttlora_operational_rows_filtered.csv", index=False)

    common_lrs = sorted(
        set(aggregated[aggregated["method_label"] == "ttlora-contraction"]["learning_rate"]).intersection(
            set(aggregated[aggregated["method_label"] == "ttlora-reconstruction"]["learning_rate"])
        )
    )
    rows = []
    for lr in common_lrs:
        contraction = aggregated[
            (aggregated["method_label"] == "ttlora-contraction") & (aggregated["learning_rate"] == lr)
        ].iloc[0]
        reconstruction = aggregated[
            (aggregated["method_label"] == "ttlora-reconstruction") & (aggregated["learning_rate"] == lr)
        ].iloc[0]
        rows.append(
            {
                "learning_rate": lr,
                "delta_best_accuracy_recon_minus_contract": reconstruction["mean_best_validation_accuracy"]
                - contraction["mean_best_validation_accuracy"],
                "delta_final_accuracy_recon_minus_contract": reconstruction["mean_final_validation_accuracy"]
                - contraction["mean_final_validation_accuracy"],
                "delta_clipped_fraction_recon_minus_contract": reconstruction["mean_clipped_step_fraction"]
                - contraction["mean_clipped_step_fraction"],
                "delta_avg_grad_norm_recon_minus_contract": reconstruction["mean_avg_grad_norm"]
                - contraction["mean_avg_grad_norm"],
                "delta_max_grad_norm_recon_minus_contract": reconstruction["mean_max_grad_norm"]
                - contraction["mean_max_grad_norm"],
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "ttlora_paired_lr_deltas.csv", index=False)


def best_lr_map(aggregated: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for method, method_rows in aggregated.groupby("method_label"):
        best_row = method_rows.sort_values(
            ["mean_best_validation_accuracy", "mean_final_validation_accuracy"], ascending=False
        ).iloc[0]
        result[method] = float(best_row["learning_rate"])
    return result


def plot_accuracy_vs_lr(aggregated: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for method in ["ttlora-contraction", "ttlora-reconstruction"]:
        rows = aggregated[aggregated["method_label"] == method].sort_values("learning_rate")
        ax.errorbar(
            rows["learning_rate"],
            rows["mean_best_validation_accuracy"],
            yerr=rows["std_best_validation_accuracy"],
            marker="o",
            linewidth=2,
            capsize=4,
            label=f"{method} best",
        )
        ax.plot(
            rows["learning_rate"],
            rows["mean_final_validation_accuracy"],
            marker="s",
            linestyle="--",
            linewidth=1.8,
            label=f"{method} final",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Accuracy")
    ax.set_title("TT-LoRA Accuracy vs Learning Rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ttlora_accuracy_vs_lr.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_stability_vs_lr(aggregated: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    metrics = [
        ("mean_clipped_step_fraction", "Mean Clipped Step Fraction"),
        ("mean_avg_grad_norm", "Mean Avg Grad Norm"),
        ("mean_max_grad_norm", "Mean Max Grad Norm"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        for method in ["ttlora-contraction", "ttlora-reconstruction"]:
            rows = aggregated[aggregated["method_label"] == method].sort_values("learning_rate")
            ax.plot(rows["learning_rate"], rows[metric], marker="o", linewidth=2, label=method)
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ttlora_stability_vs_lr.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_efficiency_vs_lr(all_runs: pd.DataFrame, output_dir: Path) -> None:
    filtered = all_runs[all_runs["max_peak_memory_gb"] > 0].copy()
    if filtered.empty:
        return

    grouped = (
        filtered.groupby(["method_label", "learning_rate"])
        .agg(
            mean_epoch_seconds=("avg_epoch_seconds", "mean"),
            std_epoch_seconds=("avg_epoch_seconds", "std"),
            mean_peak_memory_gb=("max_peak_memory_gb", "mean"),
            std_peak_memory_gb=("max_peak_memory_gb", "std"),
            num_rows=("seed", "count"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for method in ["ttlora-contraction", "ttlora-reconstruction"]:
        rows = grouped[grouped["method_label"] == method].sort_values("learning_rate")
        axes[0].errorbar(
            rows["learning_rate"],
            rows["mean_epoch_seconds"],
            yerr=rows["std_epoch_seconds"].fillna(0.0),
            marker="o",
            linewidth=2,
            capsize=4,
            label=f"{method} (n={int(rows['num_rows'].min())}-{int(rows['num_rows'].max())})",
        )
        axes[1].errorbar(
            rows["learning_rate"],
            rows["mean_peak_memory_gb"],
            yerr=rows["std_peak_memory_gb"].fillna(0.0),
            marker="o",
            linewidth=2,
            capsize=4,
            label=method,
        )
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_xlabel("Learning Rate")
    axes[1].set_xlabel("Learning Rate")
    axes[0].set_title("Mean Epoch Seconds")
    axes[1].set_title("Peak Memory (positive GPU-memory rows only)")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ttlora_efficiency_vs_lr_filtered.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_paired_deltas(aggregated: pd.DataFrame, output_dir: Path) -> None:
    common_lrs = sorted(
        set(aggregated[aggregated["method_label"] == "ttlora-contraction"]["learning_rate"]).intersection(
            set(aggregated[aggregated["method_label"] == "ttlora-reconstruction"]["learning_rate"])
        )
    )
    if not common_lrs:
        return

    rows = []
    for lr in common_lrs:
        contraction = aggregated[
            (aggregated["method_label"] == "ttlora-contraction") & (aggregated["learning_rate"] == lr)
        ].iloc[0]
        reconstruction = aggregated[
            (aggregated["method_label"] == "ttlora-reconstruction") & (aggregated["learning_rate"] == lr)
        ].iloc[0]
        rows.append(
            {
                "learning_rate": lr,
                "best_acc_delta": reconstruction["mean_best_validation_accuracy"]
                - contraction["mean_best_validation_accuracy"],
                "final_acc_delta": reconstruction["mean_final_validation_accuracy"]
                - contraction["mean_final_validation_accuracy"],
                "clip_delta": reconstruction["mean_clipped_step_fraction"]
                - contraction["mean_clipped_step_fraction"],
            }
        )
    deltas = pd.DataFrame(rows)
    x = range(len(deltas))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar([idx - 0.15 for idx in x], deltas["best_acc_delta"], width=0.3, label="Best accuracy delta")
    axes[0].bar([idx + 0.15 for idx in x], deltas["final_acc_delta"], width=0.3, label="Final accuracy delta")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_xticks(list(x), [f"{lr:.0e}" for lr in deltas["learning_rate"]])
    axes[0].set_title("Reconstruction - Contraction Accuracy Delta")
    axes[0].set_xlabel("Learning Rate")
    axes[0].set_ylabel("Delta")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(list(x), deltas["clip_delta"], width=0.45, color="#c44e52")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks(list(x), [f"{lr:.0e}" for lr in deltas["learning_rate"]])
    axes[1].set_title("Reconstruction - Contraction Clipped Fraction Delta")
    axes[1].set_xlabel("Learning Rate")
    axes[1].set_ylabel("Delta")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "ttlora_paired_lr_deltas.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_best_lr_summary(all_runs: pd.DataFrame, aggregated: pd.DataFrame, output_dir: Path) -> dict[str, float]:
    best_lrs = best_lr_map(aggregated)
    rows = []
    for method, lr in best_lrs.items():
        row = aggregated[(aggregated["method_label"] == method) & (aggregated["learning_rate"] == lr)].iloc[0]
        rows.append(row)
    best_df = pd.DataFrame(rows)
    best_df.to_csv(output_dir / "ttlora_best_lr_summary.csv", index=False)

    metrics = [
        ("mean_best_validation_accuracy", "Mean Best Validation Accuracy"),
        ("mean_final_validation_accuracy", "Mean Final Validation Accuracy"),
        ("mean_clipped_step_fraction", "Mean Clipped Step Fraction"),
        ("mean_avg_grad_norm", "Mean Avg Grad Norm"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (metric, title) in zip(axes.flatten(), metrics):
        ax.bar(best_df["method_label"], best_df[metric])
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        for idx, (_, row) in enumerate(best_df.iterrows()):
            ax.text(idx, row[metric], f"lr={row['learning_rate']:.0e}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "ttlora_best_lr_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return best_lrs


def plot_best_lr_convergence(all_runs: pd.DataFrame, output_dir: Path, best_lrs: dict[str, float]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    metrics = [
        ("train_loss", "Train Loss"),
        ("validation_loss", "Validation Loss"),
        ("validation_accuracy", "Validation Accuracy"),
        ("clipped_step_fraction", "Clipped Step Fraction"),
    ]

    for method, lr in best_lrs.items():
        selected = all_runs[(all_runs["method_label"] == method) & (all_runs["learning_rate"] == lr)].copy()
        histories = []
        for _, row in selected.iterrows():
            history_path = Path(row["run_dir"]) / "history.csv"
            history = pd.read_csv(history_path)
            histories.append(history)
        merged = pd.concat(histories, keys=range(len(histories)), names=["seed_idx", "row"])
        mean_history = merged.groupby("epoch")[["train_loss", "validation_loss", "validation_accuracy", "clipped_step_fraction"]].mean()
        for ax, (metric, title) in zip(axes.flatten(), metrics):
            ax.plot(mean_history.index, mean_history[metric], marker="o", linewidth=2, label=f"{method} lr={lr:.0e}")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ttlora_best_lr_convergence.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_report(output_dir: Path, all_runs: pd.DataFrame, aggregated: pd.DataFrame, best_lrs: dict[str, float]) -> str:
    best_rows = []
    for method, lr in best_lrs.items():
        best_rows.append(
            aggregated[(aggregated["method_label"] == method) & (aggregated["learning_rate"] == lr)].iloc[0]
        )
    best_df = pd.DataFrame(best_rows)

    contraction_best = best_df[best_df["method_label"] == "ttlora-contraction"].iloc[0]
    reconstruction_best = best_df[best_df["method_label"] == "ttlora-reconstruction"].iloc[0]

    filtered_ops = all_runs[all_runs["max_peak_memory_gb"] > 0].copy()
    filtered_ops_means = filtered_ops.groupby("method_label")[["avg_epoch_seconds", "max_peak_memory_gb"]].mean()

    lines = [
        "# TT-LoRA Variant Comparison",
        "",
        "## Headline",
        f"- Best mean validation accuracy was slightly higher for reconstruction at `lr={reconstruction_best['learning_rate']:.0e}` with `{reconstruction_best['mean_best_validation_accuracy']:.4f}`.",
        f"- Best contraction performance came at `lr={contraction_best['learning_rate']:.0e}` with `{contraction_best['mean_best_validation_accuracy']:.4f}`.",
        f"- At their best learning rates, reconstruction exceeded contraction in mean best accuracy by `{reconstruction_best['mean_best_validation_accuracy'] - contraction_best['mean_best_validation_accuracy']:+.4f}`, but it also had a higher clipped-step fraction by `{reconstruction_best['mean_clipped_step_fraction'] - contraction_best['mean_clipped_step_fraction']:+.4f}`.",
        "",
        "## What Reconstruction Helps",
        "- Slightly higher peak task performance once the learning rate is pushed high enough.",
        "- At the shared `2e-3` learning rate, reconstruction finished higher than contraction; across each method's own best learning rate, final accuracy was effectively tied and very slightly favored contraction.",
        "- Comparable seed-to-seed accuracy spread near its best operating point.",
        "",
        "## What Contraction Helps",
        "- Reaches strong performance at lower learning rates. At `5e-4` and `1e-3`, contraction outperformed reconstruction in mean best accuracy.",
        "- Lower clipping pressure at every shared learning rate in this sweep.",
        "- Lower average and peak gradient norms in the strong-learning regimes, suggesting a gentler optimization landscape.",
        "- Smaller best-to-final accuracy drop at `1e-3`, so it is a bit less sensitive to overshooting once it reaches a good point.",
        "",
        "## Shared Observations",
        f"- Both variants have the same trainable parameter count: `{int(contraction_best['trainable_parameters'])}`.",
        "- Reconstruction at `1e-4` was perfectly stable but effectively underfit, staying at the baseline validation accuracy across all seeds.",
        "- Contraction became clearly unstable at `5e-3`, with a large best-to-final collapse and extreme gradient spikes.",
        "- The real decision boundary in this sweep is not parameter count but optimization behavior.",
        "",
        "## Operational Caveat",
        "- The four seed-44 reconstruction runs were rerun with `num_workers=0` because the sandbox blocks multiprocessing dataloader workers. Those reruns also reported `0.0` GPU peak memory, so raw mean time/memory across all reconstruction runs are not directly comparable to contraction.",
        f"- Using only rows with positive GPU-memory readings, reconstruction averaged `{filtered_ops_means.loc['ttlora-reconstruction', 'avg_epoch_seconds']:.4f}` epoch seconds and `{filtered_ops_means.loc['ttlora-reconstruction', 'max_peak_memory_gb']:.4f} GB`, while contraction averaged `{filtered_ops_means.loc['ttlora-contraction', 'avg_epoch_seconds']:.4f}` epoch seconds and `{filtered_ops_means.loc['ttlora-contraction', 'max_peak_memory_gb']:.4f} GB`.",
        "- That filtered operational view suggests near-parity: reconstruction is a bit faster, contraction uses a bit less GPU memory, but the differences are small and should be treated as provisional.",
        "",
        "## Recommendation",
        "- Choose contraction if you want the safer default: it is easier to tune, performs better at lower learning rates, and consistently clips less.",
        "- Choose reconstruction if you are willing to tune learning rate aggressively and care most about squeezing out the last bit of validation accuracy.",
        "- For future characterization, keep contraction as the default structural baseline and treat reconstruction as a high-upside, higher-sensitivity variant.",
    ]
    report = "\n".join(lines)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return report


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    analysis_root = Path(args.analysis_root).expanduser().resolve()
    runs_root = Path(args.runs_root).expanduser().resolve()
    output_dir = analysis_root / args.suite_name / "analysis" / "ttlora_variants"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_runs, aggregated = load_tables(analysis_root, args.suite_name, args.dataset_name)
    save_tables(output_dir, all_runs, aggregated)
    plot_accuracy_vs_lr(aggregated, output_dir)
    plot_stability_vs_lr(aggregated, output_dir)
    plot_efficiency_vs_lr(all_runs, output_dir)
    plot_paired_deltas(aggregated, output_dir)
    best_lrs = plot_best_lr_summary(all_runs, aggregated, output_dir)
    plot_best_lr_convergence(all_runs, output_dir, best_lrs)
    report = generate_report(output_dir, all_runs, aggregated, best_lrs)

    print(f"Saved TT-LoRA variant analysis to: {output_dir}")
    print(report)


if __name__ == "__main__":
    main()
