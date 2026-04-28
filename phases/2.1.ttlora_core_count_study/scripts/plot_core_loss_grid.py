from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


DEFAULT_VARIANT_STYLES = {
    "contraction": "-",
    "reconstruction": "--",
}


def parse_args() -> argparse.Namespace:
    phase_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "For each dataset, create one image containing a grid of core-count subplots. "
            "Each subplot overlays train and validation loss for contraction and reconstruction."
        )
    )
    parser.add_argument(
        "--runs-root",
        default=str(phase_root / "runs_refined"),
        help="Root directory containing per-run folders with summary.json and history.csv.",
    )
    parser.add_argument(
        "--train-loss-column",
        default="train_loss",
        help="Training loss column from history.csv.",
    )
    parser.add_argument(
        "--validation-loss-column",
        default="validation_loss",
        help="Validation loss column from history.csv.",
    )
    parser.add_argument(
        "--output-path",
        default=str(phase_root / "analysis_outputs" / "core_loss_grids"),
        help=(
            "Directory to save the generated figures. One image per dataset will be written here. "
            "If a file path is given, its parent directory will be used."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional title prefix added to each dataset figure.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Saved image DPI.",
    )
    parser.add_argument(
        "--grid-columns",
        type=int,
        default=5,
        help="Number of subplot columns per dataset figure.",
    )
    parser.add_argument(
        "--full-history",
        action="store_true",
        help="Plot the full history instead of truncating each run at its best validation epoch.",
    )
    return parser.parse_args()


def infer_total_cores(summary: dict[str, Any], run_name: str) -> int | None:
    ttlora_shape = summary.get("ttlora_shape")
    if isinstance(ttlora_shape, list) and ttlora_shape:
        return len(ttlora_shape)

    weight_configs = summary.get("ttlora_weight_configs")
    if isinstance(weight_configs, list) and weight_configs:
        tt_shape = weight_configs[0].get("tt_shape")
        if isinstance(tt_shape, list) and tt_shape:
            return len(tt_shape)

    match = re.search(r"cores(\d+)", run_name)
    return int(match.group(1)) if match else None


def coerce_frame(frame: pd.DataFrame) -> pd.DataFrame:
    converted = frame.copy()
    for column in converted.columns:
        try:
            converted[column] = pd.to_numeric(converted[column])
        except (TypeError, ValueError):
            pass
    return converted


def load_run_records(
    runs_root: Path,
    train_loss_column: str,
    validation_loss_column: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for summary_path in sorted(runs_root.rglob("summary.json")):
        run_dir = summary_path.parent
        history_path = run_dir / "history.csv"
        if not history_path.exists():
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        history = coerce_frame(pd.read_csv(history_path))
        if "epoch" not in history.columns:
            continue
        if train_loss_column not in history.columns and validation_loss_column not in history.columns:
            continue

        dataset_name = summary.get("dataset_name") or run_dir.parent.name.split("_")[0]
        ttlora_variant = summary.get("ttlora_variant") or run_dir.parent.name.split("_", 1)[-1]
        run_name = run_dir.name
        total_cores = infer_total_cores(summary, run_name)
        if total_cores is None:
            continue

        records.append(
            {
                "dataset_name": str(dataset_name),
                "ttlora_variant": str(ttlora_variant),
                "task_type": str(summary.get("task_type") or "").lower(),
                "run_name": run_name,
                "total_cores": int(total_cores),
                "learning_rate": summary.get("learning_rate"),
                "best_epoch": summary.get("best_epoch"),
                "history": history,
                "has_train_loss": train_loss_column in history.columns,
                "has_validation_loss": validation_loss_column in history.columns,
            }
        )
    return records


def dataset_sort_key(name: str) -> tuple[int, str]:
    preferred_order = {"cola": 0, "mrpc": 1, "enron": 2, "ptb": 3}
    return (preferred_order.get(name, 999), name)


def make_core_color_map(records: list[dict[str, Any]]) -> dict[int, Any]:
    core_counts = sorted({record["total_cores"] for record in records})
    if not core_counts:
        return {}
    cmap = plt.get_cmap("viridis", max(len(core_counts), 2))
    return {core: cmap(index) for index, core in enumerate(core_counts)}


def lighten_color(color: Any, strength: float = 0.45) -> tuple[float, float, float]:
    r, g, b, *_ = color if isinstance(color, tuple) else (0.0, 0.0, 0.0)
    return (
        r + (1.0 - r) * strength,
        g + (1.0 - g) * strength,
        b + (1.0 - b) * strength,
    )


def plot_core_panel(
    ax: plt.Axes,
    core_records: list[dict[str, Any]],
    dataset_name: str,
    core_count: int,
    train_loss_column: str,
    validation_loss_column: str,
    truncate_at_best_epoch: bool,
) -> None:
    if not core_records:
        ax.axis("off")
        return

    variant_colors = {
        "contraction": "#1f77b4",
        "reconstruction": "#d62728",
    }

    for record in sorted(core_records, key=lambda item: item["ttlora_variant"]):
        history = record["history"]
        variant = record["ttlora_variant"]
        linestyle = DEFAULT_VARIANT_STYLES.get(variant, "-")
        validation_color = variant_colors.get(variant, "tab:blue")
        train_color = lighten_color(validation_color, strength=0.55)
        best_epoch = record.get("best_epoch")

        if truncate_at_best_epoch and best_epoch is not None:
            try:
                best_epoch_value = int(best_epoch)
            except (TypeError, ValueError):
                best_epoch_value = None
            if best_epoch_value is not None:
                history = history.loc[history["epoch"] <= best_epoch_value].copy()

        if history.empty:
            continue

        if record["has_train_loss"]:
            ax.plot(
                history["epoch"],
                history[train_loss_column],
                linestyle=linestyle,
                linewidth=1.6,
                color=train_color,
                alpha=0.95,
            )

        if record["has_validation_loss"]:
            ax.plot(
                history["epoch"],
                history[validation_loss_column],
                linestyle=linestyle,
                linewidth=2.2,
                color=validation_color,
                alpha=0.98,
            )

    ax.set_title(f"{dataset_name.upper()} | {core_count} cores", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)


def build_legends() -> tuple[list[Line2D], list[Line2D]]:
    variant_handles = [
        Line2D([0], [0], color="#1f77b4", lw=2.5, linestyle="-", label="Contraction"),
        Line2D([0], [0], color="#d62728", lw=2.5, linestyle="--", label="Reconstruction"),
    ]
    loss_handles = [
        Line2D([0], [0], color="#9ecae1", lw=2.0, linestyle="-", label="Train Loss"),
        Line2D([0], [0], color="#1f77b4", lw=2.5, linestyle="-", label="Validation Loss"),
    ]
    return variant_handles, loss_handles


def normalize_output_dir(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.parent
    return output_path


def save_dataset_figure(
    dataset_records: list[dict[str, Any]],
    dataset_name: str,
    train_loss_column: str,
    validation_loss_column: str,
    output_dir: Path,
    title_prefix: str | None,
    dpi: int,
    grid_columns: int,
    truncate_at_best_epoch: bool,
) -> Path:
    core_counts = sorted({record["total_cores"] for record in dataset_records})
    ncols = max(1, grid_columns)
    nrows = math.ceil(len(core_counts) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 3.9 * nrows),
        constrained_layout=True,
        sharex=False,
        sharey=False,
    )
    axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, core_count in zip(axes_flat, core_counts):
        core_records = [record for record in dataset_records if record["total_cores"] == core_count]
        plot_core_panel(
            ax=ax,
            core_records=core_records,
            dataset_name=dataset_name,
            core_count=core_count,
            train_loss_column=train_loss_column,
            validation_loss_column=validation_loss_column,
            truncate_at_best_epoch=truncate_at_best_epoch,
        )

    for ax in axes_flat[len(core_counts):]:
        ax.axis("off")

    variant_handles, loss_handles = build_legends()
    fig.legend(
        handles=variant_handles,
        loc="upper center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 1.02),
        title="TT-LoRA Variant",
    )
    fig.legend(
        handles=loss_handles,
        loc="upper center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 0.95),
        title="Loss Type",
    )

    figure_title = f"{dataset_name.upper()} Core Loss Grid"
    if title_prefix:
        figure_title = f"{title_prefix} | {figure_title}"
    fig.suptitle(figure_title, fontsize=16, fontweight="bold")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_name}_core_loss_grid.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    global plt, Line2D, pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import pandas as pd

    runs_root = Path(args.runs_root).expanduser().resolve()
    output_dir = normalize_output_dir(Path(args.output_path).expanduser().resolve())
    records = load_run_records(
        runs_root,
        train_loss_column=args.train_loss_column,
        validation_loss_column=args.validation_loss_column,
    )
    if not records:
        raise FileNotFoundError(
            "No runs with summary.json + history.csv + the requested loss columns "
            f"found under {runs_root}"
        )

    dataset_names = sorted({record["dataset_name"] for record in records}, key=dataset_sort_key)
    for dataset_name in dataset_names:
        dataset_records = [record for record in records if record["dataset_name"] == dataset_name]
        saved_path = save_dataset_figure(
            dataset_records=dataset_records,
            dataset_name=dataset_name,
            train_loss_column=args.train_loss_column,
            validation_loss_column=args.validation_loss_column,
            output_dir=output_dir,
            title_prefix=args.title,
            dpi=args.dpi,
            grid_columns=args.grid_columns,
            truncate_at_best_epoch=not args.full_history,
        )
        print(f"Saved figure to {saved_path}")


if __name__ == "__main__":
    main()
