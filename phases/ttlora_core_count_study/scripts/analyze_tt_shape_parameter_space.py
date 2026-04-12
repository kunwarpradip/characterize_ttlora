from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

from generate_tt_shapes import (
    candidate_splits,
    distinct_factor_orderings,
    infer_core_counts,
    ordered_factorizations,
)


def ttlora_rank_list(rank: int, tt_shape: tuple[int, ...]) -> tuple[int, ...]:
    if rank < 1:
        raise ValueError("TT-LoRA rank must be >= 1.")
    return (1, *([rank] * (len(tt_shape) - 1)), 1)


def ttlora_parameters_per_matrix(tt_shape: tuple[int, ...], rank: int) -> int:
    tt_rank = ttlora_rank_list(rank, tt_shape)
    return sum(tt_rank[index] * dim * tt_rank[index + 1] for index, dim in enumerate(tt_shape))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def svg_header(width: int, height: int, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<title>{escape(title)}</title>'
        '<rect width="100%" height="100%" fill="white"/>'
        '<style>'
        'text{font-family:Arial,sans-serif;fill:#222;}'
        '.title{font-size:20px;font-weight:bold;}'
        '.axis-label{font-size:13px;}'
        '.tick{font-size:11px;fill:#555;}'
        '.legend{font-size:12px;}'
        '.panel-title{font-size:15px;font-weight:bold;}'
        '</style>'
    )


def svg_footer() -> str:
    return "</svg>"


def draw_marker(x: float, y: float, color: str, marker: str) -> str:
    if marker == "square":
        return (
            f'<rect x="{x - 3:.2f}" y="{y - 3:.2f}" width="6" height="6" '
            f'fill="{color}" stroke="white" stroke-width="1"/>'
        )
    if marker == "triangle":
        return (
            f'<polygon points="{x:.2f},{y - 4:.2f} {x - 4:.2f},{y + 3:.2f} {x + 4:.2f},{y + 3:.2f}" '
            f'fill="{color}" stroke="white" stroke-width="1"/>'
        )
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}" stroke="white" stroke-width="1"/>'


def format_tick(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value/1_000:.1f}K"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def nice_ticks(min_value: float, max_value: float, count: int = 5) -> list[float]:
    if min_value == max_value:
        pad = max(abs(min_value) * 0.05, 1.0)
        min_value -= pad
        max_value += pad
    if count <= 1:
        return [min_value, max_value]
    step = (max_value - min_value) / (count - 1)
    return [min_value + index * step for index in range(count)]


def render_line_panel(
    panel_x: int,
    panel_y: int,
    panel_width: int,
    panel_height: int,
    title: str,
    x_label: str,
    y_label: str,
    series: list[dict],
) -> str:
    left = 70
    right = 20
    top = 40
    bottom = 55
    plot_x = panel_x + left
    plot_y = panel_y + top
    plot_width = panel_width - left - right
    plot_height = panel_height - top - bottom

    x_values = [float(point[0]) for item in series for point in item["points"]]
    y_values = [float(point[1]) for item in series for point in item["points"]]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        pad = max(abs(y_min) * 0.05, 1.0)
        y_min -= pad
        y_max += pad

    x_pad = (x_max - x_min) * 0.04
    y_pad = (y_max - y_min) * 0.08
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    def scale_x(value: float) -> float:
        return plot_x + (value - x_min) / (x_max - x_min) * plot_width

    def scale_y(value: float) -> float:
        return plot_y + plot_height - (value - y_min) / (y_max - y_min) * plot_height

    parts = [
        f'<rect x="{panel_x}" y="{panel_y}" width="{panel_width}" height="{panel_height}" fill="white" stroke="#dddddd"/>',
        f'<text class="panel-title" x="{panel_x + panel_width / 2:.2f}" y="{panel_y + 22}" text-anchor="middle">{escape(title)}</text>',
        f'<line x1="{plot_x}" y1="{plot_y + plot_height}" x2="{plot_x + plot_width}" y2="{plot_y + plot_height}" stroke="#444"/>',
        f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_height}" stroke="#444"/>',
        f'<text class="axis-label" x="{panel_x + panel_width / 2:.2f}" y="{panel_y + panel_height - 12}" text-anchor="middle">{escape(x_label)}</text>',
        f'<text class="axis-label" x="{panel_x + 18}" y="{panel_y + panel_height / 2:.2f}" text-anchor="middle" transform="rotate(-90 {panel_x + 18} {panel_y + panel_height / 2:.2f})">{escape(y_label)}</text>',
    ]

    for tick in nice_ticks(y_min, y_max):
        y = scale_y(tick)
        parts.append(f'<line x1="{plot_x}" y1="{y:.2f}" x2="{plot_x + plot_width}" y2="{y:.2f}" stroke="#eeeeee"/>')
        parts.append(f'<text class="tick" x="{plot_x - 8}" y="{y + 4:.2f}" text-anchor="end">{escape(format_tick(tick))}</text>')

    for tick in nice_ticks(x_min, x_max):
        x = scale_x(tick)
        parts.append(f'<line x1="{x:.2f}" y1="{plot_y}" x2="{x:.2f}" y2="{plot_y + plot_height}" stroke="#f2f2f2"/>')
        parts.append(f'<text class="tick" x="{x:.2f}" y="{plot_y + plot_height + 18}" text-anchor="middle">{escape(format_tick(tick))}</text>')

    legend_x = plot_x + plot_width - 150
    legend_y = plot_y + 12
    for index, item in enumerate(series):
        points = " ".join(f"{scale_x(float(x)):.2f},{scale_y(float(y)):.2f}" for x, y in item["points"])
        parts.append(f'<polyline fill="none" stroke="{item["color"]}" stroke-width="2.2" points="{points}"/>')
        for x_value, y_value in item["points"]:
            parts.append(draw_marker(scale_x(float(x_value)), scale_y(float(y_value)), item["color"], item["marker"]))
        ly = legend_y + index * 18
        parts.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 18}" y2="{ly}" stroke="{item["color"]}" stroke-width="2.2"/>')
        parts.append(draw_marker(legend_x + 9, ly, item["color"], item["marker"]))
        parts.append(f'<text class="legend" x="{legend_x + 25}" y="{ly + 4}">{escape(item["label"])}</text>')

    return "".join(parts)


def write_svg(path: Path, width: int, height: int, title: str, body: str) -> None:
    path.write_text(svg_header(width, height, title) + body + svg_footer(), encoding="utf-8")


def build_parameter_space(
    in_features: int,
    out_features: int,
    rank: int,
    split_strategy: str,
    allow_one_factors: bool,
    core_counts: tuple[int, ...] | None,
    qkv_multiplier: int,
    layer_multiplier: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    min_factor = 1 if allow_one_factors else 2
    if core_counts is None:
        core_counts = infer_core_counts(
            in_features=in_features,
            out_features=out_features,
            split_strategy=split_strategy,
            allow_one_factors=allow_one_factors,
        )

    raw_rows: list[dict] = []
    grouped: dict[int, list[dict]] = defaultdict(list)

    for total_cores in core_counts:
        for input_cores, output_cores in candidate_splits(total_cores, split_strategy):
            input_factorizations = ordered_factorizations(in_features, input_cores, min_factor)
            output_factorizations = ordered_factorizations(out_features, output_cores, min_factor)
            for input_factors in input_factorizations:
                for output_factors in output_factorizations:
                    for input_ordering in distinct_factor_orderings(input_factors):
                        for output_ordering in distinct_factor_orderings(output_factors):
                            tt_shape = (*input_ordering, *reversed(output_ordering))
                            per_matrix_params = ttlora_parameters_per_matrix(tt_shape, rank)
                            per_layer_qkv_params = per_matrix_params * qkv_multiplier
                            all_layers_qkv_params = per_layer_qkv_params * layer_multiplier
                            row = {
                                "weight_shape": json.dumps([out_features, in_features]),
                                "in_features": in_features,
                                "out_features": out_features,
                                "adapted_weights_per_layer": qkv_multiplier,
                                "num_layers": layer_multiplier,
                                "total_cores": total_cores,
                                "input_cores": input_cores,
                                "output_cores": output_cores,
                                "input_factors": json.dumps(list(input_ordering)),
                                "output_factors": json.dumps(list(output_ordering)),
                                "tt_shape": json.dumps(list(tt_shape)),
                                "per_matrix_params": per_matrix_params,
                                "per_layer_qkv_params": per_layer_qkv_params,
                                "all_layers_qkv_params": all_layers_qkv_params,
                            }
                            raw_rows.append(row)
                            grouped[total_cores].append(row)

    summary_rows: list[dict] = []
    min_rows: list[dict] = []
    max_rows: list[dict] = []
    for total_cores in sorted(grouped):
        group = grouped[total_cores]
        params = [int(item["all_layers_qkv_params"]) for item in group]
        min_item = min(group, key=lambda item: int(item["all_layers_qkv_params"]))
        max_item = max(group, key=lambda item: int(item["all_layers_qkv_params"]))
        unique_param_counts = sorted({int(item["all_layers_qkv_params"]) for item in group})
        summary_rows.append(
            {
                "weight_shape": json.dumps([out_features, in_features]),
                "in_features": in_features,
                "out_features": out_features,
                "adapted_weights_per_layer": qkv_multiplier,
                "num_layers": layer_multiplier,
                "total_cores": total_cores,
                "num_shapes": len(group),
                "num_unique_total_param_counts": len(unique_param_counts),
                "min_total_params": min(params),
                "max_total_params": max(params),
                "min_per_matrix_params": min(int(item["per_matrix_params"]) for item in group),
                "max_per_matrix_params": max(int(item["per_matrix_params"]) for item in group),
                "min_shape": min_item["tt_shape"],
                "max_shape": max_item["tt_shape"],
                "min_input_factors": min_item["input_factors"],
                "min_output_factors": min_item["output_factors"],
                "max_input_factors": max_item["input_factors"],
                "max_output_factors": max_item["output_factors"],
            }
        )
        min_rows.append(
            {
                "weight_shape": json.dumps([out_features, in_features]),
                "in_features": in_features,
                "out_features": out_features,
                "adapted_weights_per_layer": qkv_multiplier,
                "num_layers": layer_multiplier,
                "total_cores": total_cores,
                "num_shapes": len(group),
                "num_unique_total_param_counts": len(unique_param_counts),
                "all_layers_qkv_params": int(min_item["all_layers_qkv_params"]),
                "per_layer_qkv_params": int(min_item["per_layer_qkv_params"]),
                "per_matrix_params": int(min_item["per_matrix_params"]),
                "input_cores": int(min_item["input_cores"]),
                "output_cores": int(min_item["output_cores"]),
                "input_factors": min_item["input_factors"],
                "output_factors": min_item["output_factors"],
                "tt_shape": min_item["tt_shape"],
            }
        )
        max_rows.append(
            {
                "weight_shape": json.dumps([out_features, in_features]),
                "in_features": in_features,
                "out_features": out_features,
                "adapted_weights_per_layer": qkv_multiplier,
                "num_layers": layer_multiplier,
                "total_cores": total_cores,
                "num_shapes": len(group),
                "num_unique_total_param_counts": len(unique_param_counts),
                "all_layers_qkv_params": int(max_item["all_layers_qkv_params"]),
                "per_layer_qkv_params": int(max_item["per_layer_qkv_params"]),
                "per_matrix_params": int(max_item["per_matrix_params"]),
                "input_cores": int(max_item["input_cores"]),
                "output_cores": int(max_item["output_cores"]),
                "input_factors": max_item["input_factors"],
                "output_factors": max_item["output_factors"],
                "tt_shape": max_item["tt_shape"],
            }
        )

    return raw_rows, summary_rows, min_rows, max_rows


def plot_shape_count_and_param_bounds(summary_rows: list[dict], output_path: Path) -> None:
    shape_count_series = [
        {
            "label": "Possible TT Shapes",
            "color": "#1f77b4",
            "marker": "circle",
            "points": [(row["total_cores"], row["num_shapes"]) for row in summary_rows],
        }
    ]
    param_series = [
        {
            "label": "Lowest Total Params",
            "color": "#2ca02c",
            "marker": "circle",
            "points": [(row["total_cores"], row["min_total_params"]) for row in summary_rows],
        },
        {
            "label": "Highest Total Params",
            "color": "#d62728",
            "marker": "square",
            "points": [(row["total_cores"], row["max_total_params"]) for row in summary_rows],
        },
    ]

    width = 1420
    height = 520
    body = f'<text class="title" x="{width / 2:.2f}" y="28" text-anchor="middle">TT Shape Space for RoBERTa Attention (QKV × 12 Layers)</text>'
    body += render_line_panel(25, 50, 660, 430, "How Many TT Shapes Are Possible?", "Total TT Cores", "Number of Shapes", shape_count_series)
    body += render_line_panel(715, 50, 680, 430, "Lowest and Highest Total Parameter Counts", "Total TT Cores", "Total TT-LoRA Params", param_series)
    write_svg(output_path, width, height, "TT Shape Count and Parameter Bounds", body)


def plot_all_shape_parameter_scatter(raw_rows: list[dict], output_path: Path) -> None:
    grouped: dict[int, list[int]] = defaultdict(list)
    for row in raw_rows:
        grouped[int(row["total_cores"])].append(int(row["all_layers_qkv_params"]))

    width = 1000
    height = 560
    panel_x = 25
    panel_y = 50
    panel_width = 950
    panel_height = 470
    left = 70
    right = 20
    top = 40
    bottom = 55
    plot_x = panel_x + left
    plot_y = panel_y + top
    plot_width = panel_width - left - right
    plot_height = panel_height - top - bottom

    core_counts = sorted(grouped)
    x_min = min(core_counts) - 0.5
    x_max = max(core_counts) + 0.5
    y_values = [value for values in grouped.values() for value in values]
    y_min = min(y_values)
    y_max = max(y_values)
    y_pad = (y_max - y_min) * 0.08 if y_max > y_min else 1.0
    y_min -= y_pad
    y_max += y_pad

    def scale_x(value: float) -> float:
        return plot_x + (value - x_min) / (x_max - x_min) * plot_width

    def scale_y(value: float) -> float:
        return plot_y + plot_height - (value - y_min) / (y_max - y_min) * plot_height

    parts = [
        f'<text class="title" x="{width / 2:.2f}" y="28" text-anchor="middle">All Valid TT Shapes: Total Parameter Counts</text>',
        f'<rect x="{panel_x}" y="{panel_y}" width="{panel_width}" height="{panel_height}" fill="white" stroke="#dddddd"/>',
        f'<line x1="{plot_x}" y1="{plot_y + plot_height}" x2="{plot_x + plot_width}" y2="{plot_y + plot_height}" stroke="#444"/>',
        f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_height}" stroke="#444"/>',
        f'<text class="axis-label" x="{panel_x + panel_width / 2:.2f}" y="{panel_y + panel_height - 12}" text-anchor="middle">Total TT Cores</text>',
        f'<text class="axis-label" x="{panel_x + 18}" y="{panel_y + panel_height / 2:.2f}" text-anchor="middle" transform="rotate(-90 {panel_x + 18} {panel_y + panel_height / 2:.2f})">Total TT-LoRA Params</text>',
        f'<text class="panel-title" x="{panel_x + panel_width / 2:.2f}" y="{panel_y + 22}" text-anchor="middle">Each point is one valid symmetric TT shape</text>',
    ]

    for tick in [y_min + index * (y_max - y_min) / 4 for index in range(5)]:
        y = scale_y(tick)
        parts.append(f'<line x1="{plot_x}" y1="{y:.2f}" x2="{plot_x + plot_width}" y2="{y:.2f}" stroke="#eeeeee"/>')
        parts.append(f'<text class="tick" x="{plot_x - 8}" y="{y + 4:.2f}" text-anchor="end">{escape(format_tick(tick))}</text>')

    for core_count in core_counts:
        x = scale_x(core_count)
        parts.append(f'<line x1="{x:.2f}" y1="{plot_y}" x2="{x:.2f}" y2="{plot_y + plot_height}" stroke="#f2f2f2"/>')
        parts.append(f'<text class="tick" x="{x:.2f}" y="{plot_y + plot_height + 18}" text-anchor="middle">{core_count}</text>')
        values = sorted(grouped[core_count])
        color = "#1f77b4"
        for index, value in enumerate(values):
            jitter = ((index % 7) - 3) * 1.6
            parts.append(draw_marker(x + jitter, scale_y(value), color, "circle"))

    write_svg(output_path, width, height, "All TT Shape Parameter Counts", "".join(parts))


def plot_range_width(summary_rows: list[dict], output_path: Path) -> None:
    range_series = [
        {
            "label": "Parameter Range Width",
            "color": "#9467bd",
            "marker": "triangle",
            "points": [
                (row["total_cores"], row["max_total_params"] - row["min_total_params"])
                for row in summary_rows
            ],
        }
    ]
    unique_count_series = [
        {
            "label": "Unique Total Param Counts",
            "color": "#8c564b",
            "marker": "square",
            "points": [
                (row["total_cores"], row["num_unique_total_param_counts"])
                for row in summary_rows
            ],
        }
    ]

    width = 1420
    height = 520
    body = f'<text class="title" x="{width / 2:.2f}" y="28" text-anchor="middle">TT Shape Space Diversity by Total Core Count</text>'
    body += render_line_panel(25, 50, 660, 430, "How Wide Is the Parameter Range?", "Total TT Cores", "Max - Min Total Params", range_series)
    body += render_line_panel(715, 50, 680, 430, "How Many Unique Parameter Totals Exist?", "Total TT Cores", "Unique Total Param Counts", unique_count_series)
    write_svg(output_path, width, height, "TT Shape Space Diversity", body)


def build_parser() -> argparse.ArgumentParser:
    phase_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Analyze the full symmetric TT-shape parameter space for the core-count study."
    )
    parser.add_argument("--phase-root", default=str(phase_root))
    parser.add_argument(
        "--weight-dim",
        type=int,
        default=None,
        help="Convenience option for square weights. Sets both --in-features and --out-features.",
    )
    parser.add_argument(
        "--weight-shape",
        nargs=2,
        type=int,
        default=None,
        metavar=("OUT_FEATURES", "IN_FEATURES"),
        help="Explicit weight shape [out_features in_features]. Overrides --weight-dim.",
    )
    parser.add_argument("--in-features", type=int, default=768)
    parser.add_argument("--out-features", type=int, default=768)
    parser.add_argument("--rank", type=int, default=6)
    parser.add_argument("--split-strategy", choices=("symmetric", "near-symmetric", "all"), default="symmetric")
    parser.add_argument("--allow-one-factors", action="store_true")
    parser.add_argument("--core-counts", nargs="*", type=int, default=None)
    parser.add_argument(
        "--adapted-weights-per-layer",
        type=int,
        default=3,
        help="How many weights are adapted in each layer, e.g. 3 for query/key/value.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=12,
        help="How many layers use the same TT-LoRA adaptation pattern.",
    )
    parser.add_argument(
        "--qkv-multiplier",
        type=int,
        default=None,
        help="Deprecated alias for --adapted-weights-per-layer.",
    )
    parser.add_argument(
        "--layer-multiplier",
        type=int,
        default=None,
        help="Deprecated alias for --num-layers.",
    )
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    phase_root = Path(args.phase_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else phase_root / "analysis" / "parameter_space"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.weight_shape is not None:
        out_features, in_features = args.weight_shape
    else:
        in_features = args.weight_dim if args.weight_dim is not None else args.in_features
        out_features = args.weight_dim if args.weight_dim is not None else args.out_features
    adapted_weights_per_layer = (
        args.qkv_multiplier if args.qkv_multiplier is not None else args.adapted_weights_per_layer
    )
    num_layers = args.layer_multiplier if args.layer_multiplier is not None else args.num_layers

    raw_rows, summary_rows, min_rows, max_rows = build_parameter_space(
        in_features=in_features,
        out_features=out_features,
        rank=args.rank,
        split_strategy=args.split_strategy,
        allow_one_factors=args.allow_one_factors,
        core_counts=tuple(args.core_counts) if args.core_counts else None,
        qkv_multiplier=adapted_weights_per_layer,
        layer_multiplier=num_layers,
    )

    metadata = {
        "weight_dim": args.weight_dim,
        "weight_shape": [out_features, in_features],
        "in_features": in_features,
        "out_features": out_features,
        "rank": args.rank,
        "split_strategy": args.split_strategy,
        "allow_one_factors": args.allow_one_factors,
        "adapted_weights_per_layer": adapted_weights_per_layer,
        "num_layers": num_layers,
        "num_shapes": len(raw_rows),
        "num_core_counts": len(summary_rows),
    }

    raw_csv_path = output_dir / "all_shapes_parameter_counts.csv"
    summary_csv_path = output_dir / "parameter_bounds_by_core_count.csv"
    min_csv_path = output_dir / "lowest_parameter_shapes_by_core_count.csv"
    max_csv_path = output_dir / "highest_parameter_shapes_by_core_count.csv"
    metadata_json_path = output_dir / "parameter_space_metadata.json"
    figure1_path = output_dir / "shape_count_and_parameter_bounds.svg"
    figure2_path = output_dir / "all_shape_parameter_scatter.svg"
    figure3_path = output_dir / "parameter_space_diversity.svg"

    write_csv(raw_csv_path, raw_rows)
    write_csv(summary_csv_path, summary_rows)
    write_csv(min_csv_path, min_rows)
    write_csv(max_csv_path, max_rows)
    metadata_json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    plot_shape_count_and_param_bounds(summary_rows, figure1_path)
    plot_all_shape_parameter_scatter(raw_rows, figure2_path)
    plot_range_width(summary_rows, figure3_path)

    print(f"Wrote raw TT-shape parameter counts to {raw_csv_path}")
    print(f"Wrote per-core summary to {summary_csv_path}")
    print(f"Wrote lowest-parameter shapes to {min_csv_path}")
    print(f"Wrote highest-parameter shapes to {max_csv_path}")
    print(f"Wrote metadata to {metadata_json_path}")
    print(f"Wrote figure to {figure1_path}")
    print(f"Wrote figure to {figure2_path}")
    print(f"Wrote figure to {figure3_path}")


if __name__ == "__main__":
    main()
