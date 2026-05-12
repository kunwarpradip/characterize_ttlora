from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from transformers import AutoConfig

from analyze_tt_shape_parameter_space import (
    build_parameter_space,
    render_line_panel,
    write_csv,
    write_svg,
)


@dataclass(frozen=True)
class AttentionWeightSpec:
    weight_name: str
    module_path: str
    in_features: int
    out_features: int


def sanitize_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("_")


def parse_int_list(values: list[str] | None) -> list[int] | None:
    if not values:
        return None
    return [int(item) for item in values]


def parse_str_list(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    return [str(item).strip() for item in values if str(item).strip()]


def resolve_model_identity(config, model_name: str) -> str:
    raw_type = str(getattr(config, "model_type", "") or "").lower()
    normalized_name = model_name.strip().lower()
    if raw_type == "gpt2" or "gpt2" in normalized_name:
        return "gpt2small"
    if raw_type == "roberta" or "roberta" in normalized_name:
        return "roberta-base"
    if raw_type == "llama" or "llama" in normalized_name:
        return "llama3.2-1b" if "3.2" in normalized_name else "llama"
    raise ValueError(
        f"Unsupported model family for model_name={model_name!r}, config.model_type={raw_type!r}. "
        "Extend resolve_model_identity() for this architecture."
    )


def discover_attention_weights(model_name: str, model_path: Path) -> tuple[str, int, list[AttentionWeightSpec]]:
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
    model_identity = resolve_model_identity(config, model_name)
    model_type = str(getattr(config, "model_type", "") or "").lower()

    if model_type == "gpt2":
        hidden_size = int(getattr(config, "n_embd", getattr(config, "hidden_size")))
        num_layers = int(getattr(config, "n_layer", getattr(config, "num_hidden_layers")))
        specs = [
            AttentionWeightSpec("c_attn", "transformer.h.*.attn.c_attn", hidden_size, hidden_size * 3),
            AttentionWeightSpec("c_proj", "transformer.h.*.attn.c_proj", hidden_size, hidden_size),
        ]
        return model_identity, num_layers, specs

    if model_type == "roberta":
        hidden_size = int(getattr(config, "hidden_size"))
        num_layers = int(getattr(config, "num_hidden_layers"))
        specs = [
            AttentionWeightSpec("query", "roberta.encoder.layer.*.attention.self.query", hidden_size, hidden_size),
            AttentionWeightSpec("key", "roberta.encoder.layer.*.attention.self.key", hidden_size, hidden_size),
            AttentionWeightSpec("value", "roberta.encoder.layer.*.attention.self.value", hidden_size, hidden_size),
        ]
        return model_identity, num_layers, specs

    if model_type == "llama":
        hidden_size = int(getattr(config, "hidden_size"))
        num_layers = int(getattr(config, "num_hidden_layers"))
        num_attention_heads = int(getattr(config, "num_attention_heads"))
        num_key_value_heads = int(getattr(config, "num_key_value_heads", num_attention_heads))
        head_dim = int(getattr(config, "head_dim", hidden_size // num_attention_heads))
        kv_out = num_key_value_heads * head_dim
        specs = [
            AttentionWeightSpec("q_proj", "model.layers.*.self_attn.q_proj", hidden_size, hidden_size),
            AttentionWeightSpec("k_proj", "model.layers.*.self_attn.k_proj", hidden_size, kv_out),
            AttentionWeightSpec("v_proj", "model.layers.*.self_attn.v_proj", hidden_size, kv_out),
            AttentionWeightSpec("o_proj", "model.layers.*.self_attn.o_proj", hidden_size, hidden_size),
        ]
        return model_identity, num_layers, specs

    raise ValueError(f"Unsupported model_type '{model_type}' in {model_path}")


def parse_json_field(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def format_param_count(value: float) -> str:
    return f"{int(round(value)):,}"


def render_labeled_line_panel(
    *,
    panel_x: int,
    panel_y: int,
    panel_width: int,
    panel_height: int,
    title: str,
    x_label: str,
    y_label: str,
    points: list[tuple[float, float]],
    color: str = "#111111",
) -> str:
    left = 70
    right = 20
    top = 40
    bottom = 55
    plot_x = panel_x + left
    plot_y = panel_y + top
    plot_width = panel_width - left - right
    plot_height = panel_height - top - bottom

    x_values = [float(item[0]) for item in points]
    y_values = [float(item[1]) for item in points]
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

    y_ticks = 5
    for tick_index in range(y_ticks):
        tick = y_min + (y_max - y_min) * tick_index / max(1, y_ticks - 1)
        y = scale_y(tick)
        parts.append(f'<line x1="{plot_x}" y1="{y:.2f}" x2="{plot_x + plot_width}" y2="{y:.2f}" stroke="#eeeeee"/>')
        parts.append(f'<text class="tick" x="{plot_x - 8}" y="{y + 4:.2f}" text-anchor="end">{escape(format_param_count(tick))}</text>')

    x_ticks = sorted({int(value) for value in x_values})
    for tick in x_ticks:
        x = scale_x(float(tick))
        parts.append(f'<line x1="{x:.2f}" y1="{plot_y}" x2="{x:.2f}" y2="{plot_y + plot_height}" stroke="#f2f2f2"/>')
        parts.append(f'<text class="tick" x="{x:.2f}" y="{plot_y + plot_height + 18}" text-anchor="middle">{tick}</text>')

    polyline = " ".join(f"{scale_x(float(x)):.2f},{scale_y(float(y)):.2f}" for x, y in points)
    parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.4" points="{polyline}"/>')
    for x_value, y_value in points:
        x = scale_x(float(x_value))
        y = scale_y(float(y_value))
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}" stroke="white" stroke-width="1"/>')
        parts.append(
            f'<text class="tick" x="{x + 6:.2f}" y="{y - 6:.2f}" text-anchor="start">{escape(format_param_count(float(y_value)))}</text>'
        )
    return "".join(parts)


def build_weight_analysis(
    *,
    weight_spec: AttentionWeightSpec,
    rank: int,
    split_strategy: str,
    allow_one_factors: bool,
    core_counts: tuple[int, ...] | None,
    output_dir: Path,
) -> dict[str, Any]:
    raw_rows, summary_rows, min_rows, max_rows = build_parameter_space(
        in_features=weight_spec.in_features,
        out_features=weight_spec.out_features,
        rank=rank,
        split_strategy=split_strategy,
        allow_one_factors=allow_one_factors,
        core_counts=core_counts,
        qkv_multiplier=1,
        layer_multiplier=1,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "all_shapes_parameter_counts.csv", raw_rows)
    write_csv(output_dir / "parameter_bounds_by_core_count.csv", summary_rows)
    write_csv(output_dir / "lowest_parameter_shapes_by_core_count.csv", min_rows)
    write_csv(output_dir / "highest_parameter_shapes_by_core_count.csv", max_rows)

    metadata = {
        "weight_name": weight_spec.weight_name,
        "module_path": weight_spec.module_path,
        "weight_shape": [weight_spec.out_features, weight_spec.in_features],
        "in_features": weight_spec.in_features,
        "out_features": weight_spec.out_features,
        "rank": rank,
        "split_strategy": split_strategy,
        "allow_one_factors": allow_one_factors,
    }
    (output_dir / "parameter_space_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    min_series = [
        (
            int(row["total_cores"]),
            int(row["all_layers_qkv_params"]),
        )
        for row in min_rows
    ]
    max_series = [
        (
            int(row["total_cores"]),
            int(row["all_layers_qkv_params"]),
        )
        for row in max_rows
    ]
    count_series = [
        (
            int(row["total_cores"]),
            int(row["num_shapes"]),
        )
        for row in summary_rows
    ]
    svg_body = "".join(
        [
            render_line_panel(
                panel_x=20,
                panel_y=20,
                panel_width=560,
                panel_height=320,
                title=f"{weight_spec.weight_name}: Parameter Bounds",
                x_label="Total Core Count",
                y_label="TT-LoRA Parameters / Matrix",
                series=[
                    {"label": "minimum", "color": "#1f77b4", "marker": "circle", "points": min_series},
                    {"label": "maximum", "color": "#d62728", "marker": "square", "points": max_series},
                ],
            ),
            render_line_panel(
                panel_x=600,
                panel_y=20,
                panel_width=560,
                panel_height=320,
                title=f"{weight_spec.weight_name}: Shape Count",
                x_label="Total Core Count",
                y_label="Number of TT Shapes",
                series=[
                    {"label": "shape count", "color": "#2ca02c", "marker": "triangle", "points": count_series},
                ],
            ),
        ]
    )
    write_svg(
        output_dir / "parameter_space_summary.svg",
        width=1180,
        height=360,
        title=f"{weight_spec.weight_name} TT Parameter Space",
        body=svg_body,
    )

    lowest_by_core_count: dict[int, dict[str, Any]] = {}
    for row in min_rows:
        core_count = int(row["total_cores"])
        lowest_by_core_count[core_count] = {
            "weight_name": weight_spec.weight_name,
            "module_path": weight_spec.module_path,
            "weight_shape": [weight_spec.out_features, weight_spec.in_features],
            "input_factors": parse_json_field(row["input_factors"]),
            "output_factors": parse_json_field(row["output_factors"]),
            "tt_shape": parse_json_field(row["tt_shape"]),
            "input_cores": int(row["input_cores"]),
            "output_cores": int(row["output_cores"]),
            "total_cores": core_count,
            "per_matrix_params": int(row["per_matrix_params"]),
        }

    return {
        "metadata": metadata,
        "lowest_by_core_count": lowest_by_core_count,
        "summary_rows": summary_rows,
    }


def build_combined_rank_outputs(
    *,
    model_identity: str,
    model_path: Path,
    num_layers: int,
    rank: int,
    weight_specs: list[AttentionWeightSpec],
    analyses: dict[str, dict[str, Any]],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_weight_core_sets = [
        set(analysis["lowest_by_core_count"].keys())
        for analysis in analyses.values()
    ]
    shared_core_counts = sorted(set.intersection(*per_weight_core_sets)) if per_weight_core_sets else []

    combined_rows: list[dict[str, Any]] = []
    config_entries: list[dict[str, Any]] = []
    for core_count in shared_core_counts:
        row: dict[str, Any] = {
            "core_count": core_count,
            "rank": rank,
        }
        weights_payload: dict[str, Any] = {}
        total_params_per_layer = 0
        for spec in weight_specs:
            item = analyses[spec.weight_name]["lowest_by_core_count"][core_count]
            weights_payload[spec.weight_name] = {
                "module_path": item["module_path"],
                "weight_shape": item["weight_shape"],
                "input_factors": item["input_factors"],
                "output_factors": item["output_factors"],
                "tt_shape": item["tt_shape"],
                "input_cores": item["input_cores"],
                "output_cores": item["output_cores"],
                "per_matrix_params": item["per_matrix_params"],
            }
            total_params_per_layer += int(item["per_matrix_params"])
            prefix = spec.weight_name
            row[f"{prefix}_weight_shape"] = json.dumps(item["weight_shape"])
            row[f"{prefix}_input_factors"] = json.dumps(item["input_factors"])
            row[f"{prefix}_output_factors"] = json.dumps(item["output_factors"])
            row[f"{prefix}_tt_shape"] = json.dumps(item["tt_shape"])
            row[f"{prefix}_per_matrix_params"] = int(item["per_matrix_params"])
        row["total_params_per_layer"] = total_params_per_layer
        row["total_params_all_layers"] = total_params_per_layer * num_layers
        combined_rows.append(row)
        config_entries.append(
            {
                "core_count": core_count,
                "rank": rank,
                "num_hidden_layers": num_layers,
                "total_params_per_layer": total_params_per_layer,
                "total_params_all_layers": total_params_per_layer * num_layers,
                "weights": weights_payload,
            }
        )

    if combined_rows:
        write_csv(output_dir / f"combined_lowest_parameter_configs_rank{rank}.csv", combined_rows)
        figure_series = []
        for index, spec in enumerate(weight_specs):
            color = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"][index % 6]
            points = [
                (row["core_count"], row[f"{spec.weight_name}_per_matrix_params"])
                for row in combined_rows
            ]
            figure_series.append(
                {
                    "label": spec.weight_name,
                    "color": color,
                    "marker": "circle" if index % 3 == 0 else ("square" if index % 3 == 1 else "triangle"),
                    "points": points,
                }
            )
        svg_body = render_line_panel(
            panel_x=20,
            panel_y=20,
            panel_width=860,
            panel_height=360,
            title=f"{model_identity} rank={rank}: Minimum Parameters by Core Count",
            x_label="Total Core Count",
            y_label="TT-LoRA Parameters / Matrix",
            series=figure_series,
        )
        write_svg(
            output_dir / f"combined_min_parameter_counts_rank{rank}.svg",
            width=900,
            height=400,
            title=f"{model_identity} rank {rank} combined minimum parameters",
            body=svg_body,
        )
        overall_svg_body = render_labeled_line_panel(
            panel_x=20,
            panel_y=20,
            panel_width=860,
            panel_height=360,
            title=f"{model_identity} rank={rank}: Overall Combined Minimum Parameter Count",
            x_label="Total Core Count",
            y_label="Total TT-LoRA Parameters / Layer",
            points=[(row["core_count"], row["total_params_per_layer"]) for row in combined_rows],
        )
        write_svg(
            output_dir / f"overall_combined_min_parameter_count_rank{rank}.svg",
            width=900,
            height=400,
            title=f"{model_identity} rank {rank} overall combined minimum parameter count",
            body=overall_svg_body,
        )

    rank_payload = {
        "rank": rank,
        "shared_core_counts": shared_core_counts,
        "configs": config_entries,
    }
    return combined_rows, rank_payload


def parse_args() -> argparse.Namespace:
    project_root = Path("/home/pkunwar/characterize_ttlora")
    phase_root = project_root / "phases/2.1.ttlora_core_count_study"
    parser = argparse.ArgumentParser(
        description=(
            "Build a model-level TT-LoRA config by discovering supported attention weights, "
            "enumerating TT parameter space per weight, and saving the minimum-parameter "
            "configurations grouped by rank and core count."
        )
    )
    parser.add_argument("--model-name", required=True, help="Model alias, e.g. gpt2small or llama3.2-1b.")
    parser.add_argument("--model-path", required=True, help="Local checkpoint/config path for the model.")
    parser.add_argument(
        "--weights",
        nargs="*",
        default=None,
        help="Optional subset of attention weights to include. Defaults to all supported weights for the model.",
    )
    parser.add_argument(
        "--ranks",
        nargs="*",
        default=["2", "6", "10"],
        help="TT-LoRA ranks to analyze. Defaults to 2 6 10.",
    )
    parser.add_argument(
        "--core-counts",
        nargs="*",
        default=None,
        help="Optional explicit total core counts to analyze. Defaults to all inferred feasible counts.",
    )
    parser.add_argument(
        "--split-strategy",
        default="all",
        choices=("all", "symmetric", "near-symmetric"),
        help="How to split total TT cores between input and output factorizations.",
    )
    parser.add_argument("--allow-one-factors", action="store_true", help="Allow factors of 1 in TT factorizations.")
    parser.add_argument(
        "--analysis-root",
        default=str(phase_root / "analysis" / "model_ttshape_configs"),
        help="Root directory for analysis CSV/SVG outputs.",
    )
    parser.add_argument(
        "--config-output-dir",
        default=str(project_root / "ttshape_configs" / "ttlora"),
        help="Directory where the model-level JSON config will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    analysis_root = Path(args.analysis_root).expanduser().resolve()
    config_output_dir = Path(args.config_output_dir).expanduser().resolve()
    selected_weights = parse_str_list(args.weights)
    ranks = parse_int_list(args.ranks) or [2, 6, 10]
    core_counts = tuple(parse_int_list(args.core_counts) or []) or None

    model_identity, num_layers, discovered_specs = discover_attention_weights(args.model_name, model_path)
    if selected_weights is not None:
        selected_set = set(selected_weights)
        missing = sorted(selected_set.difference(spec.weight_name for spec in discovered_specs))
        if missing:
            raise ValueError(f"Unsupported weights for {model_identity}: {missing}")
        weight_specs = [spec for spec in discovered_specs if spec.weight_name in selected_set]
    else:
        weight_specs = list(discovered_specs)

    model_slug = sanitize_slug(model_identity)
    model_analysis_root = analysis_root / model_slug
    config_output_dir.mkdir(parents=True, exist_ok=True)
    model_analysis_root.mkdir(parents=True, exist_ok=True)

    config_payload: dict[str, Any] = {
        "model_name": model_identity,
        "model_path": str(model_path),
        "num_hidden_layers": num_layers,
        "supported_weights": {
            spec.weight_name: {
                "module_path": spec.module_path,
                "weight_shape": [spec.out_features, spec.in_features],
            }
            for spec in weight_specs
        },
        "ranks": {},
    }

    for rank in ranks:
        rank_root = model_analysis_root / f"rank{rank}"
        analyses: dict[str, dict[str, Any]] = {}
        for spec in weight_specs:
            weight_output_dir = rank_root / spec.weight_name
            analyses[spec.weight_name] = build_weight_analysis(
                weight_spec=spec,
                rank=rank,
                split_strategy=args.split_strategy,
                allow_one_factors=args.allow_one_factors,
                core_counts=core_counts,
                output_dir=weight_output_dir,
            )

        _, rank_payload = build_combined_rank_outputs(
            model_identity=model_identity,
            model_path=model_path,
            num_layers=num_layers,
            rank=rank,
            weight_specs=weight_specs,
            analyses=analyses,
            output_dir=rank_root,
        )
        config_payload["ranks"][str(rank)] = rank_payload

    config_path = config_output_dir / f"{model_slug}.json"
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    print(config_path)


if __name__ == "__main__":
    main()
