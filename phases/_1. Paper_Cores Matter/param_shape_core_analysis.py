from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
from xml.sax.saxutils import escape


@dataclass(frozen=True)
class ShapeCandidate:
    weight_name: str
    weight_shape: tuple[int, int]
    rank: int
    core_count: int
    input_cores: int
    output_cores: int
    input_factors: tuple[int, ...]
    output_factors: tuple[int, ...]
    tt_shape: tuple[int, ...]
    parameter_count: int
    dense_parameter_count: int
    compression_ratio: float
    score: float
    balance_penalty: float
    core_size_penalty: float
    ones_penalty: float
    tt_matrix_parameter_count: int
    tt_matrix_compression_ratio: float
    edge_param_count: int
    edge_param_fraction: float
    first_core_param_fraction: float
    last_core_param_fraction: float
    core_param_sizes: tuple[int, ...]
    core_param_cv: float
    max_factor: int
    min_factor: int
    uses_one_factor: bool

    def to_row(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "weight_shape",
            "input_factors",
            "output_factors",
            "tt_shape",
            "core_param_sizes",
        ):
            payload[key] = json.dumps(list(payload[key]))
        return payload


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_core_counts(values: Iterable[str] | None) -> tuple[int, ...] | None:
    if not values:
        return None
    return tuple(int(value) for value in values)


@lru_cache(maxsize=None)
def ordered_factorizations(n: int, parts: int, min_factor: int = 2) -> tuple[tuple[int, ...], ...]:
    """Return non-decreasing multiplicative factorizations of n into parts terms."""
    if n < 1:
        raise ValueError("n must be positive.")
    if parts < 1:
        raise ValueError("parts must be positive.")
    if min_factor < 1:
        raise ValueError("min_factor must be positive.")
    if parts == 1:
        return ((n,),) if n >= min_factor else tuple()

    results: list[tuple[int, ...]] = []
    upper = int(round(n ** (1 / parts))) + 1
    for factor in range(min_factor, max(min_factor, upper) + 1):
        if n % factor != 0:
            continue
        for suffix in ordered_factorizations(n // factor, parts - 1, factor):
            results.append((factor, *suffix))
    return tuple(results)


@lru_cache(maxsize=None)
def distinct_factor_orderings(factors: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Return every unique ordering of a factor tuple."""
    if len(factors) <= 1:
        return (factors,)

    remaining: dict[int, int] = {}
    for factor in factors:
        remaining[factor] = remaining.get(factor, 0) + 1

    ordered_values = tuple(sorted(remaining))
    current: list[int] = []
    results: list[tuple[int, ...]] = []

    def backtrack() -> None:
        if len(current) == len(factors):
            results.append(tuple(current))
            return
        for value in ordered_values:
            count = remaining.get(value, 0)
            if count == 0:
                continue
            remaining[value] = count - 1
            current.append(value)
            backtrack()
            current.pop()
            remaining[value] = count

    backtrack()
    return tuple(results)


def candidate_splits(total_cores: int, strategy: str) -> tuple[tuple[int, int], ...]:
    """Return valid (input_cores, output_cores) splits for a total core count."""
    if total_cores < 2:
        raise ValueError("total_cores must be at least 2.")
    if strategy == "all":
        return tuple((input_cores, total_cores - input_cores) for input_cores in range(1, total_cores))
    if strategy == "symmetric":
        if total_cores % 2 != 0:
            return tuple()
        return ((total_cores // 2, total_cores // 2),)
    if strategy == "near-symmetric":
        left = total_cores // 2
        right = total_cores - left
        return ((left, right),)
    raise ValueError("split_strategy must be one of: all, symmetric, near-symmetric.")


def feasible_factor_counts(n: int) -> tuple[int, ...]:
    counts: list[int] = []
    parts = 1
    while ordered_factorizations(n, parts, 2):
        counts.append(parts)
        parts += 1
    return tuple(counts)


def infer_core_counts(in_features: int, out_features: int, split_strategy: str) -> tuple[int, ...]:
    input_counts = feasible_factor_counts(in_features)
    output_counts = feasible_factor_counts(out_features)

    if split_strategy == "symmetric":
        return tuple(2 * count for count in sorted(set(input_counts).intersection(output_counts)))
    if split_strategy == "near-symmetric":
        inferred = {
            input_count + output_count
            for input_count in input_counts
            for output_count in output_counts
            if abs(input_count - output_count) <= 1
        }
        return tuple(sorted(inferred))
    if split_strategy == "all":
        inferred = {
            input_count + output_count
            for input_count in input_counts
            for output_count in output_counts
        }
        return tuple(sorted(inferred))
    raise ValueError("split_strategy must be one of: all, symmetric, near-symmetric.")


def ttlora_rank_list(rank: int, tt_shape: tuple[int, ...]) -> tuple[int, ...]:
    if rank < 1:
        raise ValueError("rank must be at least 1.")
    if not tt_shape:
        raise ValueError("tt_shape cannot be empty.")
    return (1, *([rank] * (len(tt_shape) - 1)), 1)


def core_parameter_sizes(tt_shape: tuple[int, ...], rank: int) -> tuple[int, ...]:
    tt_rank = ttlora_rank_list(rank, tt_shape)
    return tuple(tt_rank[idx] * dim * tt_rank[idx + 1] for idx, dim in enumerate(tt_shape))


def ttlora_parameter_count(tt_shape: tuple[int, ...], rank: int) -> int:
    return sum(core_parameter_sizes(tt_shape, rank))


def pad_to_equal_length(
    output_factors: tuple[int, ...],
    input_factors: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    length = max(len(output_factors), len(input_factors))
    padded_output = (*output_factors, *([1] * (length - len(output_factors))))
    padded_input = (*input_factors, *([1] * (length - len(input_factors))))
    return padded_output, padded_input


def score_tt_shape(
    output_factors: tuple[int, ...],
    input_factors: tuple[int, ...],
    rank: int,
) -> dict[str, float | int]:
    """Structural TT-matrix score. Lower score means more balanced/less degenerate."""
    m_shape, n_shape = pad_to_equal_length(output_factors, input_factors)
    k = len(m_shape)
    ranks = [1] + [rank] * (k - 1) + [1]

    factors = m_shape + n_shape
    log_factors = [math.log(value) for value in factors]
    balance_penalty = statistics.pstdev(log_factors)

    paired_core_sizes = [m * n for m, n in zip(m_shape, n_shape)]
    log_core_sizes = [math.log(value) for value in paired_core_sizes]
    core_size_penalty = statistics.pstdev(log_core_sizes)

    ones_penalty = sum(1 for value in factors if value == 1) / len(factors)
    tt_params = sum(ranks[idx] * m_shape[idx] * n_shape[idx] * ranks[idx + 1] for idx in range(k))
    dense_params = math.prod(m_shape) * math.prod(n_shape)
    param_penalty = tt_params / dense_params

    score = (
        1.0 * balance_penalty
        + 1.0 * core_size_penalty
        + 2.0 * ones_penalty
        + 0.5 * param_penalty
    )

    return {
        "score": score,
        "balance_penalty": balance_penalty,
        "core_size_penalty": core_size_penalty,
        "ones_penalty": ones_penalty,
        "tt_params": tt_params,
        "compression_ratio": dense_params / tt_params,
    }


def coefficient_of_variation(values: tuple[int, ...]) -> float:
    if not values:
        return 0.0
    mean = statistics.mean(values)
    if mean == 0:
        return 0.0
    return statistics.pstdev(values) / mean


def build_candidate(
    *,
    weight_name: str,
    out_features: int,
    in_features: int,
    rank: int,
    core_count: int,
    input_cores: int,
    output_cores: int,
    input_factors: tuple[int, ...],
    output_factors: tuple[int, ...],
) -> ShapeCandidate:
    tt_shape = (*input_factors, *reversed(output_factors))
    dense_params = out_features * in_features
    parameter_count = ttlora_parameter_count(tt_shape, rank)
    score_payload = score_tt_shape(output_factors=output_factors, input_factors=input_factors, rank=rank)
    sizes = core_parameter_sizes(tt_shape, rank)
    edge_param_count = sizes[0] + sizes[-1]
    factors = (*input_factors, *output_factors)

    return ShapeCandidate(
        weight_name=weight_name,
        weight_shape=(out_features, in_features),
        rank=rank,
        core_count=core_count,
        input_cores=input_cores,
        output_cores=output_cores,
        input_factors=input_factors,
        output_factors=output_factors,
        tt_shape=tt_shape,
        parameter_count=parameter_count,
        dense_parameter_count=dense_params,
        compression_ratio=dense_params / parameter_count,
        score=float(score_payload["score"]),
        balance_penalty=float(score_payload["balance_penalty"]),
        core_size_penalty=float(score_payload["core_size_penalty"]),
        ones_penalty=float(score_payload["ones_penalty"]),
        tt_matrix_parameter_count=int(score_payload["tt_params"]),
        tt_matrix_compression_ratio=float(score_payload["compression_ratio"]),
        edge_param_count=edge_param_count,
        edge_param_fraction=edge_param_count / parameter_count,
        first_core_param_fraction=sizes[0] / parameter_count,
        last_core_param_fraction=sizes[-1] / parameter_count,
        core_param_sizes=sizes,
        core_param_cv=coefficient_of_variation(sizes),
        max_factor=max(factors),
        min_factor=min(factors),
        uses_one_factor=any(value == 1 for value in factors),
    )


def generate_candidates(
    *,
    weight_name: str,
    out_features: int,
    in_features: int,
    rank: int,
    core_counts: tuple[int, ...],
    split_strategy: str,
    allow_one_factors: bool,
) -> tuple[ShapeCandidate, ...]:
    min_factor = 1 if allow_one_factors else 2
    candidates: list[ShapeCandidate] = []

    for core_count in core_counts:
        for input_cores, output_cores in candidate_splits(core_count, split_strategy):
            input_factorizations = ordered_factorizations(in_features, input_cores, min_factor)
            output_factorizations = ordered_factorizations(out_features, output_cores, min_factor)
            for input_factorization in input_factorizations:
                for output_factorization in output_factorizations:
                    for input_ordering in distinct_factor_orderings(input_factorization):
                        for output_ordering in distinct_factor_orderings(output_factorization):
                            candidates.append(
                                build_candidate(
                                    weight_name=weight_name,
                                    out_features=out_features,
                                    in_features=in_features,
                                    rank=rank,
                                    core_count=core_count,
                                    input_cores=input_cores,
                                    output_cores=output_cores,
                                    input_factors=input_ordering,
                                    output_factors=output_ordering,
                                )
                            )

    return tuple(sorted(candidates, key=lambda item: (item.core_count, item.parameter_count, item.score, item.tt_shape)))


def summarize_candidates(candidates: tuple[ShapeCandidate, ...]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[int, list[ShapeCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.core_count].append(candidate)

    summary_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for core_count in sorted(grouped):
        group = grouped[core_count]
        params = [item.parameter_count for item in group]
        scores = [item.score for item in group]
        compressions = [item.compression_ratio for item in group]
        edge_fractions = [item.edge_param_fraction for item in group]

        lowest = min(group, key=lambda item: (item.parameter_count, item.score, item.tt_shape))
        highest = max(group, key=lambda item: (item.parameter_count, -item.score, item.tt_shape))
        balanced = min(group, key=lambda item: (item.score, item.parameter_count, item.core_param_cv, item.tt_shape))
        edge_heavy = max(group, key=lambda item: (item.edge_param_fraction, -item.score, -item.parameter_count, item.tt_shape))

        summary_rows.append(
            {
                "core_count": core_count,
                "num_shapes": len(group),
                "num_unique_parameter_counts": len(set(params)),
                "min_parameter_count": min(params),
                "median_parameter_count": statistics.median(params),
                "mean_parameter_count": statistics.mean(params),
                "max_parameter_count": max(params),
                "min_compression_ratio": min(compressions),
                "median_compression_ratio": statistics.median(compressions),
                "max_compression_ratio": max(compressions),
                "min_score": min(scores),
                "median_score": statistics.median(scores),
                "max_score": max(scores),
                "min_edge_param_fraction": min(edge_fractions),
                "median_edge_param_fraction": statistics.median(edge_fractions),
                "max_edge_param_fraction": max(edge_fractions),
                "lowest_param_tt_shape": json.dumps(list(lowest.tt_shape)),
                "balanced_tt_shape": json.dumps(list(balanced.tt_shape)),
                "edge_heavy_tt_shape": json.dumps(list(edge_heavy.tt_shape)),
            }
        )

        for label, item in (
            ("lowest_parameter", lowest),
            ("highest_parameter", highest),
            ("balanced", balanced),
            ("edge_heavy", edge_heavy),
        ):
            row = item.to_row()
            row["selection"] = label
            selected_rows.append(row)

    return summary_rows, selected_rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
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


def fmt(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def svg_header(width: int, height: int, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<title>{escape(title)}</title>'
        '<rect width="100%" height="100%" fill="white"/>'
        '<style>'
        'text{font-family:Arial,sans-serif;fill:#222;}'
        '.title{font-size:20px;font-weight:bold;}'
        '.axis{font-size:13px;}'
        '.tick{font-size:11px;fill:#555;}'
        '.legend{font-size:12px;}'
        '</style>'
    )


def svg_footer() -> str:
    return "</svg>"


def value_range(values: list[float], pad_fraction: float = 0.08) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if low == high:
        pad = max(abs(low) * 0.05, 1.0)
        return low - pad, high + pad
    pad = (high - low) * pad_fraction
    return low - pad, high + pad


def nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    if count <= 1:
        return [low]
    return [low + (high - low) * idx / (count - 1) for idx in range(count)]


def draw_marker(x: float, y: float, color: str, radius: float = 3.0) -> str:
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{color}" fill-opacity="0.72"/>'


def plot_line_series(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: list[dict[str, Any]],
    width: int = 1120,
    height: int = 620,
) -> None:
    if not series or not any(item["points"] for item in series):
        return

    left, right, top, bottom = 90, 30, 62, 78
    plot_x, plot_y = left, top
    plot_w, plot_h = width - left - right, height - top - bottom
    x_values = [float(x) for item in series for x, _ in item["points"]]
    y_values = [float(y) for item in series for _, y in item["points"]]
    x_low, x_high = value_range(x_values, pad_fraction=0.04)
    y_low, y_high = value_range(y_values, pad_fraction=0.10)

    def sx(value: float) -> float:
        return plot_x + (value - x_low) / (x_high - x_low) * plot_w

    def sy(value: float) -> float:
        return plot_y + plot_h - (value - y_low) / (y_high - y_low) * plot_h

    parts = [
        svg_header(width, height, title),
        f'<text class="title" x="{width / 2:.2f}" y="30" text-anchor="middle">{escape(title)}</text>',
        f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="#444"/>',
        f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="#444"/>',
        f'<text class="axis" x="{width / 2:.2f}" y="{height - 22}" text-anchor="middle">{escape(x_label)}</text>',
        f'<text class="axis" x="22" y="{height / 2:.2f}" text-anchor="middle" transform="rotate(-90 22 {height / 2:.2f})">{escape(y_label)}</text>',
    ]

    for tick in nice_ticks(y_low, y_high):
        y = sy(tick)
        parts.append(f'<line x1="{plot_x}" y1="{y:.2f}" x2="{plot_x + plot_w}" y2="{y:.2f}" stroke="#eeeeee"/>')
        parts.append(f'<text class="tick" x="{plot_x - 8}" y="{y + 4:.2f}" text-anchor="end">{escape(fmt(tick))}</text>')

    for tick in sorted({int(round(value)) for value in x_values}):
        x = sx(float(tick))
        parts.append(f'<line x1="{x:.2f}" y1="{plot_y}" x2="{x:.2f}" y2="{plot_y + plot_h}" stroke="#f2f2f2"/>')
        parts.append(f'<text class="tick" x="{x:.2f}" y="{plot_y + plot_h + 18}" text-anchor="middle">{tick}</text>')

    legend_x = width - 260
    legend_y = 58
    for idx, item in enumerate(series):
        color = item["color"]
        points = sorted((float(x), float(y)) for x, y in item["points"])
        polyline = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{polyline}"/>')
        for x, y in points:
            parts.append(draw_marker(sx(x), sy(y), color, radius=3.4))
        ly = legend_y + idx * 18
        parts.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 20}" y2="{ly}" stroke="{color}" stroke-width="2.2"/>')
        parts.append(f'<text class="legend" x="{legend_x + 28}" y="{ly + 4}">{escape(item["label"])}</text>')

    parts.append(svg_footer())
    path.write_text("".join(parts), encoding="utf-8")


def sample_scatter(candidates: tuple[ShapeCandidate, ...], max_points: int) -> tuple[ShapeCandidate, ...]:
    if len(candidates) <= max_points:
        return candidates
    grouped: dict[int, list[ShapeCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.core_count].append(candidate)
    per_group = max(1, max_points // max(1, len(grouped)))
    sampled: list[ShapeCandidate] = []
    for core_count in sorted(grouped):
        group = grouped[core_count]
        if len(group) <= per_group:
            sampled.extend(group)
            continue
        stride = math.ceil(len(group) / per_group)
        sampled.extend(group[::stride][:per_group])
    return tuple(sampled[:max_points])


def plot_scatter(
    path: Path,
    *,
    title: str,
    candidates: tuple[ShapeCandidate, ...],
    plotted_candidates: tuple[ShapeCandidate, ...],
    width: int = 1120,
    height: int = 620,
) -> None:
    if not plotted_candidates:
        return

    left, right, top, bottom = 90, 30, 62, 78
    plot_x, plot_y = left, top
    plot_w, plot_h = width - left - right, height - top - bottom
    x_values = [float(item.core_count) for item in plotted_candidates]
    y_values = [float(item.parameter_count) for item in plotted_candidates]
    x_low, x_high = value_range(x_values, pad_fraction=0.04)
    y_low, y_high = value_range(y_values, pad_fraction=0.10)

    def sx(value: float) -> float:
        return plot_x + (value - x_low) / (x_high - x_low) * plot_w

    def sy(value: float) -> float:
        return plot_y + plot_h - (value - y_low) / (y_high - y_low) * plot_h

    parts = [
        svg_header(width, height, title),
        f'<text class="title" x="{width / 2:.2f}" y="30" text-anchor="middle">{escape(title)}</text>',
        f'<text class="tick" x="{width / 2:.2f}" y="50" text-anchor="middle">plotted {len(plotted_candidates):,} of {len(candidates):,} candidates</text>',
        f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="#444"/>',
        f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="#444"/>',
        f'<text class="axis" x="{width / 2:.2f}" y="{height - 22}" text-anchor="middle">Total TT cores</text>',
        f'<text class="axis" x="22" y="{height / 2:.2f}" text-anchor="middle" transform="rotate(-90 22 {height / 2:.2f})">TT-LoRA parameters</text>',
    ]

    for tick in nice_ticks(y_low, y_high):
        y = sy(tick)
        parts.append(f'<line x1="{plot_x}" y1="{y:.2f}" x2="{plot_x + plot_w}" y2="{y:.2f}" stroke="#eeeeee"/>')
        parts.append(f'<text class="tick" x="{plot_x - 8}" y="{y + 4:.2f}" text-anchor="end">{escape(fmt(tick))}</text>')

    for tick in sorted({item.core_count for item in plotted_candidates}):
        x = sx(float(tick))
        parts.append(f'<line x1="{x:.2f}" y1="{plot_y}" x2="{x:.2f}" y2="{plot_y + plot_h}" stroke="#f2f2f2"/>')
        parts.append(f'<text class="tick" x="{x:.2f}" y="{plot_y + plot_h + 18}" text-anchor="middle">{tick}</text>')

    for idx, candidate in enumerate(plotted_candidates):
        jitter = ((idx % 11) - 5) * 1.1
        parts.append(draw_marker(sx(candidate.core_count) + jitter, sy(candidate.parameter_count), "#1f77b4", radius=2.6))

    parts.append(svg_footer())
    path.write_text("".join(parts), encoding="utf-8")


def write_visualizations(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    candidates: tuple[ShapeCandidate, ...],
    plot_max_points: int,
) -> dict[str, Any]:
    shape_points = [(row["core_count"], row["num_shapes"]) for row in summary_rows]
    parameter_min_points = [(row["core_count"], row["min_parameter_count"]) for row in summary_rows]
    parameter_median_points = [(row["core_count"], row["median_parameter_count"]) for row in summary_rows]
    parameter_max_points = [(row["core_count"], row["max_parameter_count"]) for row in summary_rows]
    score_min_points = [(row["core_count"], row["min_score"]) for row in summary_rows]
    score_median_points = [(row["core_count"], row["median_score"]) for row in summary_rows]
    score_max_points = [(row["core_count"], row["max_score"]) for row in summary_rows]
    edge_points = [(row["core_count"], row["max_edge_param_fraction"]) for row in summary_rows]

    plot_line_series(
        output_dir / "shape_count_by_core_count.svg",
        title="Number of Valid TT Shapes by Core Count",
        x_label="Total TT cores",
        y_label="Number of shapes",
        series=[{"label": "valid shapes", "color": "#1f77b4", "points": shape_points}],
    )
    plot_line_series(
        output_dir / "parameter_bounds_by_core_count.svg",
        title="TT-LoRA Parameter Bounds by Core Count",
        x_label="Total TT cores",
        y_label="TT-LoRA parameters",
        series=[
            {"label": "minimum", "color": "#2ca02c", "points": parameter_min_points},
            {"label": "median", "color": "#1f77b4", "points": parameter_median_points},
            {"label": "maximum", "color": "#d62728", "points": parameter_max_points},
        ],
    )
    plot_line_series(
        output_dir / "score_bounds_by_core_count.svg",
        title="Structural Score Bounds by Core Count",
        x_label="Total TT cores",
        y_label="score, lower is better",
        series=[
            {"label": "minimum", "color": "#2ca02c", "points": score_min_points},
            {"label": "median", "color": "#1f77b4", "points": score_median_points},
            {"label": "maximum", "color": "#d62728", "points": score_max_points},
        ],
    )
    plot_line_series(
        output_dir / "max_edge_fraction_by_core_count.svg",
        title="Maximum Edge Parameter Fraction by Core Count",
        x_label="Total TT cores",
        y_label="fraction in first and last cores",
        series=[{"label": "max edge fraction", "color": "#9467bd", "points": edge_points}],
    )

    plotted = sample_scatter(candidates, plot_max_points)
    plot_scatter(
        output_dir / "all_shapes_parameter_scatter.svg",
        title="All Valid TT Shapes: Parameter Count Scatter",
        candidates=candidates,
        plotted_candidates=plotted,
    )

    return {
        "scatter_total_candidates": len(candidates),
        "scatter_plotted_candidates": len(plotted),
        "plot_max_points": plot_max_points,
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate and visualize the possible TT-LoRA parameter, shape, and core-count "
            "space for one weight matrix and one TT rank."
        )
    )
    parser.add_argument("--weight-name", default="c_attn")
    parser.add_argument(
        "--weight-shape",
        nargs=2,
        type=positive_int,
        metavar=("OUT_FEATURES", "IN_FEATURES"),
        default=(2304, 768),
        help="Semantic weight shape [out_features in_features]. Default is GPT-2 small c_attn.",
    )
    parser.add_argument("--rank", type=positive_int, default=6)
    parser.add_argument(
        "--core-counts",
        nargs="*",
        default=None,
        help="Optional total TT core counts. If omitted, feasible counts are inferred.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=("all", "symmetric", "near-symmetric"),
        default="all",
        help="How total cores are split between input-side and output-side factors.",
    )
    parser.add_argument(
        "--allow-one-factors",
        action="store_true",
        help="Allow factors of 1. If enabled, --core-counts must be provided.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for CSV/JSON/SVG outputs. Defaults to ./analysis/<weight>_rank<rank>_<shape>.",
    )
    parser.add_argument(
        "--plot-max-points",
        type=positive_int,
        default=20000,
        help="Maximum scatter points rendered into SVG. CSV/JSON outputs still keep all candidates.",
    )
    parser.add_argument(
        "--write-candidates-json",
        action="store_true",
        help="Also write all candidates to JSON. This can be large; CSV is always written.",
    )
    return parser


def default_output_dir(script_dir: Path, weight_name: str, out_features: int, in_features: int, rank: int) -> Path:
    safe_weight = "".join(char if char.isalnum() or char in "._-" else "_" for char in weight_name).strip("_")
    return script_dir / "analysis" / f"{safe_weight}_out{out_features}_in{in_features}_rank{rank}"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    out_features, in_features = tuple(args.weight_shape)
    core_counts = parse_core_counts(args.core_counts)

    if args.allow_one_factors and core_counts is None:
        raise ValueError("--core-counts must be provided when --allow-one-factors is enabled.")
    if core_counts is None:
        core_counts = infer_core_counts(in_features, out_features, args.split_strategy)

    script_dir = Path(__file__).resolve().parent
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_output_dir(script_dir, args.weight_name, out_features, in_features, args.rank)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = generate_candidates(
        weight_name=args.weight_name,
        out_features=out_features,
        in_features=in_features,
        rank=args.rank,
        core_counts=core_counts,
        split_strategy=args.split_strategy,
        allow_one_factors=args.allow_one_factors,
    )
    summary_rows, selected_rows = summarize_candidates(candidates)

    all_candidates_csv = output_dir / "all_tt_shapes.csv"
    summary_csv = output_dir / "summary_by_core_count.csv"
    selected_csv = output_dir / "selected_shapes_by_core_count.csv"
    metadata_json = output_dir / "metadata.json"

    write_csv(all_candidates_csv, (candidate.to_row() for candidate in candidates))
    write_csv(summary_csv, summary_rows)
    write_csv(selected_csv, selected_rows)

    visualization_metadata = write_visualizations(
        output_dir=output_dir,
        summary_rows=summary_rows,
        candidates=candidates,
        plot_max_points=args.plot_max_points,
    )

    if args.write_candidates_json:
        write_json(output_dir / "all_tt_shapes.json", [candidate.to_row() for candidate in candidates])

    metadata = {
        "weight_name": args.weight_name,
        "weight_shape": [out_features, in_features],
        "rank": args.rank,
        "split_strategy": args.split_strategy,
        "allow_one_factors": args.allow_one_factors,
        "core_counts": list(core_counts),
        "num_candidates": len(candidates),
        "num_core_counts": len(summary_rows),
        "outputs": {
            "all_candidates_csv": str(all_candidates_csv),
            "summary_csv": str(summary_csv),
            "selected_csv": str(selected_csv),
            "metadata_json": str(metadata_json),
            "shape_count_svg": str(output_dir / "shape_count_by_core_count.svg"),
            "parameter_bounds_svg": str(output_dir / "parameter_bounds_by_core_count.svg"),
            "score_bounds_svg": str(output_dir / "score_bounds_by_core_count.svg"),
            "edge_fraction_svg": str(output_dir / "max_edge_fraction_by_core_count.svg"),
            "scatter_svg": str(output_dir / "all_shapes_parameter_scatter.svg"),
        },
        "visualization": visualization_metadata,
    }
    write_json(metadata_json, metadata)

    print(f"weight_name={args.weight_name}")
    print(f"weight_shape=[{out_features}, {in_features}] rank={args.rank}")
    print(f"split_strategy={args.split_strategy} core_counts={list(core_counts)}")
    print(f"generated_candidates={len(candidates):,}")
    print(f"wrote={output_dir}")


if __name__ == "__main__":
    main()
