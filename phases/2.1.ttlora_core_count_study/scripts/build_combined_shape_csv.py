from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path


def parse_key_value_specs(specs: list[str] | None, value_name: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"Invalid {value_name} '{spec}'. Expected format LABEL=VALUE.")
        label, value = spec.split("=", 1)
        label = label.strip()
        value = value.strip()
        if not label or not value:
            raise ValueError(f"Invalid {value_name} '{spec}'. Both LABEL and VALUE are required.")
        mapping[label] = value
    return mapping


def parse_literal(value: str):
    text = value.strip()
    if not text:
        return value
    if not text.startswith(("[", "(", "{", "'", '"')):
        return value
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return value


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def to_int(value: object, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not parse integer for {label}: {value}") from exc


def normalize_source_rows(
    label: str,
    csv_path: Path,
    multiplier: int,
    description: str,
) -> tuple[dict[int, dict[str, object]], dict[str, object]]:
    rows = read_csv_rows(csv_path)
    by_core_count: dict[int, dict[str, object]] = {}
    for row in rows:
        total_cores = to_int(row["total_cores"], f"{csv_path}:total_cores")
        if total_cores in by_core_count:
            raise ValueError(f"Duplicate total_cores={total_cores} found in {csv_path}")

        normalized = dict(row)
        for key in (
            "weight_shape",
            "input_factors",
            "output_factors",
            "tt_shape",
        ):
            if key in normalized and isinstance(normalized[key], str):
                normalized[key] = parse_literal(str(normalized[key]))
        by_core_count[total_cores] = normalized

    metadata_path = csv_path.with_name("parameter_space_metadata.json")
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    summary = {
        "label": label,
        "csv_path": str(csv_path),
        "folder_path": str(csv_path.parent),
        "multiplier": multiplier,
        "description": description,
        "rank": metadata.get("rank"),
        "weight_shape": metadata.get("weight_shape"),
        "min_total_cores": min(by_core_count),
        "max_total_cores": max(by_core_count),
        "num_rows": len(by_core_count),
    }
    return by_core_count, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine multiple lowest_parameter_shapes_by_core_count.csv files into one shared CSV."
    )
    parser.add_argument(
        "--source-spec",
        action="append",
        required=True,
        help="Repeatable LABEL=/path/to/lowest_parameter_shapes_by_core_count.csv mapping.",
    )
    parser.add_argument(
        "--multiplier",
        action="append",
        default=None,
        help="Optional repeatable LABEL=INT mapping. Defaults to 1.",
    )
    parser.add_argument(
        "--description",
        action="append",
        default=None,
        help="Optional repeatable LABEL=TEXT mapping.",
    )
    parser.add_argument(
        "--join-mode",
        choices=("inner", "outer"),
        default="inner",
        help="Use inner intersection or outer union of total_cores across sources.",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--output-metadata-json",
        default=None,
        help="Optional metadata JSON summary path.",
    )
    args = parser.parse_args()

    source_specs = parse_key_value_specs(args.source_spec, "--source-spec")
    multipliers = {key: int(value) for key, value in parse_key_value_specs(args.multiplier, "--multiplier").items()}
    descriptions = parse_key_value_specs(args.description, "--description")

    normalized_sources: dict[str, dict[int, dict[str, object]]] = {}
    source_summaries: list[dict[str, object]] = []
    for label, raw_csv_path in source_specs.items():
        csv_path = Path(raw_csv_path).expanduser().resolve()
        multiplier = multipliers.get(label, 1)
        description = descriptions.get(label, "")
        rows_by_core_count, summary = normalize_source_rows(label, csv_path, multiplier, description)
        normalized_sources[label] = rows_by_core_count
        source_summaries.append(summary)

    all_core_sets = [set(rows.keys()) for rows in normalized_sources.values()]
    if args.join_mode == "inner":
        core_counts = sorted(set.intersection(*all_core_sets))
    else:
        core_counts = sorted(set.union(*all_core_sets))
    if not core_counts:
        raise ValueError("No core counts remain after combining the provided sources.")

    num_layers_values = {
        to_int(next(iter(rows.values()))["num_layers"], "num_layers")
        for rows in normalized_sources.values()
    }
    if len(num_layers_values) != 1:
        raise ValueError(f"All sources must share the same num_layers. Found: {sorted(num_layers_values)}")
    shared_num_layers = next(iter(num_layers_values))

    output_rows: list[dict[str, object]] = []
    for total_cores in core_counts:
        combined_row: dict[str, object] = {
            "total_cores": total_cores,
            "combined_num_sources": len(normalized_sources),
            "combined_num_layers": shared_num_layers,
        }
        combined_adapted_weights_per_layer = 0
        combined_per_layer_params = 0

        missing_labels: list[str] = []
        for label, rows_by_core_count in normalized_sources.items():
            source_row = rows_by_core_count.get(total_cores)
            if source_row is None:
                missing_labels.append(label)
                continue
            multiplier = multipliers.get(label, 1)
            combined_adapted_weights_per_layer += multiplier
            per_matrix_params = to_int(source_row["per_matrix_params"], f"{label}:per_matrix_params")
            combined_per_layer_params += per_matrix_params * multiplier

            for key, value in source_row.items():
                if key == "total_cores":
                    continue
                combined_row[f"{label}_{key}"] = json.dumps(value) if isinstance(value, (list, tuple, dict)) else value
            combined_row[f"{label}_multiplier"] = multiplier
            combined_row[f"{label}_combined_per_layer_params"] = per_matrix_params * multiplier
            combined_row[f"{label}_combined_all_layers_params"] = per_matrix_params * multiplier * shared_num_layers
            combined_row[f"{label}_source_csv"] = source_summaries[[s["label"] for s in source_summaries].index(label)]["csv_path"]
            combined_row[f"{label}_source_folder"] = source_summaries[[s["label"] for s in source_summaries].index(label)]["folder_path"]
            combined_row[f"{label}_description"] = descriptions.get(label, "")

        if args.join_mode == "outer" and missing_labels:
            combined_row["missing_source_labels"] = ",".join(sorted(missing_labels))
        elif missing_labels:
            raise ValueError(
                f"Internal error: join_mode=inner should not leave missing labels, but total_cores={total_cores} "
                f"is missing {missing_labels}."
            )

        combined_row["combined_adapted_weights_per_layer"] = combined_adapted_weights_per_layer
        combined_row["combined_per_layer_params"] = combined_per_layer_params
        combined_row["combined_all_layers_params"] = combined_per_layer_params * shared_num_layers
        output_rows.append(combined_row)

    output_csv_path = Path(args.output_csv).expanduser().resolve()
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    for row in output_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    preferred_front = [
        "total_cores",
        "combined_num_sources",
        "combined_num_layers",
        "combined_adapted_weights_per_layer",
        "combined_per_layer_params",
        "combined_all_layers_params",
    ]
    fieldnames = [name for name in preferred_front if name in fieldnames] + [
        name for name in fieldnames if name not in preferred_front
    ]

    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    metadata = {
        "output_csv_path": str(output_csv_path),
        "join_mode": args.join_mode,
        "num_rows": len(output_rows),
        "min_total_cores": min(core_counts),
        "max_total_cores": max(core_counts),
        "combined_num_layers": shared_num_layers,
        "sources": source_summaries,
    }
    if args.output_metadata_json:
        output_metadata_path = Path(args.output_metadata_json).expanduser().resolve()
        output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        output_metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote combined CSV to {output_csv_path}")
    if args.output_metadata_json:
        print(f"Wrote combined metadata to {Path(args.output_metadata_json).expanduser().resolve()}")


if __name__ == "__main__":
    main()
