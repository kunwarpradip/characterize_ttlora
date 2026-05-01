from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from ast import literal_eval


DEFAULT_SEEDS = (647761,)
DEFAULT_VARIANTS = ("contraction", "reconstruction")
GENERATION_DATASETS = frozenset(
    {
        "ptb",
        "enron",
        "gsm8k",
        "cnn",
        "cnn_dailymail",
        "cnn_daily_mail",
        "cnn/dailymail",
    }
)
GENERATION_SUPPORTED_WEIGHTS = frozenset(
    {
        "c_attn",
        "c_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "wq",
        "wk",
        "wv",
        "wo",
    }
)


def format_lr(lr: float) -> str:
    return f"{lr:.0e}".replace("+0", "").replace("+", "")


def dataset_task_type(dataset_name: str) -> str:
    return "generation" if dataset_name.lower() in GENERATION_DATASETS else "classification"


def resolve_requested_device(device: str, gpu_id: int | None) -> str:
    return f"cuda:{gpu_id}" if gpu_id is not None else device


def normalize_user_layer_indices(layers: list[int] | tuple[int, ...] | None) -> tuple[int, ...] | None:
    if not layers:
        return None
    normalized: list[int] = []
    for layer in layers:
        if int(layer) < 1:
            raise ValueError(
                f"Layer indices are 1-based in this runner. Received invalid layer index: {layer}"
            )
        normalized.append(int(layer) - 1)
    return tuple(normalized)


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
 

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_execution_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_shape_candidates(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"No TT-shape candidates found in {path}")
    required_keys = {
        "total_cores",
        "input_cores",
        "output_cores",
        "input_factors",
        "output_factors",
        "tt_shape",
    }
    for index, candidate in enumerate(candidates):
        missing = required_keys.difference(candidate)
        if missing:
            raise ValueError(
                f"Candidate {index} in {path} is missing required keys: {sorted(missing)}"
            )
    return candidates


def parse_int_sequence(value: str, label: str) -> tuple[int, ...]:
    try:
        parsed = literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Could not parse {label}: {value}") from exc
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"{label} must parse to a list/tuple, got: {value}")
    return tuple(int(item) for item in parsed)


def load_lowest_parameter_shapes_csv(
    path: Path,
    weight_shape: tuple[int, int] | None,
) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")

    candidates: list[dict] = []
    for index, row in enumerate(rows):
        candidate_weight_shape = (
            int(row["out_features"]),
            int(row["in_features"]),
        )
        if weight_shape is not None and candidate_weight_shape != weight_shape:
            continue
        try:
            input_factors = parse_int_sequence(row["input_factors"], "input_factors")
            output_factors = parse_int_sequence(row["output_factors"], "output_factors")
            tt_shape = parse_int_sequence(row["tt_shape"], "tt_shape")
        except KeyError as exc:
            raise ValueError(f"Row {index} in {path} is missing expected column {exc}") from exc

        candidates.append(
            {
                "total_cores": int(row["total_cores"]),
                "input_cores": int(row["input_cores"]),
                "output_cores": int(row["output_cores"]),
                "input_factors": input_factors,
                "output_factors": output_factors,
                "tt_shape": tt_shape,
                "weight_shape": candidate_weight_shape,
                "all_layers_qkv_params": int(row["all_layers_qkv_params"]),
            }
        )

    if not candidates:
        if weight_shape is None:
            raise ValueError(f"No TT-shape candidates found in {path}")
        raise ValueError(
            f"No TT-shape candidates found in {path} for weight shape {list(weight_shape)}"
        )
    return candidates


def validate_shape_csv_metadata(
    csv_path: Path,
    ttlora_rank: int,
    weight_shape: tuple[int, int] | None,
) -> None:
    metadata_path = csv_path.with_name("parameter_space_metadata.json")
    if not metadata_path.exists():
        return
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return

    metadata_rank = metadata.get("rank")
    if metadata_rank is not None and int(metadata_rank) != int(ttlora_rank):
        raise ValueError(
            f"CSV metadata rank {metadata_rank} does not match --ttlora-rank {ttlora_rank}. "
            f"CSV: {csv_path}"
        )

    metadata_weight_shape = metadata.get("weight_shape")
    if weight_shape is not None and metadata_weight_shape is not None:
        normalized_metadata_shape = tuple(int(item) for item in metadata_weight_shape)
        if normalized_metadata_shape != weight_shape:
            raise ValueError(
                f"CSV metadata weight shape {list(normalized_metadata_shape)} does not match "
                f"--weight-shape {list(weight_shape)}. CSV: {csv_path}"
            )


def parse_generation_weight_specs(specs: list[str] | None) -> dict[str, Path]:
    if not specs:
        return {}
    mapping: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --generation-weight-spec value '{spec}'. Expected format WEIGHT_NAME=/path/to/csv."
            )
        weight_name, csv_path = spec.split("=", 1)
        weight_name = weight_name.strip().lower()
        if weight_name not in GENERATION_SUPPORTED_WEIGHTS:
            supported = ", ".join(sorted(GENERATION_SUPPORTED_WEIGHTS))
            raise ValueError(f"Unsupported generation target weight '{weight_name}'. Supported weights: {supported}.")
        mapping[weight_name] = Path(csv_path.strip()).expanduser().resolve()
    return mapping


def parse_generation_combined_weight_groups(specs: list[str] | None) -> dict[str, tuple[str, ...]]:
    if not specs:
        return {}
    mapping: dict[str, tuple[str, ...]] = {}
    used_weights: set[str] = set()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --generation-combined-weight-group value '{spec}'. "
                "Expected format GROUP_LABEL=weight1,weight2,..."
            )
        group_label, weights_spec = spec.split("=", 1)
        group_label = group_label.strip()
        if not group_label:
            raise ValueError(f"Invalid --generation-combined-weight-group value '{spec}'. Group label is empty.")
        weights = tuple(weight.strip().lower() for weight in weights_spec.split(",") if weight.strip())
        if not weights:
            raise ValueError(
                f"Invalid --generation-combined-weight-group value '{spec}'. "
                "At least one target weight is required."
            )
        unsupported = [weight for weight in weights if weight not in GENERATION_SUPPORTED_WEIGHTS]
        if unsupported:
            supported = ", ".join(sorted(GENERATION_SUPPORTED_WEIGHTS))
            raise ValueError(
                f"Unsupported generation target weights {unsupported} in group '{group_label}'. "
                f"Supported weights: {supported}."
            )
        overlap = sorted(set(weights).intersection(used_weights))
        if overlap:
            raise ValueError(
                f"Target weights {overlap} were assigned more than once across "
                "--generation-combined-weight-group arguments."
            )
        used_weights.update(weights)
        mapping[group_label] = weights
    return mapping


def load_generation_weight_bundles(
    weight_csvs: dict[str, Path],
    ttlora_rank: int,
    core_counts: list[int] | None,
) -> tuple[list[dict], dict[str, str]]:
    if not weight_csvs:
        return [], {}

    per_weight_candidates: dict[str, dict[int, dict]] = {}
    metadata_sources: dict[str, str] = {}
    for weight_name, csv_path in weight_csvs.items():
        validate_shape_csv_metadata(csv_path, ttlora_rank, None)
        candidates = load_lowest_parameter_shapes_csv(csv_path, None)
        per_weight_candidates[weight_name] = {
            int(candidate["total_cores"]): candidate for candidate in candidates
        }
        metadata_sources[weight_name] = str(csv_path)

    shared_core_counts = set.intersection(
        *(set(candidate_map.keys()) for candidate_map in per_weight_candidates.values())
    )
    if core_counts:
        shared_core_counts &= {int(item) for item in core_counts}
    if not shared_core_counts:
        raise ValueError(
            "No shared total core counts were found across the provided generation weight CSVs."
        )

    bundles: list[dict] = []
    for total_cores in sorted(shared_core_counts):
        bundle_weights: dict[str, dict] = {}
        for weight_name, candidate_map in per_weight_candidates.items():
            bundle_weights[weight_name] = candidate_map[total_cores]
        bundles.append(
            {
                "total_cores": total_cores,
                "weights": bundle_weights,
            }
        )
    return bundles, metadata_sources


def load_combined_generation_weight_bundles(
    combined_csv_path: Path,
    combined_weight_groups: dict[str, tuple[str, ...]],
    core_counts: list[int] | None,
) -> tuple[list[dict], dict[str, str]]:
    with combined_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {combined_csv_path}")

    if not combined_weight_groups:
        raise ValueError(
            "Combined generation CSV support requires at least one "
            "--generation-combined-weight-group GROUP=weight1,weight2,... argument."
        )

    metadata_sources: dict[str, str] = {}
    bundles: list[dict] = []
    allowed_core_counts = {int(item) for item in core_counts} if core_counts else None
    seen_core_counts: set[int] = set()

    for index, row in enumerate(rows):
        total_cores = int(row["total_cores"])
        if allowed_core_counts is not None and total_cores not in allowed_core_counts:
            continue
        if total_cores in seen_core_counts:
            raise ValueError(
                f"Combined CSV {combined_csv_path} contains duplicate total_cores={total_cores} rows."
            )
        seen_core_counts.add(total_cores)

        bundle_weights: dict[str, dict] = {}
        for group_label, target_weights in combined_weight_groups.items():
            required_cols = (
                f"{group_label}_tt_shape",
                f"{group_label}_input_factors",
                f"{group_label}_output_factors",
                f"{group_label}_weight_shape",
            )
            missing = [col for col in required_cols if col not in row]
            if missing:
                raise ValueError(
                    f"Combined CSV {combined_csv_path} is missing columns for group '{group_label}': {missing}"
                )

            tt_shape = parse_int_sequence(row[f"{group_label}_tt_shape"], f"{group_label}_tt_shape")
            input_factors = parse_int_sequence(
                row[f"{group_label}_input_factors"],
                f"{group_label}_input_factors",
            )
            output_factors = parse_int_sequence(
                row[f"{group_label}_output_factors"],
                f"{group_label}_output_factors",
            )
            weight_shape = parse_int_sequence(
                row[f"{group_label}_weight_shape"],
                f"{group_label}_weight_shape",
            )
            if len(weight_shape) != 2:
                raise ValueError(
                    f"Expected two values in {group_label}_weight_shape for total_cores={total_cores}, "
                    f"got {list(weight_shape)}"
                )

            source_csv = row.get(f"{group_label}_source_csv", "") or str(combined_csv_path)
            for weight_name in target_weights:
                bundle_weights[weight_name] = {
                    "total_cores": total_cores,
                    "input_cores": int(row.get(f"{group_label}_input_cores") or len(input_factors)),
                    "output_cores": int(row.get(f"{group_label}_output_cores") or len(output_factors)),
                    "input_factors": input_factors,
                    "output_factors": output_factors,
                    "tt_shape": tt_shape,
                    "weight_shape": tuple(int(item) for item in weight_shape),
                }
                metadata_sources[weight_name] = source_csv

        bundles.append({"total_cores": total_cores, "weights": bundle_weights})

    if allowed_core_counts is not None:
        missing_core_counts = sorted(allowed_core_counts.difference(seen_core_counts))
        if missing_core_counts:
            raise ValueError(
                f"Combined CSV {combined_csv_path} does not contain requested core counts: {missing_core_counts}"
            )
    if not bundles:
        raise ValueError(f"No TT-shape candidates found in combined CSV {combined_csv_path}")

    bundles.sort(key=lambda item: int(item["total_cores"]))
    return bundles, metadata_sources


def build_parser() -> argparse.ArgumentParser:
    phase_root = Path(__file__).resolve().parents[1]
    project_root = phase_root.parents[1]

    parser = argparse.ArgumentParser(
        description=(
            "Run the TT-LoRA core-count study using the canonical merge-lineage TT shapes."
        )
    )
    parser.add_argument("--suite-name", required=True)
    parser.add_argument("--phase-root", default=str(phase_root))
    parser.add_argument("--project-root", default=str(project_root))
    parser.add_argument(
        "--shape-json",
        default=str(phase_root / "analysis" / "tt_shape.json"),
        help="Path to the merge-lineage TT-shape JSON generated for this phase.",
    )
    parser.add_argument(
        "--shape-csv",
        default=None,
        help=(
            "Path to a notebook-produced lowest_parameter_shapes_by_core_count.csv file. "
            "When provided, this is used instead of --shape-json."
        ),
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--train-script", default=str(project_root / "train.py"))
    parser.add_argument(
        "--generation-train-script",
        default=str(project_root / "train_generation.py"),
        help="Local GPT-2 training entrypoint inside characterize_ttlora for ptb/enron runs.",
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--generation-weight-spec",
        action="append",
        default=None,
        help=(
            "Repeatable mapping from GPT-2 weight name to the CSV containing lowest-parameter TT shapes, "
            "e.g. --generation-weight-spec c_attn=/path/to/c_attn.csv "
            "--generation-weight-spec c_proj=/path/to/c_proj.csv"
        ),
    )
    parser.add_argument(
        "--generation-combined-shape-csv",
        default=None,
        help=(
            "Path to a combined lowest-parameter CSV produced by the notebook, where each row already "
            "contains grouped TT shapes such as kv_* and qo_* columns."
        ),
    )
    parser.add_argument(
        "--generation-combined-weight-group",
        action="append",
        default=None,
        help=(
            "Repeatable mapping from a combined CSV group label to concrete target weights, "
            "e.g. --generation-combined-weight-group kv=k_proj,v_proj "
            "--generation-combined-weight-group qo=q_proj,o_proj"
        ),
    )

    parser.add_argument("--dataset-name", default="mrpc")
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        default=None,
        help="Optional multi-dataset sweep. Overrides --dataset-name when provided.",
    )
    parser.add_argument("--dataset-root", default=str(project_root / "datasets"))
    parser.add_argument("--model-path", default=str(project_root / "roberta-base" / "checkpoints"))
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument(
        "--generation-model-path",
        default=str(project_root / "gpt2small" / "checkpoints"),
        help="Local GPT-2 checkpoint directory for ptb/enron datasets.",
    )
    parser.add_argument("--generation-tokenizer-path", default=None)

    parser.add_argument("--core-counts", nargs="*", type=int, default=None)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument(
        "--two-core-learning-rate",
        type=float,
        default=2e-4,
        help="Override learning rate for the 2-core setting.",
    )
    parser.add_argument("--lr-scheduler", default="none")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--patience",
        type=int,
        default=1000,
        help="Set high to avoid early stopping so the 200-epoch curves are fully visible.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="Convenience override for selecting a specific CUDA device, e.g. --gpu-id 1 -> cuda:1.",
    )
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--step-metrics-every", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--generation-max-length",
        type=int,
        default=1024,
        help="Sequence length to use for GPT-2 generation datasets.",
    )
    parser.add_argument(
        "--generation-training-format",
        default="blocks",
        choices=("blocks", "prompt_completion"),
        help="Generation data layout to pass to train_generation.py.",
    )
    parser.add_argument(
        "--generation-eval-max-new-tokens",
        type=int,
        default=256,
        help="Row-level generation evaluation length to pass to train_generation.py.",
    )

    parser.add_argument("--target-modules", nargs="+", default=("key", "query", "value"))
    parser.add_argument(
        "--adapt-layers",
        nargs="*",
        type=int,
        default=None,
        help=(
            "1-based layer indices to adapt. Example: 1 2 3 ... 12 for all layers, or 5 6 to adapt only "
            "the 5th and 6th layers."
        ),
    )
    parser.add_argument("--ttlora-rank", type=int, default=6)
    parser.add_argument("--ttlora-alpha", type=float, default=8.0)
    parser.add_argument(
        "--weight-shape",
        nargs=2,
        type=int,
        default=None,
        metavar=("OUT_FEATURES", "IN_FEATURES"),
        help="Filter CSV-based TT shapes to a specific weight shape.",
    )

    parser.add_argument(
        "--summary-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep disabled to retain history.csv and step_history.csv for curve analysis.",
    )
    parser.add_argument("--overwrite-run-dir", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs already marked successful in the suite execution log.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    resolved_device = resolve_requested_device(args.device, args.gpu_id)

    phase_root = Path(args.phase_root).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else phase_root / "suites"
    runs_root = Path(args.runs_root).expanduser().resolve() if args.runs_root else phase_root / "runs"
    suite_dir = output_root / args.suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    datasets = args.dataset_names if args.dataset_names else [args.dataset_name]
    dataset_task_types = {dataset_name: dataset_task_type(dataset_name) for dataset_name in datasets}
    has_classification_dataset = any(task == "classification" for task in dataset_task_types.values())
    resolved_adapt_layers = normalize_user_layer_indices(args.adapt_layers)
    generation_weight_csvs = parse_generation_weight_specs(args.generation_weight_spec)
    generation_combined_weight_groups = parse_generation_combined_weight_groups(
        args.generation_combined_weight_group
    )
    if args.generation_combined_shape_csv and generation_weight_csvs:
        raise ValueError(
            "Use either --generation-weight-spec or --generation-combined-shape-csv, not both."
        )
    if generation_combined_weight_groups and not args.generation_combined_shape_csv:
        raise ValueError(
            "--generation-combined-weight-group requires --generation-combined-shape-csv."
        )
    if args.generation_combined_shape_csv:
        generation_weight_bundles, generation_weight_sources = load_combined_generation_weight_bundles(
            Path(args.generation_combined_shape_csv).expanduser().resolve(),
            generation_combined_weight_groups,
            args.core_counts,
        )
    else:
        generation_weight_bundles, generation_weight_sources = load_generation_weight_bundles(
            generation_weight_csvs,
            args.ttlora_rank,
            args.core_counts,
        )

    shape_source_path: Path | None = None
    candidates: list[dict] = []
    if has_classification_dataset:
        if args.shape_csv:
            shape_source_path = Path(args.shape_csv).expanduser().resolve()
            validate_shape_csv_metadata(
                shape_source_path,
                args.ttlora_rank,
                tuple(args.weight_shape) if args.weight_shape else None,
            )
            candidates = load_lowest_parameter_shapes_csv(
                shape_source_path,
                tuple(args.weight_shape) if args.weight_shape else None,
            )
        else:
            shape_source_path = Path(args.shape_json).expanduser().resolve()
            candidates = load_shape_candidates(shape_source_path)
        if args.core_counts:
            allowed = set(args.core_counts)
            candidates = [candidate for candidate in candidates if int(candidate["total_cores"]) in allowed]
            if not candidates:
                raise ValueError(
                    f"No TT-shape candidates matched --core-counts {sorted(allowed)} in {shape_source_path}"
                )

    execution_log_path = suite_dir / "execution_log.csv"
    execution_rows = load_execution_log(execution_log_path)
    completed_ok = {row["run_name"] for row in execution_rows if str(row.get("returncode")) == "0"}

    run_specs: list[dict] = []
    for dataset_name in datasets:
        task_type = dataset_task_type(dataset_name)
        if task_type == "generation" and not generation_weight_bundles:
            raise ValueError(
                "Generation datasets require per-weight TT-shape CSVs. "
                "Provide them with repeated --generation-weight-spec WEIGHT=/path/to/csv values."
            )
        if task_type == "generation" and generation_weight_bundles:
            candidate_entries = generation_weight_bundles
        else:
            candidate_entries = candidates
        for variant in args.variants:
            if variant not in {"contraction", "reconstruction"}:
                raise ValueError(f"Unsupported TT-LoRA variant: {variant}")
            for candidate in candidate_entries:
                if task_type == "generation" and generation_weight_bundles:
                    total_cores = int(candidate["total_cores"])
                    input_cores = None
                    output_cores = None
                    input_factors = tuple()
                    output_factors = tuple()
                    tt_shape = tuple()
                    generation_weight_candidates = candidate["weights"]
                else:
                    total_cores = int(candidate["total_cores"])
                    input_cores = int(candidate["input_cores"])
                    output_cores = int(candidate["output_cores"])
                    input_factors = tuple(int(item) for item in candidate["input_factors"])
                    output_factors = tuple(int(item) for item in candidate["output_factors"])
                    tt_shape = tuple(int(item) for item in candidate["tt_shape"])
                    generation_weight_candidates = {}
                learning_rate = (
                    args.two_core_learning_rate if total_cores == 2 else args.learning_rate
                )

                for seed in args.seeds:
                    dataset_runs_root = runs_root / f"{dataset_name}_{variant}"
                    dataset_runs_root.mkdir(parents=True, exist_ok=True)
                    run_name = "_".join(
                        [
                            "ttcore",
                            args.suite_name,
                            dataset_name,
                            variant,
                            f"cores{total_cores}",
                            f"rank{args.ttlora_rank}",
                            f"lr{format_lr(learning_rate)}",
                            f"seed{seed}",
                        ]
                    )
                    notes = (
                        f"phase=ttlora_core_count_study suite={args.suite_name} "
                        f"dataset={dataset_name} variant={variant} total_cores={total_cores}"
                    )

                    spec_cwd = str(project_root)
                    spec_env = None
                    if task_type == "classification":
                        command = [
                            args.python_bin,
                            args.train_script,
                            "--dataset-name",
                            dataset_name,
                            "--dataset-root",
                            args.dataset_root,
                            "--model-path",
                            args.model_path,
                            "--output-dir",
                            str(dataset_runs_root),
                            "--run-name",
                            run_name,
                            "--max-length",
                            str(args.max_length),
                            "--epochs",
                            str(args.epochs),
                            "--batch-size",
                            str(args.batch_size),
                            "--eval-batch-size",
                            str(args.eval_batch_size),
                            "--gradient-accumulation-steps",
                            str(args.gradient_accumulation_steps),
                            "--learning-rate",
                            str(learning_rate),
                            "--lr-scheduler",
                            args.lr_scheduler,
                            "--weight-decay",
                            str(args.weight_decay),
                            "--warmup-ratio",
                            str(args.warmup_ratio),
                            "--max-grad-norm",
                            str(args.max_grad_norm),
                            "--num-workers",
                            str(args.num_workers),
                            "--seed",
                            str(seed),
                            "--patience",
                            str(args.patience),
                            "--device",
                            resolved_device,
                            "--log-every-steps",
                            str(args.log_every_steps),
                            "--step-metrics-every",
                            str(args.step_metrics_every),
                            "--adaptation-method",
                            "ttlora",
                            "--ttlora-variant",
                            variant,
                            "--ttlora-rank",
                            str(args.ttlora_rank),
                            "--ttlora-alpha",
                            str(args.ttlora_alpha),
                            "--ttlora-shape",
                            *[str(item) for item in tt_shape],
                            "--ttlora-input-factors",
                            *[str(item) for item in input_factors],
                            "--ttlora-output-factors",
                            *[str(item) for item in output_factors],
                            "--target-modules",
                            *args.target_modules,
                            "--notes",
                            notes,
                        ]
                        if args.tokenizer_path:
                            command.extend(["--tokenizer-path", args.tokenizer_path])
                        if args.summary_only:
                            command.append("--summary-only")
                        else:
                            command.append("--no-summary-only")
                        if resolved_adapt_layers:
                            command.extend(["--adapt-layers", *[str(item) for item in resolved_adapt_layers]])
                        if args.overwrite_run_dir:
                            command.append("--overwrite-run-dir")
                    else:
                        weight_config_dir = suite_dir / "weight_configs"
                        weight_config_dir.mkdir(parents=True, exist_ok=True)
                        weight_config_path = weight_config_dir / f"{run_name}.json"
                        save_json(
                            weight_config_path,
                            {
                                "dataset_name": dataset_name,
                                "suite_name": args.suite_name,
                                "total_cores": total_cores,
                                "weights": {
                                    weight_name: {
                                        "tt_shape": list(weight_candidate["tt_shape"]),
                                        "input_factors": list(weight_candidate["input_factors"]),
                                        "output_factors": list(weight_candidate["output_factors"]),
                                        "weight_shape": list(weight_candidate["weight_shape"]),
                                    }
                                    for weight_name, weight_candidate in generation_weight_candidates.items()
                                },
                            },
                        )
                        command = [
                            args.python_bin,
                            args.generation_train_script,
                            "--dataset-name",
                            dataset_name,
                            "--dataset-root",
                            args.dataset_root,
                            "--model-path",
                            args.generation_model_path,
                            "--output-dir",
                            str(dataset_runs_root),
                            "--run-name",
                            run_name,
                            "--max-length",
                            str(args.generation_max_length),
                            "--training-format",
                            args.generation_training_format,
                            "--epochs",
                            str(args.epochs),
                            "--batch-size",
                            str(args.batch_size),
                            "--eval-batch-size",
                            str(args.eval_batch_size),
                            "--gradient-accumulation-steps",
                            str(args.gradient_accumulation_steps),
                            "--learning-rate",
                            str(learning_rate),
                            "--lr-scheduler",
                            args.lr_scheduler,
                            "--weight-decay",
                            str(args.weight_decay),
                            "--warmup-ratio",
                            str(args.warmup_ratio),
                            "--max-grad-norm",
                            str(args.max_grad_norm),
                            "--num-workers",
                            str(args.num_workers),
                            "--seed",
                            str(seed),
                            "--patience",
                            str(args.patience),
                            "--device",
                            resolved_device,
                            "--log-every-steps",
                            str(args.log_every_steps),
                            "--step-metrics-every",
                            str(args.step_metrics_every),
                            "--generation-eval-max-new-tokens",
                            str(args.generation_eval_max_new_tokens),
                            "--ttlora-rank",
                            str(args.ttlora_rank),
                            "--ttlora-alpha",
                            str(args.ttlora_alpha),
                            "--ttlora-variant",
                            variant,
                            "--ttlora-weight-config",
                            str(weight_config_path),
                            "--notes",
                            notes,
                        ]
                        if args.generation_tokenizer_path:
                            command.extend(["--tokenizer-path", args.generation_tokenizer_path])
                        if args.summary_only:
                            command.append("--summary-only")
                        else:
                            command.append("--no-summary-only")
                        if resolved_adapt_layers:
                            command.extend(["--adapt-layers", *[str(item) for item in resolved_adapt_layers]])
                        if args.overwrite_run_dir:
                            command.append("--overwrite-run-dir")

                    run_specs.append(
                        {
                            "run_name": run_name,
                            "suite_name": args.suite_name,
                            "dataset_name": dataset_name,
                            "task_type": task_type,
                            "ttlora_variant": variant,
                            "ttlora_rank": args.ttlora_rank,
                            "ttlora_alpha": args.ttlora_alpha,
                            "learning_rate": learning_rate,
                            "seed": seed,
                            "total_cores": total_cores,
                            "input_cores": input_cores,
                            "output_cores": output_cores,
                            "input_factors": list(input_factors),
                            "output_factors": list(output_factors),
                            "tt_shape": list(tt_shape),
                            "generation_weight_candidates": {
                                weight_name: {
                                    "input_factors": list(weight_candidate["input_factors"]),
                                    "output_factors": list(weight_candidate["output_factors"]),
                                    "tt_shape": list(weight_candidate["tt_shape"]),
                                    "weight_shape": list(weight_candidate["weight_shape"]),
                                }
                                for weight_name, weight_candidate in generation_weight_candidates.items()
                            },
                            "dataset_runs_root": str(dataset_runs_root),
                            "cwd": spec_cwd,
                            "env": spec_env,
                            "command": command,
                        }
                    )

    manifest = {
        "suite_name": args.suite_name,
        "phase_root": str(phase_root),
        "shape_source": str(shape_source_path) if shape_source_path is not None else None,
        "shape_source_type": "csv" if args.shape_csv else "json",
        "dataset_name": args.dataset_name,
        "dataset_names": list(datasets),
        "dataset_task_types": dataset_task_types,
        "model_path": args.model_path,
        "generation_model_path": args.generation_model_path,
        "generation_train_script": args.generation_train_script,
        "generation_weight_sources": generation_weight_sources if generation_weight_sources else None,
        "generation_combined_shape_csv": (
            str(Path(args.generation_combined_shape_csv).expanduser().resolve())
            if args.generation_combined_shape_csv
            else None
        ),
        "generation_combined_weight_groups": (
            {key: list(value) for key, value in generation_combined_weight_groups.items()}
            if generation_combined_weight_groups
            else None
        ),
        "target_modules": list(args.target_modules),
        "adapt_layers_user": list(args.adapt_layers) if args.adapt_layers else None,
        "adapt_layers_zero_based": list(resolved_adapt_layers) if resolved_adapt_layers else None,
        "variants": list(args.variants),
        "seeds": list(args.seeds),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "two_core_learning_rate": args.two_core_learning_rate,
        "lr_scheduler": args.lr_scheduler,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "num_workers": args.num_workers,
        "patience": args.patience,
        "device": resolved_device,
        "gpu_id": args.gpu_id,
        "log_every_steps": args.log_every_steps,
        "step_metrics_every": args.step_metrics_every,
        "weight_shape": list(args.weight_shape) if args.weight_shape else None,
        "generation_training_format": args.generation_training_format,
        "generation_eval_max_new_tokens": args.generation_eval_max_new_tokens,
        "summary_only": args.summary_only,
        "runs_root": str(runs_root),
        "num_runs": len(run_specs),
        "run_specs": [
            {
                **{key: value for key, value in spec.items() if key not in {"command", "env"}},
                "command": spec["command"],
            }
            for spec in run_specs
        ],
    }
    save_json(suite_dir / "manifest.json", manifest)

    for spec in run_specs:
        if args.resume and spec["run_name"] in completed_ok:
            print(f"[resume] Skipping completed run {spec['run_name']}", flush=True)
            continue

        print(f"[run] {spec['run_name']}", flush=True)
        print(" ".join(spec["command"]), flush=True)

        if args.dry_run:
            continue

        started_at = time.time()
        result = subprocess.run(
            spec["command"],
            check=False,
            cwd=spec.get("cwd"),
            env=spec.get("env"),
        )
        elapsed_seconds = time.time() - started_at

        summary_path = Path(spec["dataset_runs_root"]) / spec["run_name"] / "summary.json"
        summary: dict = {}
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                summary = {}

        execution_rows.append(
            {
                "run_name": spec["run_name"],
                "suite_name": args.suite_name,
                "dataset_name": spec["dataset_name"],
                "task_type": spec["task_type"],
                "ttlora_variant": spec["ttlora_variant"],
                "ttlora_rank": spec["ttlora_rank"],
                "ttlora_alpha": spec["ttlora_alpha"],
                "learning_rate": spec["learning_rate"],
                "seed": spec["seed"],
                "total_cores": spec["total_cores"],
                "input_cores": spec["input_cores"],
                "output_cores": spec["output_cores"],
                "input_factors": json.dumps(spec["input_factors"]),
                "output_factors": json.dumps(spec["output_factors"]),
                "tt_shape": json.dumps(spec["tt_shape"]),
                "generation_weight_candidates": json.dumps(spec["generation_weight_candidates"]),
                "returncode": result.returncode,
                "elapsed_seconds": elapsed_seconds,
                "summary_path": str(summary_path) if summary_path.exists() else "",
                "history_path": summary.get("history_path", ""),
                "step_history_path": summary.get("step_history_path", ""),
                "best_checkpoint_dir": summary.get("best_checkpoint_dir", ""),
                "best_epoch": summary.get("best_epoch"),
                "epochs_ran": summary.get("epochs_ran"),
                "best_validation_accuracy": summary.get("best_validation_accuracy"),
                "best_validation_loss": summary.get("best_validation_loss"),
                "best_validation_perplexity": summary.get("best_validation_perplexity"),
                "best_validation_token_accuracy": summary.get("best_validation_token_accuracy"),
                "final_validation_accuracy": summary.get("final_validation_accuracy"),
                "initial_validation_accuracy": summary.get("initial_validation_accuracy"),
                "initial_validation_loss": summary.get("initial_validation_loss"),
                "initial_validation_token_accuracy": summary.get("initial_validation_token_accuracy"),
                "final_validation_loss": summary.get("final_validation_loss"),
                "initial_validation_perplexity": summary.get("initial_validation_perplexity"),
                "final_validation_perplexity": summary.get("final_validation_perplexity"),
                "final_validation_token_accuracy": summary.get("final_validation_token_accuracy"),
                "max_peak_memory_gb": summary.get("max_peak_memory_gb"),
                "avg_epoch_seconds": summary.get("avg_epoch_seconds"),
                "avg_grad_norm": summary.get("avg_grad_norm"),
                "max_grad_norm": summary.get("max_grad_norm"),
                "min_grad_norm": summary.get("min_grad_norm"),
                "avg_clipped_step_fraction": summary.get("avg_clipped_step_fraction"),
                "total_clipped_steps": summary.get("total_clipped_steps"),
                "step_metrics_logged": summary.get("step_metrics_logged"),
                "trainable_parameters": summary.get("trainable_parameters"),
                "frozen_parameters": summary.get("frozen_parameters"),
                "total_parameters": summary.get("total_parameters"),
                "resolved_run_dir": summary.get("resolved_run_dir", ""),
            }
        )
        write_csv(execution_log_path, execution_rows)

        if result.returncode != 0:
            print(
                f"[run-failed] {spec['run_name']} exited with return code {result.returncode}",
                flush=True,
            )

        if result.returncode != 0 and args.stop_on_failure:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
