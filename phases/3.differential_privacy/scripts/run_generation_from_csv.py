from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


def add_compact_default_args(parser: argparse.ArgumentParser, project_root: Path) -> None:
    parser.add_argument("--default-dataset-root", default=None)
    parser.add_argument("--default-output-root", default=None)
    parser.add_argument("--default-ttshape-config-root", default=str(project_root / "ttshape_configs" / "ttlora"))
    parser.add_argument("--default-core-count", type=int, default=6)
    parser.add_argument("--default-ttlora-variant", default="reconstruction", choices=("contraction", "reconstruction"))
    parser.add_argument("--default-seed", type=int, default=647761)
    parser.add_argument("--default-batch-size", type=int, default=32)
    parser.add_argument("--default-eval-batch-size", type=int, default=32)
    parser.add_argument("--default-gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--default-max-length", type=int, default=1024)
    parser.add_argument("--default-training-format", default="blocks", choices=("blocks", "prompt_completion"))
    parser.add_argument("--default-lr-scheduler", default="none")
    parser.add_argument("--default-weight-decay", type=float, default=0.01)
    parser.add_argument("--default-warmup-ratio", type=float, default=0.06)
    parser.add_argument("--default-num-workers", type=int, default=0)
    parser.add_argument("--default-device", default="auto")
    parser.add_argument("--default-log-every-steps", type=int, default=10)
    parser.add_argument("--default-step-metrics-every", type=int, default=1)
    parser.add_argument("--default-dp-poisson-sampling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--default-dp-secure-mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--default-dp-grad-sample-mode", default="hooks", choices=("functorch", "hooks", "ghost"))
    parser.add_argument("--default-generation-eval-max-new-tokens", type=int, default=256)


def parse_args() -> argparse.Namespace:
    project_root = Path("/home/pkunwar/characterize_ttlora")
    parser = argparse.ArgumentParser(
        description=(
            "Run train_generation.py from a CSV file, one row per run. "
            "Column names map to train_generation.py CLI flags."
        )
    )
    parser.add_argument("--csv-path", required=True, help="Path to the input CSV file.")
    parser.add_argument(
        "--project-root",
        default=str(project_root),
        help="Project root containing train_generation.py.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable to use for launching train_generation.py.",
    )
    parser.add_argument(
        "--train-script",
        default=None,
        help="Optional explicit path to train_generation.py. Defaults to <project-root>/train_generation.py.",
    )
    parser.add_argument(
        "--only-run-name",
        action="append",
        default=None,
        help="Repeatable filter. If provided, run only rows whose run_name matches one of these values.",
    )
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip rows whose output_dir/run_name/summary.json already exists.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop at the first failed row. Disable to continue through the CSV.",
    )
    add_compact_default_args(parser, project_root)
    return parser.parse_args()


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip() == ""


def parse_bool(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Could not parse boolean value: {value!r}")


def parse_list(value: str) -> list[str]:
    stripped = value.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        loaded = json.loads(stripped)
        if not isinstance(loaded, list):
            raise ValueError(f"Expected a JSON list, got: {value!r}")
        return [str(item) for item in loaded]
    if "," in stripped:
        return [item.strip() for item in stripped.split(",") if item.strip()]
    return shlex.split(stripped)


def normalize_optional_value(value: Any) -> Any:
    if value is None:
        return None
    stripped = str(value).strip()
    if not stripped:
        return None
    if stripped.lower() in {"none", "null", "full", "all", "default"}:
        return None
    return stripped


def model_paths(project_root: Path, model_name: str) -> tuple[Path, Path]:
    normalized = model_name.strip().lower()
    mapping = {
        "gpt2": project_root / "gpt2small" / "checkpoints",
        "gpt2small": project_root / "gpt2small" / "checkpoints",
        "roberta-base": project_root / "roberta-base" / "checkpoints",
        "roberta_base": project_root / "roberta-base" / "checkpoints",
        "llama3.2-1b": project_root / "llama3.2-1b" / "checkpoints",
        "llama3_2_1b": project_root / "llama3.2-1b" / "checkpoints",
        "llama3-2-1b": project_root / "llama3.2-1b" / "checkpoints",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported model alias '{model_name}'. Add it to model_paths().")
    model_path = mapping[normalized]
    return model_path, model_path


def sanitize_name_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return token.strip("_")


def format_float_token(value: str | float) -> str:
    text = str(value).strip()
    text = text.replace("+", "")
    text = text.replace(".", "p")
    return text


def _canonical_model_slug(model_name: str) -> str:
    normalized = model_name.strip().lower()
    if normalized in {"gpt2", "gpt2small"}:
        return "gpt2small"
    if normalized in {"roberta-base", "roberta_base"}:
        return "roberta-base"
    if normalized in {"llama3.2-1b", "llama3_2_1b", "llama3-2-1b"}:
        return "llama3.2-1b"
    return sanitize_name_token(model_name)


def resolve_ttlora_model_config_path(
    *,
    model_name: str,
    ttshape_config_root: Path,
) -> Path:
    model_slug = _canonical_model_slug(model_name)
    path = ttshape_config_root / f"{model_slug}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find TT shape config for model '{model_name}' at {path}. "
            "Generate it first with build_model_ttlora_config.py or provide ttlora_weight_config explicitly."
        )
    return path


def _parse_ttlora_weight_selection(row: dict[str, str], model_payload: dict[str, Any]) -> list[str]:
    raw_value = (
        normalize_optional_value(row.get("ttlora_weights"))
        or normalize_optional_value(row.get("weights_to_adapt"))
        or normalize_optional_value(row.get("target_weights"))
    )
    if raw_value is None:
        return list(model_payload["supported_weights"].keys())
    requested = parse_list(str(raw_value))
    supported = set(model_payload["supported_weights"].keys())
    unsupported = sorted(set(requested).difference(supported))
    if unsupported:
        raise ValueError(
            f"Unsupported TT-LoRA weights {unsupported}. Supported weights for this model: {sorted(supported)}"
        )
    return requested


def materialize_ttlora_weight_config(
    *,
    model_config_path: Path,
    output_path: Path,
    core_count: int,
    rank: int,
    selected_weights: list[str],
) -> Path:
    payload = json.loads(model_config_path.read_text(encoding="utf-8"))
    if isinstance(payload.get("ranks"), dict):
        rank_payload = payload["ranks"].get(str(rank))
        if rank_payload is None:
            available_ranks = sorted(int(item) for item in payload["ranks"].keys())
            raise ValueError(
                f"No TT shape entry found for rank={rank} in {model_config_path}. Available ranks: {available_ranks}"
            )
        configs = rank_payload.get("configs", [])
    else:
        configs = payload.get("configs", [])

    entry = next((item for item in configs if int(item["core_count"]) == core_count), None)
    if entry is None:
        available = [int(item["core_count"]) for item in configs]
        raise ValueError(
            f"No TT shape entry found for core_count={core_count} in {model_config_path}. Available: {available}"
        )

    weights_payload: dict[str, Any] = {}
    for weight_name in selected_weights:
        weight_entry = entry["weights"].get(weight_name)
        if weight_entry is None:
            raise ValueError(f"Weight '{weight_name}' is not present in core_count={core_count} of {model_config_path}")
        weights_payload[weight_name] = {
            "tt_shape": list(weight_entry["tt_shape"]),
            "input_factors": list(weight_entry["input_factors"]),
            "output_factors": list(weight_entry["output_factors"]),
            "weight_shape": list(weight_entry["weight_shape"]) if weight_entry.get("weight_shape") is not None else None,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"weights": weights_payload}, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def derive_run_name(expanded_row: dict[str, Any]) -> str:
    dataset = sanitize_name_token(str(expanded_row["dataset_name"]))
    model = sanitize_name_token(str(expanded_row["model"]))
    method = sanitize_name_token(str(expanded_row["adaptation_method"]))
    rank = int(
        expanded_row["ttlora_rank"]
        if expanded_row["adaptation_method"] == "ttlora"
        else expanded_row.get("lora_rank", 0) or 0
    )
    lr_token = format_float_token(expanded_row["learning_rate"])
    seed = int(expanded_row["seed"])
    if parse_bool(expanded_row.get("dp_enabled", False)):
        eps_token = format_float_token(expanded_row["dp_target_epsilon"])
        if expanded_row["adaptation_method"] == "ttlora":
            core_count = int(expanded_row["core_count"])
            variant = sanitize_name_token(str(expanded_row["ttlora_variant"]))
            return f"{model}_{dataset}_{method}_{variant}_cores{core_count}_rank{rank}_eps{eps_token}_lr{lr_token}_seed{seed}"
        return f"{model}_{dataset}_{method}_rank{rank}_eps{eps_token}_lr{lr_token}_seed{seed}"
    if expanded_row["adaptation_method"] == "ttlora":
        core_count = int(expanded_row["core_count"])
        variant = sanitize_name_token(str(expanded_row["ttlora_variant"]))
        return f"{model}_{dataset}_{method}_{variant}_cores{core_count}_rank{rank}_lr{lr_token}_seed{seed}"
    return f"{model}_{dataset}_{method}_rank{rank}_lr{lr_token}_seed{seed}"


def expand_row_defaults(row: dict[str, str], args: argparse.Namespace, project_root: Path) -> dict[str, Any]:
    expanded: dict[str, Any] = dict(row)
    model_name = normalize_optional_value(row.get("model")) or normalize_optional_value(row.get("model_name"))
    if model_name is None:
        model_name = normalize_optional_value(row.get("model_path"))
    if model_name is None:
        raise ValueError("Each CSV row must include either 'model' or 'model_name'.")
    expanded["model"] = model_name

    dataset_name = normalize_optional_value(row.get("dataset")) or normalize_optional_value(row.get("dataset_name"))
    if dataset_name is None:
        raise ValueError("Each CSV row must include either 'dataset' or 'dataset_name'.")
    expanded["dataset_name"] = dataset_name

    model_path, tokenizer_path = model_paths(project_root, str(model_name))
    expanded["model_path"] = normalize_optional_value(row.get("model_path")) or str(model_path)
    expanded["tokenizer_path"] = normalize_optional_value(row.get("tokenizer_path")) or str(tokenizer_path)
    expanded["dataset_root"] = (
        normalize_optional_value(row.get("dataset_root"))
        or args.default_dataset_root
        or str(project_root / "datasets")
    )

    expanded["output_dir"] = (
        normalize_optional_value(row.get("output_dir"))
        or args.default_output_root
        or str(project_root / "phases/3.differential_privacy/runs")
    )
    expanded["max_length"] = normalize_optional_value(row.get("max_length")) or str(args.default_max_length)
    expanded["training_format"] = normalize_optional_value(row.get("training_format")) or args.default_training_format
    expanded["batch_size"] = normalize_optional_value(row.get("batch_size")) or str(args.default_batch_size)
    expanded["eval_batch_size"] = normalize_optional_value(row.get("eval_batch_size")) or str(args.default_eval_batch_size)
    expanded["gradient_accumulation_steps"] = (
        normalize_optional_value(row.get("gradient_accumulation_steps")) or str(args.default_gradient_accumulation_steps)
    )
    expanded["epochs"] = normalize_optional_value(row.get("epochs")) or normalize_optional_value(row.get("epoch_number"))
    expanded["learning_rate"] = normalize_optional_value(row.get("learning_rate"))
    expanded["lr_scheduler"] = normalize_optional_value(row.get("lr_scheduler")) or args.default_lr_scheduler
    expanded["weight_decay"] = normalize_optional_value(row.get("weight_decay")) or str(args.default_weight_decay)
    expanded["warmup_ratio"] = normalize_optional_value(row.get("warmup_ratio")) or str(args.default_warmup_ratio)
    expanded["max_grad_norm"] = normalize_optional_value(row.get("max_grad_norm"))
    expanded["num_workers"] = normalize_optional_value(row.get("num_workers")) or str(args.default_num_workers)
    expanded["seed"] = normalize_optional_value(row.get("seed")) or str(args.default_seed)
    expanded["patience"] = normalize_optional_value(row.get("patience"))
    expanded["device"] = normalize_optional_value(row.get("device")) or args.default_device
    expanded["log_every_steps"] = normalize_optional_value(row.get("log_every_steps")) or str(args.default_log_every_steps)
    expanded["step_metrics_every"] = normalize_optional_value(row.get("step_metrics_every")) or str(args.default_step_metrics_every)
    expanded["adaptation_method"] = normalize_optional_value(row.get("adaptation_method")) or "ttlora"
    expanded["dp_enabled"] = normalize_optional_value(row.get("dp_enabled")) or "false"
    expanded["dp_target_epsilon"] = normalize_optional_value(row.get("dp_target_epsilon"))
    expanded["dp_target_delta"] = normalize_optional_value(row.get("dp_target_delta")) or "1e-5"
    expanded["dp_max_grad_norm"] = normalize_optional_value(row.get("dp_max_grad_norm")) or expanded["max_grad_norm"]
    expanded["dp_poisson_sampling"] = normalize_optional_value(row.get("dp_poisson_sampling"))
    if expanded["dp_poisson_sampling"] is None:
        expanded["dp_poisson_sampling"] = "true" if args.default_dp_poisson_sampling else "false"
    expanded["dp_secure_mode"] = normalize_optional_value(row.get("dp_secure_mode"))
    if expanded["dp_secure_mode"] is None:
        expanded["dp_secure_mode"] = "true" if args.default_dp_secure_mode else "false"
    expanded["dp_grad_sample_mode"] = normalize_optional_value(row.get("dp_grad_sample_mode")) or args.default_dp_grad_sample_mode
    expanded["generation_eval_max_new_tokens"] = (
        normalize_optional_value(row.get("generation_eval_max_new_tokens")) or str(args.default_generation_eval_max_new_tokens)
    )
    expanded["max_train_samples"] = normalize_optional_value(row.get("max_train_samples"))
    expanded["max_eval_samples"] = normalize_optional_value(row.get("max_eval_samples"))
    expanded["notes"] = normalize_optional_value(row.get("notes"))

    if expanded["epochs"] is None:
        raise ValueError("Each CSV row must include 'epochs' or 'epoch_number'.")
    if expanded["learning_rate"] is None:
        raise ValueError("Each CSV row must include 'learning_rate'.")
    if expanded["patience"] is None:
        raise ValueError("Each CSV row must include 'patience'.")
    if expanded["max_grad_norm"] is None:
        raise ValueError("Each CSV row must include 'max_grad_norm'.")

    if expanded["adaptation_method"] == "ttlora":
        rank_value = normalize_optional_value(row.get("rank")) or normalize_optional_value(row.get("ttlora_rank"))
        alpha_value = normalize_optional_value(row.get("alpha")) or normalize_optional_value(row.get("ttlora_alpha"))
        if rank_value is None or alpha_value is None:
            raise ValueError("TT-LoRA rows must include 'rank' and 'alpha'.")
        expanded["ttlora_rank"] = rank_value
        expanded["ttlora_alpha"] = alpha_value
        expanded["ttlora_variant"] = (
            normalize_optional_value(row.get("ttlora_variant"))
            or normalize_optional_value(row.get("ttlora_adaptation"))
            or args.default_ttlora_variant
        )
        expanded["core_count"] = normalize_optional_value(row.get("core_count")) or str(args.default_core_count)
        expanded["ttlora_weights"] = (
            normalize_optional_value(row.get("ttlora_weights"))
            or normalize_optional_value(row.get("weights_to_adapt"))
            or normalize_optional_value(row.get("target_weights"))
        )
        expanded["run_name"] = derive_run_name(expanded)
        expanded["ttlora_weight_config"] = normalize_optional_value(row.get("ttlora_weight_config"))
        if expanded["ttlora_weight_config"] is None:
            ttshape_config_root = Path(args.default_ttshape_config_root).expanduser().resolve()
            model_config_path = resolve_ttlora_model_config_path(
                model_name=str(model_name),
                ttshape_config_root=ttshape_config_root,
            )
            model_payload = json.loads(model_config_path.read_text(encoding="utf-8"))
            selected_weights = _parse_ttlora_weight_selection(row, model_payload)
            generated_config_dir = Path(expanded["output_dir"]).expanduser().resolve() / "_generated_ttlora_weight_configs"
            generated_config_path = generated_config_dir / f"{expanded['run_name']}.json"
            expanded["ttlora_weight_config"] = str(
                materialize_ttlora_weight_config(
                    model_config_path=model_config_path,
                    output_path=generated_config_path,
                    core_count=int(expanded["core_count"]),
                    rank=int(expanded["ttlora_rank"]),
                    selected_weights=selected_weights,
                )
            )
    elif expanded["adaptation_method"] == "lora":
        rank_value = normalize_optional_value(row.get("rank")) or normalize_optional_value(row.get("lora_rank"))
        alpha_value = normalize_optional_value(row.get("alpha")) or normalize_optional_value(row.get("lora_alpha"))
        expanded["lora_rank"] = rank_value
        expanded["lora_alpha"] = alpha_value

    if is_blank(expanded.get("run_name")):
        expanded["run_name"] = derive_run_name(expanded)
    return expanded


def add_scalar_arg(command: list[str], flag: str, value: Any) -> None:
    if is_blank(value):
        return
    command.extend([flag, str(value).strip()])


def add_store_true_arg(command: list[str], flag: str, value: Any) -> None:
    if is_blank(value):
        return
    if parse_bool(value):
        command.append(flag)


def add_boolean_optional_arg(command: list[str], flag: str, value: Any) -> None:
    if is_blank(value):
        return
    normalized_flag = flag.lstrip("-")
    command.append(flag if parse_bool(value) else f"--no-{normalized_flag}")


def add_list_arg(command: list[str], flag: str, value: Any) -> None:
    if is_blank(value):
        return
    items = parse_list(str(value))
    if not items:
        return
    command.append(flag)
    command.extend(items)


def build_command(
    row: dict[str, str],
    *,
    python_executable: str,
    train_script: Path,
) -> list[str]:
    command = [python_executable, str(train_script)]

    add_scalar_arg(command, "--dataset-name", row.get("dataset_name"))
    add_scalar_arg(command, "--dataset-root", row.get("dataset_root"))
    add_scalar_arg(command, "--model-path", row.get("model_path"))
    add_scalar_arg(command, "--tokenizer-path", row.get("tokenizer_path"))
    add_scalar_arg(command, "--output-dir", row.get("output_dir"))
    add_scalar_arg(command, "--run-name", row.get("run_name"))

    add_store_true_arg(command, "--overwrite-run-dir", row.get("overwrite_run_dir"))
    add_store_true_arg(command, "--resume-from-last-epoch", row.get("resume_from_last_epoch"))
    add_boolean_optional_arg(command, "--summary-only", row.get("summary_only"))

    add_scalar_arg(command, "--max-length", row.get("max_length"))
    add_scalar_arg(command, "--train-split", row.get("train_split"))
    add_scalar_arg(command, "--validation-split", row.get("validation_split"))
    add_scalar_arg(command, "--text-column", row.get("text_column"))
    add_scalar_arg(command, "--training-format", row.get("training_format"))

    add_scalar_arg(command, "--batch-size", row.get("batch_size"))
    add_scalar_arg(command, "--eval-batch-size", row.get("eval_batch_size"))
    add_scalar_arg(command, "--gradient-accumulation-steps", row.get("gradient_accumulation_steps"))
    add_scalar_arg(command, "--epochs", row.get("epochs"))
    add_scalar_arg(command, "--learning-rate", row.get("learning_rate"))
    add_scalar_arg(command, "--lr-scheduler", row.get("lr_scheduler"))
    add_scalar_arg(command, "--weight-decay", row.get("weight_decay"))
    add_scalar_arg(command, "--warmup-ratio", row.get("warmup_ratio"))
    add_scalar_arg(command, "--max-grad-norm", row.get("max_grad_norm"))

    add_store_true_arg(command, "--dp-enabled", row.get("dp_enabled"))
    add_scalar_arg(command, "--dp-target-epsilon", row.get("dp_target_epsilon"))
    add_scalar_arg(command, "--dp-target-delta", row.get("dp_target_delta"))
    add_scalar_arg(command, "--dp-noise-multiplier", row.get("dp_noise_multiplier"))
    add_scalar_arg(command, "--dp-max-grad-norm", row.get("dp_max_grad_norm"))
    add_boolean_optional_arg(command, "--dp-poisson-sampling", row.get("dp_poisson_sampling"))
    add_boolean_optional_arg(command, "--dp-secure-mode", row.get("dp_secure_mode"))
    add_scalar_arg(command, "--dp-grad-sample-mode", row.get("dp_grad_sample_mode"))

    add_scalar_arg(command, "--num-workers", row.get("num_workers"))
    add_scalar_arg(command, "--seed", row.get("seed"))
    add_scalar_arg(command, "--patience", row.get("patience"))
    add_scalar_arg(command, "--device", row.get("device"))
    add_scalar_arg(command, "--gpu-id", row.get("gpu_id"))
    add_scalar_arg(command, "--log-every-steps", row.get("log_every_steps"))
    add_scalar_arg(command, "--step-metrics-every", row.get("step_metrics_every"))

    add_scalar_arg(command, "--adaptation-method", row.get("adaptation_method"))
    add_scalar_arg(command, "--lora-rank", row.get("lora_rank"))
    add_scalar_arg(command, "--lora-alpha", row.get("lora_alpha"))
    add_list_arg(command, "--lora-target-weights", row.get("lora_target_weights"))
    add_scalar_arg(command, "--ttlora-rank", row.get("ttlora_rank"))
    add_scalar_arg(command, "--ttlora-alpha", row.get("ttlora_alpha"))
    add_scalar_arg(command, "--ttlora-variant", row.get("ttlora_variant"))
    add_scalar_arg(command, "--ttlora-weight-config", row.get("ttlora_weight_config"))
    add_list_arg(command, "--adapt-layers", row.get("adapt_layers"))

    add_scalar_arg(command, "--max-train-samples", row.get("max_train_samples"))
    add_scalar_arg(command, "--max-eval-samples", row.get("max_eval_samples"))
    add_scalar_arg(command, "--generation-eval-samples", row.get("generation_eval_samples"))
    add_scalar_arg(command, "--generation-eval-max-new-tokens", row.get("generation_eval_max_new_tokens"))
    add_scalar_arg(command, "--notes", row.get("notes"))

    extra_args = row.get("extra_args")
    if not is_blank(extra_args):
        command.extend(shlex.split(str(extra_args)))
    return command


def row_is_enabled(row: dict[str, str]) -> bool:
    enabled = row.get("enabled")
    if is_blank(enabled):
        return True
    return parse_bool(enabled)


def run_dir_for_row(row: dict[str, str]) -> Path | None:
    run_name = row.get("run_name")
    output_dir = row.get("output_dir")
    if is_blank(run_name) or is_blank(output_dir):
        return None
    return Path(str(output_dir)).expanduser().resolve() / str(run_name).strip()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()
    train_script = (
        Path(args.train_script).expanduser().resolve()
        if args.train_script is not None
        else project_root / "train_generation.py"
    )

    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    selected_run_names = set(args.only_run_name or [])
    executed = 0
    skipped = 0

    for index, row in enumerate(rows, start=1):
        expanded_row = expand_row_defaults(row, args, project_root)
        run_name = str(expanded_row.get("run_name", "")).strip() or f"row_{index}"

        if not row_is_enabled(expanded_row):
            print(f"[skip-disabled] {run_name}")
            skipped += 1
            continue

        if selected_run_names and run_name not in selected_run_names:
            continue

        run_dir = run_dir_for_row(expanded_row)
        if args.skip_completed and run_dir is not None and (run_dir / "summary.json").exists():
            print(f"[skip-completed] {run_name}")
            skipped += 1
            continue

        command = build_command(
            expanded_row,
            python_executable=args.python_executable,
            train_script=train_script,
        )
        print(f"[run] {run_name}")
        sys.exit(1)
        print(" ".join(shlex.quote(part) for part in command))
        
        if args.dry_run:
            continue

        result = subprocess.run(command, cwd=str(project_root), check=False)
        executed += 1
        if result.returncode != 0:
            print(f"[failed] {run_name} rc={result.returncode}")
            if args.fail_fast:
                raise SystemExit(result.returncode)
        else:
            print(f"[done] {run_name}")

    print(f"[summary] rows={len(rows)} executed={executed} skipped={skipped}")


if __name__ == "__main__":
    main()
