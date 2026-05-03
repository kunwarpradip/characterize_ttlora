from __future__ import annotations

import argparse
import json
from pathlib import Path

from cttlora.generation_config import (
    GenerationDataConfig,
    GenerationExperimentConfig,
    GenerationModelConfig,
    GenerationTrainingConfig,
    TTLoRAWeightConfig,
)
from cttlora.generation_training import run_generation_experiment


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="GPT-2 TT-LoRA training scaffold for local generation datasets.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-root", default=str(project_root / "datasets"))
    parser.add_argument("--model-path", default=str(project_root / "gpt2small" / "checkpoints"))
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite-run-dir", action="store_true")
    parser.add_argument(
        "--resume-from-last-epoch",
        action="store_true",
        help=(
            "Resume a generation run from the saved best checkpoint under the same run directory, "
            "continuing from the epoch after that checkpoint."
        ),
    )
    parser.add_argument(
        "--summary-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save only summary.json and skip history/checkpoint artifacts.",
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--text-column", default="text")
    parser.add_argument(
        "--training-format",
        default="blocks",
        choices=("blocks", "prompt_completion"),
        help=(
            "blocks concatenates tokenized text into LM blocks. "
            "prompt_completion keeps rows intact and masks prompt tokens from the loss."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--lr-scheduler",
        default="none",
        choices=("none", "linear_with_warmup", "cosine_with_warmup", "constant_with_warmup"),
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="Convenience override for selecting a specific CUDA device, e.g. --gpu-id 1 -> cuda:1.",
    )
    parser.add_argument("--log-every-steps", type=int, default=20)
    parser.add_argument("--step-metrics-every", type=int, default=1)
    parser.add_argument(
        "--adaptation-method",
        default="ttlora",
        choices=("full", "ttlora", "lora"),
        help="Generation adaptation method. Use lora for a plain LoRA baseline.",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument(
        "--lora-target-weights",
        nargs="*",
        default=("q_proj", "k_proj", "v_proj", "o_proj"),
        help="LLaMA attention weights to adapt for plain LoRA.",
    )
    parser.add_argument("--ttlora-rank", type=int, default=6)
    parser.add_argument("--ttlora-alpha", type=float, default=8.0)
    parser.add_argument(
        "--ttlora-variant",
        default="contraction",
        choices=("contraction", "reconstruction"),
    )
    parser.add_argument(
        "--ttlora-weight-config",
        default=None,
        help="Path to a per-run JSON file describing the adapted GPT-2 weights and their TT shapes.",
    )
    parser.add_argument(
        "--adapt-layers",
        nargs="*",
        type=int,
        default=None,
        help="Optional zero-based GPT-2 block indices to adapt. The suite runner converts user-facing 1-based indices.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument(
        "--generation-eval-samples",
        type=int,
        default=None,
        help="Optional row-level generation-evaluation sample cap. Used for GSM8K exact-match evaluation.",
    )
    parser.add_argument(
        "--generation-eval-max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens per example for row-level generation evaluation.",
    )
    parser.add_argument("--notes", default=None)
    return parser


def load_weight_configs(path: Path) -> tuple[TTLoRAWeightConfig, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    weights = payload.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"No weight configs found in {path}")

    loaded: list[TTLoRAWeightConfig] = []
    for weight_name, config in weights.items():
        loaded.append(
            TTLoRAWeightConfig(
                weight_name=weight_name,
                tt_shape=tuple(int(item) for item in config["tt_shape"]),
                input_factors=tuple(int(item) for item in config["input_factors"]),
                output_factors=tuple(int(item) for item in config["output_factors"]),
                weight_shape=tuple(int(item) for item in config["weight_shape"])
                if config.get("weight_shape") is not None
                else None,
            )
        )
    return tuple(loaded)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    resolved_device = f"cuda:{args.gpu_id}" if args.gpu_id is not None else args.device

    model_path = args.model_path
    tokenizer_path = args.tokenizer_path or model_path
    run_name = args.run_name or (
        f"generation_{args.dataset_name}_{args.ttlora_variant}"
        if args.adaptation_method == "ttlora"
        else f"generation_{args.dataset_name}_lora"
    )
    weight_config_path = (
        Path(args.ttlora_weight_config).expanduser().resolve()
        if args.ttlora_weight_config is not None
        else None
    )
    if args.adaptation_method == "ttlora" and weight_config_path is None:
        parser.error("--ttlora-weight-config is required when --adaptation-method ttlora")

    experiment = GenerationExperimentConfig(
        model=GenerationModelConfig(
            model_name_or_path=model_path,
            tokenizer_name_or_path=tokenizer_path,
            ttlora_rank=args.ttlora_rank,
            ttlora_alpha=args.ttlora_alpha,
            ttlora_variant=args.ttlora_variant,
            weight_configs=load_weight_configs(weight_config_path) if weight_config_path is not None else (),
            adaptation_method=args.adaptation_method,
            adapt_layers=tuple(args.adapt_layers) if args.adapt_layers else None,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_target_weights=tuple(args.lora_target_weights),
        ),
        data=GenerationDataConfig(
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root,
            max_length=args.max_length,
            train_split=args.train_split,
            validation_split=args.validation_split,
            text_column=args.text_column,
            training_format=args.training_format,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            generation_eval_samples=args.generation_eval_samples,
            generation_eval_max_new_tokens=args.generation_eval_max_new_tokens,
        ),
        training=GenerationTrainingConfig(
            output_dir=args.output_dir,
            run_name=run_name,
            overwrite_run_dir=args.overwrite_run_dir,
            resume_from_last_epoch=args.resume_from_last_epoch,
            summary_only=args.summary_only,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            lr_scheduler=args.lr_scheduler,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            max_grad_norm=args.max_grad_norm,
            num_workers=args.num_workers,
            seed=args.seed,
            patience=args.patience,
            device=resolved_device,
            log_every_steps=args.log_every_steps,
            step_metrics_every=args.step_metrics_every,
        ),
        notes=args.notes,
        metadata={
            "phase": "ttlora_core_count_study",
            "ttlora_weight_config_path": str(weight_config_path) if weight_config_path is not None else None,
        },
    )
    summary = run_generation_experiment(experiment)
    print("Run complete.")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
