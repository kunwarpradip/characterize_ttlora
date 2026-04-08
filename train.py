from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from cttlora import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from cttlora.training import run_phase1_experiment


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent
    default_dataset_root = os.environ.get("STACK_V2_ROOT", str(project_root / "datasets"))
    parser = argparse.ArgumentParser(description="Phase 1 training scaffold for TT-LoRA characterization.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-root", default=default_dataset_root)
    parser.add_argument("--model-path", default="/home/pkunwar/characterize_ttlora/roberta-base/checkpoints")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--overwrite-run-dir",
        action="store_true",
        help="Allow overwriting an existing run directory.",
    )
    parser.add_argument(
        "--summary-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save only summary.json for the run and skip checkpoints/history/config artifacts.",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument(
        "--lr-scheduler",
        default="none",
        choices=("none", "linear_with_warmup", "cosine_with_warmup", "constant_with_warmup"),
        help=(
            "Learning-rate schedule. "
            "When a scheduler is used, --learning-rate is treated as the base/peak LR scale."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every-steps", type=int, default=20)
    parser.add_argument(
        "--step-metrics-every",
        type=int,
        default=1,
        help="Write step-level metrics every N optimizer steps.",
    )
    parser.add_argument(
        "--adaptation-method",
        default="full",
        choices=("full", "classifier-only", "lora", "ttlora"),
    )
    parser.add_argument(
        "--target-modules",
        nargs="*",
        default=("query", "value"),
        help="Metadata hook for future TT-LoRA layer targeting.",
    )
    parser.add_argument(
        "--adapt-layers",
        nargs="*",
        type=int,
        default=None,
        help="Optional list of encoder layer indices to adapt. Defaults to all layers.",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--ttlora-rank", type=int, default=5)
    parser.add_argument("--ttlora-alpha", type=float, default=8.0)
    parser.add_argument(
        "--ttlora-variant",
        default="contraction",
        choices=("contraction", "reconstruction"),
        help="TT-LoRA update implementation to use.",
    )
    parser.add_argument(
        "--ttlora-shape",
        nargs="*",
        type=int,
        default=(64, 4, 3, 3, 4, 64),
        help="Flattened TT shape for the adapted matrix.",
    )
    parser.add_argument(
        "--ttlora-input-factors",
        nargs="*",
        type=int,
        default=(64, 4, 3),
        help="Input-side TT factors whose product must match in_features.",
    )
    parser.add_argument(
        "--ttlora-output-factors",
        nargs="*",
        type=int,
        default=(64, 4, 3),
        help="Output-side TT factors whose product must match out_features.",
    )
    parser.add_argument(
        "--text-columns",
        nargs="*",
        default=None,
        help="Override the task registry for a new dataset.",
    )
    parser.add_argument("--num-labels", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--notes", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model_path = args.model_path
    tokenizer_path = args.tokenizer_path or model_path
    default_run_name = (
        f"phase1_{args.dataset_name}_{args.adaptation_method}_{args.ttlora_variant}"
        if args.adaptation_method == "ttlora"
        else f"phase1_{args.dataset_name}_{args.adaptation_method}"
    )
    run_name = args.run_name or default_run_name

    experiment = ExperimentConfig(
        model=ModelConfig(
            model_name_or_path=model_path,
            tokenizer_name_or_path=tokenizer_path,
            adaptation_method=args.adaptation_method,
            target_modules=tuple(args.target_modules),
            adapt_layers=tuple(args.adapt_layers) if args.adapt_layers else None,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            ttlora_rank=args.ttlora_rank,
            ttlora_alpha=args.ttlora_alpha,
            ttlora_variant=args.ttlora_variant,
            ttlora_shape=tuple(args.ttlora_shape),
            ttlora_input_factors=tuple(args.ttlora_input_factors),
            ttlora_output_factors=tuple(args.ttlora_output_factors),
        ),
        data=DataConfig(
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root,
            max_length=args.max_length,
            text_columns=tuple(args.text_columns) if args.text_columns else None,
            num_labels=args.num_labels,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
        ),
        training=TrainingConfig(
            output_dir=args.output_dir,
            run_name=run_name,
            overwrite_run_dir=args.overwrite_run_dir,
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
            device=args.device,
            log_every_steps=args.log_every_steps,
            step_metrics_every=args.step_metrics_every,
        ),
        notes=args.notes,
        metadata={"phase": "phase1"},
    )
    summary = run_phase1_experiment(experiment)
    print("Run complete.")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
