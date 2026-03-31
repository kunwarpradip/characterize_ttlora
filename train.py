from __future__ import annotations

import argparse
import os

from cttlora import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from cttlora.training import run_phase1_experiment


def build_parser() -> argparse.ArgumentParser:
    default_dataset_root = os.environ.get("Dataset ROOT", "/usr/projects/unsupgan/afia/stack_v2")
    parser = argparse.ArgumentParser(description="Phase 1 training scaffold for TT-LoRA characterization.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-root", default=default_dataset_root)
    parser.add_argument("--model-path", default="/home/pkunwar/characterize_ttlora/roberta-base/checkpoints")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every-steps", type=int, default=20)
    parser.add_argument(
        "--adaptation-method",
        default="full",
        choices=("full", "classifier-only", "ttlora"),
    )
    parser.add_argument(
        "--target-modules",
        nargs="*",
        default=("query", "value"),
        help="Metadata hook for future TT-LoRA layer targeting.",
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
    run_name = args.run_name or f"phase1_{args.dataset_name}_{args.adaptation_method}"

    experiment = ExperimentConfig(
        model=ModelConfig(
            model_name_or_path=model_path,
            tokenizer_name_or_path=tokenizer_path,
            adaptation_method=args.adaptation_method,
            target_modules=tuple(args.target_modules),
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
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            max_grad_norm=args.max_grad_norm,
            num_workers=args.num_workers,
            seed=args.seed,
            patience=args.patience,
            device=args.device,
            log_every_steps=args.log_every_steps,
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
