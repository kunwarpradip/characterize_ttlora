from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

from suite_utils import save_json, write_csv


DEFAULT_LR_MAP: dict[str, tuple[float, ...]] = {
    "full": (1e-5, 2e-5, 5e-5),
    "lora": (1e-4, 5e-4, 1e-3, 2e-3),
    "ttlora-contraction": (5e-4, 1e-3, 2e-3, 5e-3),
    "ttlora-reconstruction": (1e-4, 5e-4, 1e-3, 2e-3),
}


def method_key(adaptation_method: str, ttlora_variant: str | None) -> str:
    if adaptation_method == "ttlora":
        return f"ttlora-{ttlora_variant}"
    return adaptation_method


def lr_candidates(args, adaptation_method: str, ttlora_variant: str | None) -> tuple[float, ...]:
    if args.learning_rates:
        return tuple(args.learning_rates)
    return DEFAULT_LR_MAP[method_key(adaptation_method, ttlora_variant)]


def ttlora_variant_values(methods: list[str]) -> list[tuple[str, str | None]]:
    values: list[tuple[str, str | None]] = []
    for method in methods:
        if method == "ttlora-contraction":
            values.append(("ttlora", "contraction"))
        elif method == "ttlora-reconstruction":
            values.append(("ttlora", "reconstruction"))
        else:
            values.append((method, None))
    return values


def format_lr(lr: float) -> str:
    return f"{lr:.0e}".replace("+0", "").replace("+", "")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a characterization experiment suite.")
    parser.add_argument("--suite-name", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=("full", "lora", "ttlora-contraction", "ttlora-reconstruction"),
        choices=("full", "lora", "ttlora-contraction", "ttlora-reconstruction"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument("--learning-rates", nargs="*", type=float, default=None)
    parser.add_argument("--output-root", default="/home/pkunwar/characterize_ttlora/suites")
    parser.add_argument("--runs-root", default="/home/pkunwar/characterize_ttlora/runs")
    parser.add_argument("--train-script", default="/home/pkunwar/characterize_ttlora/train.py")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dataset-root", default="/home/pkunwar/characterize_ttlora/datasets")
    parser.add_argument("--model-path", default="/home/pkunwar/characterize_ttlora/roberta-base/checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr-scheduler", default="none")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-modules", nargs="*", default=("query", "value"))
    parser.add_argument("--adapt-layers", nargs="*", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--ttlora-rank", type=int, default=5)
    parser.add_argument("--ttlora-alpha", type=float, default=8.0)
    parser.add_argument("--ttlora-shape", nargs="*", type=int, default=(64, 4, 3, 3, 4, 64))
    parser.add_argument("--ttlora-input-factors", nargs="*", type=int, default=(64, 4, 3))
    parser.add_argument("--ttlora-output-factors", nargs="*", type=int, default=(64, 4, 3))
    parser.add_argument("--log-every-steps", type=int, default=20)
    parser.add_argument("--step-metrics-every", type=int, default=1)
    parser.add_argument("--overwrite-run-dir", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    suite_dir = Path(args.output_root).expanduser().resolve() / args.suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    run_specs = []
    for dataset_name, seed, (adaptation_method, ttlora_variant) in itertools.product(
        args.datasets,
        args.seeds,
        ttlora_variant_values(list(args.methods)),
    ):
        for lr in lr_candidates(args, adaptation_method, ttlora_variant):
            run_name_parts = [
                "suite",
                args.suite_name,
                dataset_name,
                adaptation_method,
            ]
            if ttlora_variant is not None:
                run_name_parts.append(ttlora_variant)
            run_name_parts.extend([f"lr{format_lr(lr)}", f"seed{seed}"])
            run_name = "_".join(run_name_parts)

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
                args.runs_root,
                "--run-name",
                run_name,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--eval-batch-size",
                str(args.eval_batch_size),
                "--gradient-accumulation-steps",
                str(args.gradient_accumulation_steps),
                "--learning-rate",
                str(lr),
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
                args.device,
                "--log-every-steps",
                str(args.log_every_steps),
                "--step-metrics-every",
                str(args.step_metrics_every),
                "--adaptation-method",
                adaptation_method,
                "--target-modules",
                *args.target_modules,
            ]

            if args.overwrite_run_dir:
                command.append("--overwrite-run-dir")
            if args.adapt_layers:
                command.extend(["--adapt-layers", *[str(item) for item in args.adapt_layers]])
            if adaptation_method == "lora":
                command.extend(["--lora-rank", str(args.lora_rank), "--lora-alpha", str(args.lora_alpha)])
            if adaptation_method == "ttlora":
                command.extend(
                    [
                        "--ttlora-variant",
                        ttlora_variant or "contraction",
                        "--ttlora-rank",
                        str(args.ttlora_rank),
                        "--ttlora-alpha",
                        str(args.ttlora_alpha),
                        "--ttlora-shape",
                        *[str(item) for item in args.ttlora_shape],
                        "--ttlora-input-factors",
                        *[str(item) for item in args.ttlora_input_factors],
                        "--ttlora-output-factors",
                        *[str(item) for item in args.ttlora_output_factors],
                    ]
                )

            run_specs.append(
                {
                    "dataset_name": dataset_name,
                    "adaptation_method": adaptation_method,
                    "ttlora_variant": ttlora_variant,
                    "seed": seed,
                    "learning_rate": lr,
                    "run_name": run_name,
                    "command": command,
                }
            )

    manifest_path = suite_dir / "planned_runs.json"
    save_json(manifest_path, {"suite_name": args.suite_name, "runs": run_specs})
    print(f"Planned {len(run_specs)} runs. Manifest saved to {manifest_path}")

    if args.dry_run:
        for spec in run_specs:
            print(" ".join(spec["command"]))
        return

    execution_rows = []
    for idx, spec in enumerate(run_specs, start=1):
        print(f"[{idx}/{len(run_specs)}] Running {spec['run_name']}")
        start = time.time()
        completed = subprocess.run(spec["command"], text=True)
        elapsed = time.time() - start
        execution_rows.append(
            {
                "run_name": spec["run_name"],
                "dataset_name": spec["dataset_name"],
                "adaptation_method": spec["adaptation_method"],
                "ttlora_variant": spec["ttlora_variant"],
                "seed": spec["seed"],
                "learning_rate": spec["learning_rate"],
                "returncode": completed.returncode,
                "elapsed_seconds": elapsed,
            }
        )
        write_csv(suite_dir / "execution_log.csv", execution_rows)
        if completed.returncode != 0 and args.stop_on_failure:
            raise SystemExit(completed.returncode)

    print(f"Execution log saved to {suite_dir / 'execution_log.csv'}")


if __name__ == "__main__":
    main()
