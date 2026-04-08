from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_LR_GRID = (
    1e-1,
    2e-1,
    3e-1,
    5e-1,
    1e-2,
    2e-2,
    3e-2,
    5e-2,
    1e-3,
    2e-3,
    3e-3,
    5e-3,
    1e-4,
    2e-4,
    3e-4,
    5e-4,
    1e-5,
    2e-5,
    3e-5,
    5e-5,
)
METHOD_SPECS = (("ttlora", "contraction"), ("ttlora", "reconstruction"))


def format_lr(lr: float) -> str:
    return f"{lr:.0e}".replace("+0", "").replace("+", "")


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


def build_parser() -> argparse.ArgumentParser:
    phase_root = Path(__file__).resolve().parents[1]
    project_root = phase_root.parents[1]

    parser = argparse.ArgumentParser(description="Run the TT-LoRA contraction vs reconstruction training suite.")
    parser.add_argument("--suite-name", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Explicit seed list. If omitted, --num-random-seeds seeds are generated from --seed-generator-seed.",
    )
    parser.add_argument("--num-random-seeds", type=int, default=5)
    parser.add_argument("--seed-generator-seed", type=int, default=1337)
    parser.add_argument("--learning-rates", nargs="*", type=float, default=DEFAULT_LR_GRID)
    parser.add_argument("--phase-root", default=str(phase_root))
    parser.add_argument("--project-root", default=str(project_root))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--train-script", default=str(project_root / "train.py"))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dataset-root", default=str(project_root / "datasets"))
    parser.add_argument("--model-path", default=str(project_root / "roberta-base" / "checkpoints"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr-scheduler", default="none")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-modules", nargs="*", default=("query", "value"))
    parser.add_argument("--adapt-layers", nargs="*", type=int, default=None)
    parser.add_argument(
        "--ttlora-ranks",
        nargs="+",
        type=int,
        default=(2, 4, 6, 8),
        help="TT-LoRA ranks to sweep while keeping one shared TT shape.",
    )
    parser.add_argument("--ttlora-alpha", type=float, default=8.0)
    parser.add_argument("--ttlora-shape", nargs="*", type=int, default=(64, 4, 3, 3, 4, 64))
    parser.add_argument("--ttlora-input-factors", nargs="*", type=int, default=(64, 4, 3))
    parser.add_argument("--ttlora-output-factors", nargs="*", type=int, default=(64, 4, 3))
    parser.add_argument("--log-every-steps", type=int, default=20)
    parser.add_argument("--step-metrics-every", type=int, default=1)
    parser.add_argument(
        "--summary-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save only summary.json per run, skipping checkpoints, history csvs, and config artifacts.",
    )
    parser.add_argument(
        "--launcher",
        choices=("sequential", "ray"),
        default="sequential",
        help="Execution backend for the sweep. Use 'ray' to schedule one run per available GPU across a Ray cluster.",
    )
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--cpus-per-run", type=float, default=1.0)
    parser.add_argument("--gpus-per-run", type=float, default=1.0)
    parser.add_argument("--overwrite-run-dir", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs already present in the execution log with returncode 0.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.seeds:
        seeds = tuple(args.seeds)
    else:
        generator = random.Random(args.seed_generator_seed)
        seeds = tuple(generator.sample(range(1, 1_000_000), args.num_random_seeds))

    phase_root = Path(args.phase_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else phase_root / "suites"
    runs_root = Path(args.runs_root).expanduser().resolve() if args.runs_root else phase_root / "runs"
    suite_dir = output_root / args.suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    execution_log_path = suite_dir / "execution_log.csv"
    execution_rows = load_execution_log(execution_log_path)
    completed_ok = {row["run_name"] for row in execution_rows if str(row.get("returncode")) == "0"}

    run_specs = []
    for dataset_name, seed, ttlora_rank, (adaptation_method, ttlora_variant), lr in itertools.product(
        args.datasets,
        seeds,
        args.ttlora_ranks,
        METHOD_SPECS,
        args.learning_rates,
    ):
        run_name = "_".join(
            [
                "ttcomp",
                args.suite_name,
                dataset_name,
                f"rank{ttlora_rank}",
                ttlora_variant,
                f"lr{format_lr(lr)}",
                f"seed{seed}",
            ]
        )
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
            str(runs_root),
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
            "--ttlora-variant",
            ttlora_variant,
            "--ttlora-rank",
            str(ttlora_rank),
            "--ttlora-alpha",
            str(args.ttlora_alpha),
            "--ttlora-shape",
            *[str(item) for item in args.ttlora_shape],
            "--ttlora-input-factors",
            *[str(item) for item in args.ttlora_input_factors],
            "--ttlora-output-factors",
            *[str(item) for item in args.ttlora_output_factors],
            "--target-modules",
            *args.target_modules,
        ]
        if args.summary_only:
            command.append("--summary-only")
        if args.adapt_layers:
            command.extend(["--adapt-layers", *[str(item) for item in args.adapt_layers]])
        if args.overwrite_run_dir:
            command.append("--overwrite-run-dir")

        run_specs.append(
            {
                "dataset_name": dataset_name,
                "adaptation_method": adaptation_method,
                "ttlora_variant": ttlora_variant,
                "ttlora_rank": ttlora_rank,
                "seed": seed,
                "learning_rate": lr,
                "run_name": run_name,
                "command": command,
            }
        )

    manifest_path = suite_dir / "planned_runs.json"
    save_json(
        manifest_path,
        {
            "suite_name": args.suite_name,
            "datasets": list(args.datasets),
            "seeds": list(seeds),
            "ttlora_ranks": list(args.ttlora_ranks),
            "learning_rates": list(args.learning_rates),
            "runs": run_specs,
        },
    )
    print(f"Planned {len(run_specs)} runs. Manifest saved to {manifest_path}")

    if args.dry_run:
        for spec in run_specs:
            status = "SKIP" if args.resume and spec["run_name"] in completed_ok else "RUN"
            print(status, " ".join(spec["command"]))
        return

    pending_specs = [spec for spec in run_specs if not (args.resume and spec["run_name"] in completed_ok)]
    existing_rows_by_name = {row["run_name"]: row for row in execution_rows}

    if args.launcher == "sequential":
        for idx, spec in enumerate(run_specs, start=1):
            if args.resume and spec["run_name"] in completed_ok:
                print(f"[{idx}/{len(run_specs)}] Skipping completed run {spec['run_name']}")
                continue
            print(f"[{idx}/{len(run_specs)}] Running {spec['run_name']}")
            start = time.time()
            completed = subprocess.run(spec["command"], text=True)
            elapsed = time.time() - start
            row = {
                "run_name": spec["run_name"],
                "dataset_name": spec["dataset_name"],
                "adaptation_method": spec["adaptation_method"],
                "ttlora_variant": spec["ttlora_variant"],
                "ttlora_rank": spec["ttlora_rank"],
                "seed": spec["seed"],
                "learning_rate": spec["learning_rate"],
                "returncode": completed.returncode,
                "elapsed_seconds": elapsed,
            }
            existing_rows_by_name[spec["run_name"]] = row
            ordered_rows = [existing_rows_by_name[name] for name in sorted(existing_rows_by_name)]
            write_csv(execution_log_path, ordered_rows)
            if completed.returncode != 0 and args.stop_on_failure:
                raise SystemExit(completed.returncode)
    else:
        try:
            import ray
        except ImportError as exc:
            raise SystemExit("Ray is not installed in the current environment. Install ray to use --launcher ray.") from exc

        ray.init(address=args.ray_address)

        @ray.remote(num_cpus=args.cpus_per_run, num_gpus=args.gpus_per_run)
        def run_spec_remote(spec: dict) -> dict:
            start = time.time()
            completed = subprocess.run(spec["command"], text=True)
            elapsed = time.time() - start
            return {
                "run_name": spec["run_name"],
                "dataset_name": spec["dataset_name"],
                "adaptation_method": spec["adaptation_method"],
                "ttlora_variant": spec["ttlora_variant"],
                "ttlora_rank": spec["ttlora_rank"],
                "seed": spec["seed"],
                "learning_rate": spec["learning_rate"],
                "returncode": completed.returncode,
                "elapsed_seconds": elapsed,
            }

        futures = [run_spec_remote.remote(spec) for spec in pending_specs]
        completed_count = 0
        while futures:
            done, futures = ray.wait(futures, num_returns=1)
            row = ray.get(done[0])
            completed_count += 1
            print(f"[{completed_count}/{len(pending_specs)}] Finished {row['run_name']} rc={row['returncode']}")
            existing_rows_by_name[row["run_name"]] = row
            ordered_rows = [existing_rows_by_name[name] for name in sorted(existing_rows_by_name)]
            write_csv(execution_log_path, ordered_rows)
            if row["returncode"] != 0 and args.stop_on_failure:
                raise SystemExit(row["returncode"])

    print(f"Execution log saved to {execution_log_path}")


if __name__ == "__main__":
    main()
