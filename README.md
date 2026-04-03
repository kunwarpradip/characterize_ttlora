# characterize_ttlora

Phase 1 now has a clean local training scaffold for establishing a stable baseline before TT-LoRA-specific ablations.

Code layout:

- `train.py`: main CLI entrypoint for Phase 1 runs
- `cttlora/config.py`: experiment configuration dataclasses
- `cttlora/tasks.py`: task registry and dataset normalization
- `cttlora/data.py`: local dataset loading and tokenization
- `cttlora/modeling.py`: model loading and adaptation hooks
- `cttlora/adapters.py`: local LoRA and TT-LoRA module wrappers
- `cttlora/training.py`: training loop, metrics, checkpoints, and summaries
- `test_train.py` and `test_utils.py`: kept as reference code

Example run:

```bash
python train.py \
  --dataset-name mrpc \
  --dataset-root /home/pkunwar/characterize_ttlora/datasets \
  --model-path /home/pkunwar/characterize_ttlora/roberta-base/checkpoints \
  --adaptation-method full \
  --batch-size 16 \
  --lr-scheduler none \
  --epochs 5 \
  --output-dir runs
```

Outputs are written under `runs/<run-name>/`:

- `config.json`
- `history.csv`
- `step_history.csv`
- `summary.json`
- `checkpoints/best/`

By default, if a run directory already exists, the code creates a versioned directory such as `phase1_mrpc_full_v1`, `phase1_mrpc_full_v2`, and so on. Reuse the exact same run directory only with:

```bash
--overwrite-run-dir
```

Current Phase 1 scope:

- stable RoBERTa sequence-classification baseline
- fixed learning rate by default for cleaner characterization
- RoBERTa `lora` and `ttlora` adaptation methods for attention `query/key/value` targets
- TT-LoRA `contraction` and `reconstruction` variants for direct comparison
- gradient accumulation support
- train/validation convergence tracking
- gradient norm logging
- step-level optimization diagnostics
- parameter counting
- peak GPU memory tracking
- versioned run directories for repeated experiments

Example TT-LoRA contraction run:

```bash
python train.py \
  --dataset-name mrpc \
  --model-path /home/pkunwar/characterize_ttlora/roberta-base/checkpoints \
  --adaptation-method ttlora \
  --ttlora-variant contraction \
  --target-modules query value \
  --ttlora-rank 5 \
  --ttlora-alpha 8 \
  --ttlora-shape 64 4 3 3 4 64 \
  --ttlora-input-factors 64 4 3 \
  --ttlora-output-factors 64 4 3
```

Example TT-LoRA reconstruction run:

```bash
python train.py \
  --dataset-name mrpc \
  --model-path /home/pkunwar/characterize_ttlora/roberta-base/checkpoints \
  --adaptation-method ttlora \
  --ttlora-variant reconstruction \
  --target-modules query value \
  --ttlora-rank 5 \
  --ttlora-alpha 8 \
  --ttlora-shape 64 4 3 3 4 64
```

Available LR schedule choices:

- `none`
- `linear_with_warmup`
- `cosine_with_warmup`
- `constant_with_warmup`

When a scheduler is enabled, `--learning-rate` remains the optimizer LR scale used by the schedule.

Experiment suite workflow:

1. Launch a matrix of runs:

```bash
python run_experiment_suite.py \
  --suite-name ttlora_method_decision \
  --datasets mrpc \
  --methods full lora ttlora-contraction ttlora-reconstruction \
  --seeds 42 43 44
```

2. Aggregate and analyze the suite:

```bash
python analyze_experiment_suite.py \
  --suite-name ttlora_method_decision
```

This produces:

- `suites/<suite-name>/planned_runs.json`
- `suites/<suite-name>/execution_log.csv`
- `suites/<suite-name>/analysis/all_runs_summary.csv`
- `suites/<suite-name>/analysis/aggregated_summary.csv`
- `suites/<suite-name>/analysis/report.md`
- suite-level plots for accuracy vs LR, memory vs accuracy, and stability vs accuracy
