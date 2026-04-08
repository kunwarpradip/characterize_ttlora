# TT-LoRA Variant Comparison

This phase isolates the comparison between TT-LoRA `contraction` and `reconstruction`.

Design rules for this phase:

- One codebase, one adapter implementation, one switch:
  `cttlora.adapters.TTLoRALinearWrapper(mode={"contraction","reconstruction"})`
- The only intentional implementation difference is the TT-LoRA forward path.
- Functional equivalence should be checked before benchmarking.
- Inference comparison and training comparison are treated as separate experiments.

Folder layout:

- `scripts/`
  Phase-local launchers and validation utilities.
- `runs/`
  Training runs for this phase only.
- `suites/`
  Suite manifests, execution logs, and suite-specific analyses.
- `analysis/`
  Standalone reports such as equivalence checks and inference benchmarks.

Recommended workflow:

1. Run the functional equivalence check:

```bash
python /home/pkunwar/characterize_ttlora/phases/ttlora_variant_comparison/scripts/verify_equivalence.py \
  --dataset-name mrpc
```

2. Launch the training comparison suite:

```bash
python /home/pkunwar/characterize_ttlora/phases/ttlora_variant_comparison/scripts/run_training_suite.py \
  --suite-name lr_selection_multi_dataset \
  --datasets mrpc
```

Default training-grid behavior:

- methods: `contraction`, `reconstruction`
- ranks: `2, 4, 6, 8`
- one shared TT shape for every run in the suite
- learning rates: `{1,2,3,5} x {10^-1, 10^-2, 10^-3, 10^-4, 10^-5}`
- seeds: `5` random seeds by default
- sweep storage mode: `summary-only` by default

For distributed sweeps on a Ray cluster:

```bash
python /home/pkunwar/characterize_ttlora/phases/ttlora_variant_comparison/scripts/run_training_suite.py \
  --suite-name rank_lr_selection \
  --datasets mrpc sst2 rte \
  --launcher ray \
  --ray-address auto \
  --resume
```

This schedules one run per GPU by default with:

- `--gpus-per-run 1`
- `--cpus-per-run 1`

3. Aggregate and plot the results:

```bash
python /home/pkunwar/characterize_ttlora/phases/ttlora_variant_comparison/scripts/analyze_training_suite.py \
  --suite-name lr_selection_multi_dataset
```

4. Benchmark inference paths on a trained run:

```bash
python /home/pkunwar/characterize_ttlora/phases/ttlora_variant_comparison/scripts/benchmark_inference_paths.py \
  --run-dir /home/pkunwar/characterize_ttlora/phases/ttlora_variant_comparison/runs/<run-name>
```
