# TT-LoRA Core Count Study

This phase is for characterizing how the number of TT cores affects TT-LoRA
behavior while keeping the rest of the setup fixed.

The first step is to generate valid TT shapes that match the current TT-LoRA
implementation in `cttlora/adapters.py`.

For a linear layer with:

- `in_features`
- `out_features`

the current implementation expects:

- `input_factors` such that `prod(input_factors) == in_features`
- `output_factors` such that `prod(output_factors) == out_features`
- `tt_shape = input_factors + reversed(output_factors)`

So the total number of TT cores is:

- `len(tt_shape) = len(input_factors) + len(output_factors)`

Use the generator script to enumerate candidate shapes:

```bash
python /home/pkunwar/characterize_ttlora/phases/ttlora_core_count_study/scripts/generate_tt_shapes.py \
  --in-features 768 \
  --out-features 768 \
  --family-mode merge-lineage \
  --split-strategy symmetric \
  --output-json /home/pkunwar/characterize_ttlora/phases/ttlora_core_count_study/analysis/tt_shape.json
```

Recommended default for square RoBERTa attention projections:

- use even total core counts
- use `split-strategy symmetric`
- use the `merge-lineage` family when the main variable is the number of TT cores

That keeps the number of input and output cores balanced and makes the core
count comparison cleaner.

To run the phase-2 training sweep on `roberta-base` + `mrpc` with TT-LoRA
applied to `key`, `query`, and `value` across all RoBERTa layers:

```bash
python /home/pkunwar/characterize_ttlora/phases/ttlora_core_count_study/scripts/run_core_count_suite.py \
  --suite-name roberta_base_mrpc_kqv_200ep \
  --shape-json /home/pkunwar/characterize_ttlora/phases/ttlora_core_count_study/analysis/tt_shape.json \
  --dataset-name mrpc \
  --model-path /home/pkunwar/characterize_ttlora/roberta-base/checkpoints \
  --target-modules key query value \
  --ttlora-rank 6 \
  --learning-rate 2e-3 \
  --epochs 200 \
  --patience 1000 \
  --num-workers 0 \
  --step-metrics-every 1 \
  --no-summary-only
```

This runner writes:

- a suite manifest at `suites/<suite-name>/manifest.json`
- a run-level execution log at `suites/<suite-name>/execution_log.csv`
- per-run `summary.json`, `history.csv`, and `step_history.csv` under `runs/`

Those artifacts are intended to support later analysis of:

- validation-accuracy vs core count
- loss curves over 200 epochs
- gradient norms and clipping behavior
- runtime and memory trends
