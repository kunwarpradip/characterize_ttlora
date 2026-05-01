#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/pkunwar/characterize_ttlora"
PHASE_ROOT="$PROJECT_ROOT/phases/2.1.ttlora_core_count_study"
PYTHON_BIN="/home/pkunwar/miniconda3/envs/characterize-ttlora/bin/python"
COMBINED_CSV="$PHASE_ROOT/analysis_outputs/combined_parameter_configs/llama3_2_1b_kv_qo_combined_lowest_parameter_configs.csv"
SUITE_NAME="llama3_2_1b_gsm8k_core20_reconstruction_ray"
MANIFEST_PATH="$PHASE_ROOT/suites/$SUITE_NAME/manifest.json"

"$PYTHON_BIN" "$PHASE_ROOT/scripts/run_core_count_suite.py" \
  --suite-name "$SUITE_NAME" \
  --project-root "$PROJECT_ROOT" \
  --phase-root "$PHASE_ROOT" \
  --dataset-name gsm8k \
  --generation-model-path "$PROJECT_ROOT/llama3.2-1b/checkpoints" \
  --generation-tokenizer-path "$PROJECT_ROOT/llama3.2-1b/checkpoints" \
  --generation-train-script "$PROJECT_ROOT/train_generation.py" \
  --generation-combined-shape-csv "$COMBINED_CSV" \
  --generation-combined-weight-group kv=k_proj,v_proj \
  --generation-combined-weight-group qo=q_proj,o_proj \
  --core-counts 20 \
  --variants reconstruction \
  --seeds 647761 \
  --ttlora-rank 6 \
  --ttlora-alpha 8.0 \
  --epochs 200 \
  --patience 5 \
  --batch-size 2 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 16 \
  --learning-rate 2e-3 \
  --two-core-learning-rate 2e-3 \
  --lr-scheduler none \
  --weight-decay 0.01 \
  --warmup-ratio 0.06 \
  --max-grad-norm 1.0 \
  --num-workers 0 \
  --generation-max-length 1024 \
  --generation-training-format prompt_completion \
  --generation-eval-max-new-tokens 256 \
  --adapt-layers 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 \
  --device cuda:0 \
  --log-every-steps 10 \
  --step-metrics-every 1 \
  --dry-run

"$PYTHON_BIN" "$PHASE_ROOT/scripts/run_core_count_suite_ray.py" \
  --manifest-path "$MANIFEST_PATH" \
  --ray-address auto \
  --cpus-per-run 4 \
  --gpus-per-run 1 \
  --resume \
  --resume-from-summaries
