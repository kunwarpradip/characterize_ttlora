#!/usr/bin/env bash
set -euo pipefail

source /home/pkunwar/miniconda3/etc/profile.d/conda.sh
conda activate characterize-ttlora

PROJECT_ROOT="/home/pkunwar/characterize_ttlora"
PHASE_ROOT="${PROJECT_ROOT}/phases/3.differential_privacy"
RUNS_ROOT="${PHASE_ROOT}/runs"

mkdir -p "${RUNS_ROOT}"

python "${PROJECT_ROOT}/train_generation.py" \
  --dataset-name enron \
  --dataset-root "${PROJECT_ROOT}/datasets" \
  --model-path "${PROJECT_ROOT}/gpt2small/checkpoints" \
  --output-dir "${RUNS_ROOT}" \
  --run-name "gpt2_enron_ttlora_dp_smoke_contraction_cores2_rank6_eps8" \
  --epochs 1 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --learning-rate 2e-4 \
  --lr-scheduler none \
  --weight-decay 0.01 \
  --warmup-ratio 0.0 \
  --max-grad-norm 1.0 \
  --num-workers 0 \
  --seed 647761 \
  --patience 1 \
  --device auto \
  --log-every-steps 10 \
  --step-metrics-every 1 \
  --adaptation-method ttlora \
  --ttlora-rank 6 \
  --ttlora-alpha 8.0 \
  --ttlora-variant contraction \
  --ttlora-weight-config "${PROJECT_ROOT}/phases/2.1.ttlora_core_count_study/suites/gpt2_ptb_enron_core_study/weight_configs/ttcore_gpt2_ptb_enron_core_study_enron_contraction_cores2_rank6_lr2e-04_seed647761.json" \
  --max-train-samples 32 \
  --max-eval-samples 16 \
  --dp-enabled \
  --dp-target-epsilon 8.0 \
  --dp-target-delta 1e-5 \
  --dp-max-grad-norm 1.0 \
  --no-dp-poisson-sampling \
  --dp-grad-sample-mode hooks \
  --notes "phase=3.differential_privacy smoke_test=true model=gpt2small dataset=enron"
