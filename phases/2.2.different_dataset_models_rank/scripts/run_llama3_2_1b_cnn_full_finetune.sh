#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/pkunwar/characterize_ttlora"
PHASE_ROOT="${PROJECT_ROOT}/phases/2.2.different_dataset_models_rank"
PYTHON="/home/pkunwar/miniconda3/envs/characterize-ttlora/bin/python"

"${PYTHON}" "${PROJECT_ROOT}/train_generation.py" \
  --dataset-name cnn \
  --dataset-root "${PROJECT_ROOT}/datasets" \
  --model-path "${PROJECT_ROOT}/llama3.2-1b/checkpoints" \
  --tokenizer-path "${PROJECT_ROOT}/llama3.2-1b/checkpoints" \
  --output-dir "${PHASE_ROOT}/runs/cnn_llama3_2_1b" \
  --run-name "llama3_2_1b_cnn_full_finetune" \
  --adaptation-method full \
  --max-length 1024 \
  --training-format prompt_completion \
  --batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 32 \
  --learning-rate 1e-5 \
  --lr-scheduler none \
  --weight-decay 0.01 \
  --warmup-ratio 0.06 \
  --max-grad-norm 1.0 \
  --epochs 3 \
  --patience 3 \
  --num-workers 0 \
  --seed 647761 \
  --device cuda:0 \
  --log-every-steps 10 \
  --step-metrics-every 1 \
  --generation-eval-max-new-tokens 256 \
  --notes "phase=2.2.different_dataset_models_rank model=llama3.2-1b dataset=cnn variant=full_finetune"
