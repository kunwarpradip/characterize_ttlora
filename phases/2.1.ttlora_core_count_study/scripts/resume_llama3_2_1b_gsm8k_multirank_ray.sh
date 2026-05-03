#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/pkunwar/characterize_ttlora"
PHASE_ROOT="$PROJECT_ROOT/phases/2.1.ttlora_core_count_study"
PYTHON_BIN="/home/pkunwar/miniconda3/envs/characterize-ttlora/bin/python"
SUITE_NAME="llama3_2_1b_gsm8k_reconstruction_cores2to20_ranks2_6_10_ray"
MANIFEST_PATH="$PHASE_ROOT/suites/$SUITE_NAME/manifest.json"

"$PYTHON_BIN" "$PHASE_ROOT/scripts/run_core_count_suite_ray.py" \
  --manifest-path "$MANIFEST_PATH" \
  --ray-address auto \
  --cpus-per-run 4 \
  --gpus-per-run 1 \
  --resume \
  --rerun-failed \
  --resume-from-summaries \
  --resume-generation-from-last-epoch
