#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/pkunwar/characterize_ttlora"
PHASE_ROOT="$PROJECT_ROOT/phases/2.1.ttlora_core_count_study"
PYTHON_BIN="/home/pkunwar/miniconda3/envs/characterize-ttlora/bin/python"
ANALYSIS_ROOT="$PHASE_ROOT/analysis"
COMBINED_OUTPUT_ROOT="$PHASE_ROOT/analysis_outputs/combined_parameter_configs"
SUITES_ROOT="$PHASE_ROOT/suites"

RANKS=(2 6 10)
CORE_COUNTS=()
for core in $(seq 2 20); do
  CORE_COUNTS+=("$core")
done

MERGED_SUITE_NAME="llama3_2_1b_gsm8k_reconstruction_cores2to20_ranks2_6_10_ray"
MERGED_MANIFEST_PATH="$SUITES_ROOT/$MERGED_SUITE_NAME/manifest.json"
mkdir -p "$SUITES_ROOT/$MERGED_SUITE_NAME"

MANIFEST_ARGS=()

for rank in "${RANKS[@]}"; do
  KV_DIR="$ANALYSIS_ROOT/parameter_space_llama3_2_1b_kv_rank${rank}"
  QO_DIR="$ANALYSIS_ROOT/parameter_space_llama3_2_1b_qo_rank${rank}"
  COMBINED_CSV="$COMBINED_OUTPUT_ROOT/llama3_2_1b_kv_qo_rank${rank}_combined_lowest_parameter_configs.csv"
  COMBINED_META="$COMBINED_OUTPUT_ROOT/llama3_2_1b_kv_qo_rank${rank}_combined_metadata.json"
  SUITE_NAME="llama3_2_1b_gsm8k_reconstruction_cores2to20_rank${rank}_ray"
  SUITE_MANIFEST_PATH="$SUITES_ROOT/$SUITE_NAME/manifest.json"

  "$PYTHON_BIN" "$PHASE_ROOT/scripts/analyze_tt_shape_parameter_space.py" \
    --phase-root "$PHASE_ROOT" \
    --weight-shape 512 2048 \
    --rank "$rank" \
    --split-strategy all \
    --adapted-weights-per-layer 2 \
    --num-layers 16 \
    --core-counts "${CORE_COUNTS[@]}" \
    --output-dir "$KV_DIR"

  "$PYTHON_BIN" "$PHASE_ROOT/scripts/analyze_tt_shape_parameter_space.py" \
    --phase-root "$PHASE_ROOT" \
    --weight-shape 2048 2048 \
    --rank "$rank" \
    --split-strategy all \
    --adapted-weights-per-layer 2 \
    --num-layers 16 \
    --core-counts "${CORE_COUNTS[@]}" \
    --output-dir "$QO_DIR"

  "$PYTHON_BIN" "$PHASE_ROOT/scripts/build_combined_shape_csv.py" \
    --source-spec "kv=$KV_DIR/lowest_parameter_shapes_by_core_count.csv" \
    --source-spec "qo=$QO_DIR/lowest_parameter_shapes_by_core_count.csv" \
    --multiplier "kv=2" \
    --multiplier "qo=2" \
    --description "kv=Shared by k_proj and v_proj" \
    --description "qo=Shared by q_proj and o_proj" \
    --join-mode inner \
    --output-csv "$COMBINED_CSV" \
    --output-metadata-json "$COMBINED_META"

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
    --core-counts "${CORE_COUNTS[@]}" \
    --variants reconstruction \
    --seeds 647761 \
    --ttlora-rank "$rank" \
    --ttlora-alpha 8.0 \
    --epochs 200 \
    --patience 5 \
    --batch-size 2 \
    --eval-batch-size 2 \
    --gradient-accumulation-steps 16 \
    --learning-rate 5e-5 \
    --two-core-learning-rate 5e-5 \
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

  MANIFEST_ARGS+=(--manifest-path "$SUITE_MANIFEST_PATH")
done

"$PYTHON_BIN" "$PHASE_ROOT/scripts/merge_suite_manifests.py" \
  "${MANIFEST_ARGS[@]}" \
  --output-path "$MERGED_MANIFEST_PATH" \
  --suite-name "$MERGED_SUITE_NAME"

"$PYTHON_BIN" "$PHASE_ROOT/scripts/run_core_count_suite_ray.py" \
  --manifest-path "$MERGED_MANIFEST_PATH" \
  --ray-address auto \
  --cpus-per-run 4 \
  --gpus-per-run 1 \
  --resume \
  --resume-from-summaries
