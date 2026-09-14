#!/usr/bin/env bash
# Phase 1: zero-shot baseline retrieval metrics for all four models on the
# shared 5% benchmark holdout, run across both GPUs.
#
# Scheduling: jina_omni is by far the slowest (a unified decoder pays a full
# forward on BOTH the image and the text side), so it gets GPU 0 to itself
# while the three dual encoders run back-to-back on GPU 1. That balances the
# makespan at roughly the length of the jina run instead of serializing all
# four.
set -uo pipefail

cd "$(dirname "$0")/.."
source /home/calypso/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl-vllm

SPLIT=/home/calypso/aic/data/splits/benchmark_split.jsonl
IMAGES=/home/calypso/aic/data/keyframes
RESULTS=/home/calypso/aic/results
LOGS=/home/calypso/aic/logs
mkdir -p "$LOGS" "$RESULTS"

run_one () {  # gpu model config batch
  local gpu=$1 model=$2 config=$3 batch=$4
  echo "=== [GPU $gpu] $model baseline start $(date +%T) ==="
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/benchmark.py \
    --model "$model" --phase baseline \
    --config "$config" \
    --benchmark-split "$SPLIT" \
    --images-root "$IMAGES" \
    --output-dir "$RESULTS/$model" \
    --batch-size "$batch" --num-workers 12 --precision fp16
  echo "=== [GPU $gpu] $model baseline exit=$? $(date +%T) ==="
}

# GPU 0: the slow one, alone.
(
  run_one 0 jina_omni configs/models/jina_omni_v5_small.yaml 32
) > "$LOGS/phase1_gpu0.log" 2>&1 &
PID0=$!

# GPU 1: the three dual encoders, slowest first.
(
  run_one 1 pe_core_bigg configs/models/pe_core_bigg.yaml 32
  run_one 1 metaclip2    configs/models/metaclip2.yaml    64
  run_one 1 siglip2      configs/models/siglip2_vit_gopt16_384.yaml 64
) > "$LOGS/phase1_gpu1.log" 2>&1 &
PID1=$!

wait $PID0 $PID1
echo "PHASE1 COMPLETE $(date +%T)"
