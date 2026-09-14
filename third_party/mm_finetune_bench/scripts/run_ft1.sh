#!/usr/bin/env bash
# Fine-tune round 1 + Phase 2 benchmark, one model per GPU.
#
#   GPU 0: siglip2            (slowest to train)
#   GPU 1: dfn5b_vith14_378 -> bge_vl_large
#
# Each model: 1 epoch on data/splits/finetune_train.jsonl, then scripts/
# benchmark.py --phase finetuned on the untouched 5% holdout, writing
# results/<model>/finetuned_metrics.json next to the Phase-1 baseline.
# Finally builds results/report_ft1/results_comparison.{csv,md}.
#
# Re-running is safe: a model whose checkpoints/last.pt exists skips training;
# one with only step_*.pt checkpoints resumes from the newest.
set -uo pipefail

cd "$(dirname "$0")/.."
source /home/calypso/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl-vllm

TRAIN=/home/calypso/aic/data/splits/finetune_train.jsonl
BENCH=/home/calypso/aic/data/splits/benchmark_split.jsonl
IMAGES=/home/calypso/aic/data/keyframes
RUNS=/home/calypso/aic/runs
RESULTS=/home/calypso/aic/results
LOGS=/home/calypso/aic/logs
mkdir -p "$LOGS" "$RUNS"

train_and_bench () {  # gpu name config train_batch eval_batch
  local gpu=$1 name=$2 config=$3 batch=$4 eval_batch=$5
  local ckpt_dir="$RUNS/${name}_ft1/checkpoints"

  if [ -f "$ckpt_dir/last.pt" ]; then
    echo "=== [GPU $gpu] $name: last.pt exists, skipping training ==="
  else
    local latest
    latest=$(ls -t "$ckpt_dir"/step_*.pt 2>/dev/null | head -1)
    local resume_args=()
    [ -n "$latest" ] && resume_args=(--resume_from_checkpoint "$latest")
    echo "=== [GPU $gpu] $name train start $(date '+%F %T') ${latest:+(resume from $latest)} ==="
    CUDA_VISIBLE_DEVICES=$gpu python3 scripts/train.py --model "$name" --config "$config" \
      --train-config configs/train_ft1.yaml --train-split "$TRAIN" --images-root "$IMAGES" \
      --output-dir "$RUNS/${name}_ft1" "${resume_args[@]}" \
      --set per_device_batch_size="$batch"
    local rc=$?
    echo "=== [GPU $gpu] $name train exit=$rc $(date '+%F %T') ==="
    [ $rc -ne 0 ] && return $rc
  fi

  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/benchmark.py --model "$name" --phase finetuned \
    --config "$config" --checkpoint "$ckpt_dir/last.pt" \
    --benchmark-split "$BENCH" --images-root "$IMAGES" \
    --output-dir "$RESULTS/$name" --batch-size "$eval_batch" --num-workers 12 --precision fp16
  echo "=== [GPU $gpu] $name bench exit=$? $(date '+%F %T') ==="
}

(
  train_and_bench 0 siglip2 configs/models/siglip2_vit_gopt16_384.yaml 256 64
) > "$LOGS/ft1_gpu0.log" 2>&1 &
PID0=$!

(
  train_and_bench 1 dfn5b_vith14_378 configs/models/dfn5b_vith14_378.yaml 256 64
  train_and_bench 1 bge_vl_large     configs/models/bge_vl_large.yaml     512 128
) > "$LOGS/ft1_gpu1.log" 2>&1 &
PID1=$!

wait $PID0 $PID1

python3 scripts/run_all_benchmarks.py --results-dir "$RESULTS" --output-dir "$RESULTS/report_ft1" \
  --models siglip2 dfn5b_vith14_378 bge_vl_large
echo "FT1 COMPLETE $(date '+%F %T')"
