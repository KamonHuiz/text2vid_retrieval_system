#!/usr/bin/env bash
# PE-Core bigG fine-tune (2 GPUs, all-gathered negatives, hard-negative batch
# plan, in-loop validation) followed by the Phase 2 benchmark.
#
# Phase 2 evaluates best.pt (highest val mean R@1), falling back to last.pt,
# on the untouched 5% holdout, and also saves the benchmark embeddings next
# to the other three fine-tuned models' so fusion can include PE-Core.
#
# Re-running is safe: last.pt present => training is skipped; otherwise the
# newest step_*.pt is resumed.
set -uo pipefail

cd "$(dirname "$0")/.."
source /home/calypso/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl-vllm

# PCIe peer-to-peer between the two A6000s hangs silently on this machine:
# the first NCCL collective never completes (verified with a standalone
# all_reduce). Over shared memory instead it runs at ~80 ms per 192 MB, which
# is noise next to a PE-Core step.
export NCCL_P2P_DISABLE=1
# The first OOM left 5 GiB reserved-but-unusable to fragmentation; expandable segments reclaim it.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NAME=pe_core_bigg
CONFIG=configs/models/pe_core_bigg.yaml
RUN=/home/calypso/aic/runs/${NAME}_ft1
CKPT=$RUN/checkpoints
TRAIN=/home/calypso/aic/data/splits/finetune_train.jsonl
BENCH=/home/calypso/aic/data/splits/benchmark_split.jsonl
IMAGES=/home/calypso/aic/data/keyframes
PLAN=/home/calypso/aic/data/emb/bge_vl_ft1_train/hard_batches_b256_g4.npz
VAL=/home/calypso/aic/data/splits/val_2k.jsonl
RESULTS=/home/calypso/aic/results

if [ -f "$CKPT/last.pt" ]; then
  echo "=== [GPU 0,1] $NAME: last.pt exists, skipping training ==="
else
  LATEST=$(ls -t "$CKPT"/step_*.pt 2>/dev/null | head -1)
  RESUME=()
  [ -n "$LATEST" ] && RESUME=(--resume_from_checkpoint "$LATEST")
  echo "=== [GPU 0,1] $NAME train start $(date '+%F %T') ${LATEST:+(resume from $LATEST)} ==="
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node 2 scripts/train.py \
    --model "$NAME" --config "$CONFIG" --train-config configs/train_ft_pecore.yaml \
    --train-split "$TRAIN" --images-root "$IMAGES" --output-dir "$RUN" \
    --batch-plan "$PLAN" --val-split "$VAL" "${RESUME[@]}"
  rc=$?
  echo "=== [GPU 0,1] $NAME train exit=$rc $(date '+%F %T') ==="
  [ $rc -ne 0 ] && exit $rc
fi

EVAL_CKPT=$CKPT/best.pt
[ -f "$EVAL_CKPT" ] || EVAL_CKPT=$CKPT/last.pt
echo "=== Phase 2 benchmark with $EVAL_CKPT ==="
CUDA_VISIBLE_DEVICES=0 python3 scripts/benchmark.py --model "$NAME" --phase finetuned --config "$CONFIG" \
  --checkpoint "$EVAL_CKPT" --benchmark-split "$BENCH" --images-root "$IMAGES" \
  --output-dir "$RESULTS/$NAME" --batch-size 64 --num-workers 12 --precision fp16 --save-embeddings
echo "=== [GPU 0,1] $NAME bench exit=$? $(date '+%F %T') ==="

mkdir -p "$RESULTS/fusion_ft1/$NAME"
cp "$RESULTS/$NAME/finetuned_embeddings.npz" "$RESULTS/fusion_ft1/$NAME/"

python3 scripts/run_all_benchmarks.py --results-dir "$RESULTS" --output-dir "$RESULTS/report_ft1" \
  --models siglip2 dfn5b_vith14_378 bge_vl_large "$NAME"
echo "PECORE FT1 COMPLETE $(date '+%F %T')"
