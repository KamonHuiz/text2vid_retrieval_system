# Multimodal Fine-Tuning & Benchmarking Suite

Production-grade scaffold for fine-tuning and benchmarking four image-text
contrastive models on an existing image/caption dataset (the AIC keyframe
corpus, organized into per-video-lesson subsets `L21` .. `L30`):

| Key            | Model                                             | Backend                         |
|----------------|----------------------------------------------------|----------------------------------|
| `pe_core_bigg` | PE Core BigG (Perception Encoder, Meta)             | `open_clip_torch`                |
| `metaclip2`    | MetaCLIP 2 (worldwide, Meta)                        | `open_clip_torch`                |
| `siglip2`      | ViT-gopt-16-SigLIP2-384 (Google)                    | `transformers` (`Siglip2Model`)  |
| `jina_omni`    | `jinaai/jina-embeddings-v5-omni-small-retrieval`    | `transformers` (`trust_remote_code`) |

This repository is **code only**. Nothing in it downloads weights,
instantiates a model, or launches a training/eval run on import — every
side-effecting action is gated behind an explicit CLI entry point in
`scripts/`.

## Layout

```
configs/                   YAML configs (per-model + shared train/benchmark defaults)
mm_bench/                  Installable package
  data/                    Dataset schema, deterministic 5% split, torch Dataset/collate
  models/                  Common wrapper interface + one adapter per architecture
  peft_utils/              LoRA/QLoRA application helpers (peft)
  losses/                  InfoNCE (learnable logit scale) and SigLIP sigmoid loss
  training/                Checkpointing, optimizer/scheduler factories, Trainer loop
  evaluation/               Retrieval metrics, zero-shot/fine-tuned evaluator, comparison report
scripts/                   CLI entry points (the only files meant to be executed)
docs/HYPERPARAMETERS.md    Architecture-by-architecture hyperparameter discussion
```

## Workflow

```bash
# 1. Build the deterministic 5% benchmark holdout, per subset "L", from the
#    captioned keyframe corpus produced by the captioning pipeline.
python scripts/make_splits.py \
    --keyframes-root /home/calypso/aic/data/keyframes \
    --captions-glob "/home/calypso/aic/data/captions/captions_shard*.jsonl" \
    --output-dir /home/calypso/aic/data/splits \
    --benchmark-fraction 0.05 --seed 42

# 2. Phase 1 -- zero-shot baseline (v1) on the benchmark split, no checkpoint.
python scripts/benchmark.py --model siglip2 --phase baseline \
    --config configs/models/siglip2_vit_gopt16_384.yaml \
    --benchmark-split /home/calypso/aic/data/splits/benchmark_split.jsonl \
    --output-dir results/siglip2

# 3. Fine-tune (LoRA by default; resumable).
python scripts/train.py --model siglip2 \
    --config configs/models/siglip2_vit_gopt16_384.yaml \
    --train-config configs/train_default.yaml \
    --train-split /home/calypso/aic/data/splits/trainval_split.jsonl \
    --output-dir runs/siglip2 \
    --resume_from_checkpoint runs/siglip2/checkpoints/last.pt   # omit for fresh run

# 4. Phase 2 -- evaluate the fine-tuned checkpoint (v2) on the same split.
python scripts/benchmark.py --model siglip2 --phase finetuned \
    --config configs/models/siglip2_vit_gopt16_384.yaml \
    --checkpoint runs/siglip2/checkpoints/best.pt \
    --benchmark-split /home/calypso/aic/data/splits/benchmark_split.jsonl \
    --output-dir results/siglip2

# 5. Repeat 2-4 for the other three models, then build the comparison report.
python scripts/run_all_benchmarks.py --results-dir results --output-dir results/report
```

Every script accepts `--dry-run`, which swaps in a tiny synthetic model and a
handful of in-memory fake samples so the full control-flow (data loading,
loss, optimizer step, checkpoint save/resume, metric computation, report
generation) can be smoke-tested without downloading any weights or touching
a real GPU.

## Resume semantics

`training.checkpoint.save_checkpoint` / `load_checkpoint` persist, atomically:
model weights (or LoRA adapter weights only, when PEFT is enabled), optimizer
state, LR scheduler state, AMP `GradScaler` state, Python/NumPy/Torch/CUDA RNG
state, and `{epoch, global_step, best_metric}`. `scripts/train.py
--resume_from_checkpoint <path>` restores all of the above and continues the
loop from `global_step + 1`, so interrupting and resuming a run is
bit-for-bit equivalent to letting it run uninterrupted (module to
non-determinism already inherent in cuDNN kernels).
