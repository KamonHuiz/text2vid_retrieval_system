#!/usr/bin/env python3
"""Fine-tune one model with full resume support, on one GPU or several.

One GPU:
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py --model siglip2 \\
        --config configs/models/siglip2_vit_gopt16_384.yaml --train-config configs/train_ft1.yaml \\
        --train-split /home/calypso/aic/data/splits/finetune_train.jsonl \\
        --images-root /home/calypso/aic/data/keyframes --output-dir /home/calypso/aic/runs/siglip2_ft1

Two GPUs (all-gathered negatives), hard-negative batch plan, in-loop validation:
    torchrun --nproc_per_node 2 scripts/train.py --model pe_core_bigg \\
        --config configs/models/pe_core_bigg.yaml --train-config configs/train_ft_pecore.yaml \\
        --train-split /home/calypso/aic/data/splits/finetune_train.jsonl \\
        --images-root /home/calypso/aic/data/keyframes --output-dir /home/calypso/aic/runs/pe_core_bigg_ft1 \\
        --batch-plan /home/calypso/aic/data/emb/bge_vl_ft1_train/hard_batches_b256_g4.npz \\
        --val-split /home/calypso/aic/data/splits/val_2k.jsonl

Resume:      add --resume_from_checkpoint <run>/checkpoints/step_N.pt
Smoke test:  add --max-steps 5 --set log_every_steps=1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mm_bench.data.dataset import ImageCaptionDataset, load_samples_from_split_file
from mm_bench.data.dryrun_data import make_dry_run_dataset
from mm_bench.data.types import Sample
from mm_bench.evaluation.retrieval_metrics import compute_retrieval_metrics
from mm_bench.models.registry import build_model
from mm_bench.training.trainer import DTYPES, Trainer
from mm_bench.utils.config import apply_dotted_overrides, load_config
from mm_bench.utils.device import move_batch
from mm_bench.utils.logging_utils import get_console_logger
from mm_bench.utils.seed import seed_everything

logger = get_console_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Run name (logs only). The architecture comes from --config's `wrapper` key.")
    p.add_argument("--config", required=True, help="Model YAML (configs/models/*.yaml).")
    p.add_argument("--train-config", default="configs/train_default.yaml")
    p.add_argument("--train-split", help="Training JSONL (required unless --dry-run).")
    p.add_argument("--images-root", help="Root the split's image paths are relative to (required unless --dry-run).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-plan", default=None, help="npz from scripts/mine_hard_batches.py (batches, dup_id, paths).")
    p.add_argument("--val-split", default=None, help="JSONL for in-loop validation; best.pt tracks its mean R@1.")
    p.add_argument("--val-batch-size", type=int, default=64)
    p.add_argument("--resume_from_checkpoint", default=None)
    p.add_argument("--max-steps", type=int, default=None, help="Stop after N optimizer steps (smoke tests).")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--set", dest="overrides", nargs="*", default=[],
                   help="key=value overrides. Dotted keys (lora.r=32) go to the model config, bare keys (epochs=1) to the train config.")
    return p.parse_args()


def setup_distributed() -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return 0, 1
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    # Generous timeout: non-main ranks wait at a barrier while rank 0 runs
    # validation or writes a checkpoint.
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    return dist.get_rank(), world_size


def make_eval_fn(val_samples: List[Sample], images_root: str, batch_size: int, dtype: torch.dtype) -> Callable:
    """Validation closure: mean of text->image and image->text R@1 over the
    val split (candidates = the val split itself). The DataLoader is built on
    first use so only the rank that actually evaluates pays for it."""
    loader: Optional[DataLoader] = None

    def eval_fn(model) -> Dict[str, float]:
        nonlocal loader
        if loader is None:
            ds = ImageCaptionDataset(val_samples, images_root=images_root, image_transform=model.get_image_transform(),
                                     text_tokenizer=model.get_text_tokenizer())
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=model.get_collate_fn())
        device = next(model.parameters()).device
        images, texts = [], []
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
            for b in loader:
                images.append(model.encode_image(move_batch(b["pixel_values"], device)).float())
                texts.append(model.encode_text(move_batch(b["text_tokens"], device)).float())
        m = compute_retrieval_metrics(torch.cat(images), torch.cat(texts))
        t2i, i2t = m["text_to_image"], m["image_to_text"]
        return {"t2i_r1": t2i["recall@1"], "t2i_r5": t2i["recall@5"], "i2t_r1": i2t["recall@1"],
                "i2t_r5": i2t["recall@5"], "mean_recall": (t2i["recall@1"] + i2t["recall@1"]) / 2}

    return eval_fn


def main() -> None:
    args = parse_args()
    rank, world_size = setup_distributed()
    if rank != 0:
        logging.disable(logging.INFO)  # one copy of every log line, from rank 0

    model_cfg = load_config(args.config)
    train_cfg = load_config(args.train_config)
    if args.overrides:
        model_cfg = apply_dotted_overrides(model_cfg, [o for o in args.overrides if "." in o.split("=")[0]])
        train_cfg = apply_dotted_overrides(train_cfg, [o for o in args.overrides if "." not in o.split("=")[0]])
    model_cfg["flash_attention"] = train_cfg.get("flash_attention", False)

    if args.dry_run:
        dry_cfg = train_cfg.get("dry_run", {})
        samples, images_root = make_dry_run_dataset(tempfile.mkdtemp(prefix="mm_bench_dryrun_"),
                                                    num_samples=dry_cfg.get("num_fake_samples", 32),
                                                    image_size=dry_cfg.get("fake_image_size", 224))
        train_cfg.update(per_device_batch_size=min(train_cfg["per_device_batch_size"], 8), epochs=1,
                         num_workers=0, save_every_steps=2, eval_every_steps=10_000)
    else:
        if not args.train_split or not args.images_root:
            raise SystemExit("--train-split and --images-root are required unless --dry-run is set.")
        samples = load_samples_from_split_file(args.train_split)
        images_root = args.images_root
        logger.info(f"Loaded {len(samples)} training samples from {args.train_split}")

    batch_plan = dup_ids = None
    if args.batch_plan:
        plan = np.load(args.batch_plan)
        if not np.array_equal(plan["paths"], np.array([s.image_path for s in samples])):
            raise SystemExit("--batch-plan was mined for a different training split (paths/order differ).")
        batch_plan, dup_ids = plan["batches"], plan["dup_id"]
        logger.info(f"Batch plan: {batch_plan.shape[0]} x {batch_plan.shape[1]} from {args.batch_plan}")

    # Same seed on every rank before LoRA init; Trainer also broadcasts rank 0's
    # trainable weights, so ranks start identical either way.
    seed_everything(train_cfg.get("seed", 42))
    model = build_model(model_cfg, dry_run=args.dry_run)
    logger.info(model.trainable_parameters_report())

    eval_fn = None
    if args.val_split:
        eval_fn = make_eval_fn(load_samples_from_split_file(args.val_split), images_root, args.val_batch_size,
                               DTYPES[train_cfg.get("precision", "bf16")])

    trainer = Trainer(model=model, train_samples=samples, images_root=images_root, train_cfg=train_cfg,
                      output_dir=args.output_dir, eval_fn=eval_fn, max_steps=args.max_steps, rank=rank,
                      world_size=world_size, batch_plan=batch_plan, dup_ids=dup_ids)
    if args.resume_from_checkpoint:
        trainer.resume_from_checkpoint(args.resume_from_checkpoint)

    logger.info(f"batch={trainer.global_batch} accum={trainer.grad_accum_steps} steps={trainer.num_training_steps} "
                f"world={world_size} local_batch={trainer.local_batch} loss={trainer.loss_type} "
                f"logit_scale_max={trainer.logit_scale_max:.3f} plan={'mined' if batch_plan is not None else 'random'}")
    final_state = trainer.fit()
    logger.info(f"Training complete: epoch={final_state.epoch} global_step={final_state.global_step} best={final_state.best_metric}")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
