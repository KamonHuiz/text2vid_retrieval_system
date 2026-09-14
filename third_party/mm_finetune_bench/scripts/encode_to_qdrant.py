#!/usr/bin/env python3
"""Encode a keyframe split with one model; upsert image vectors into Qdrant
and/or save image+caption embeddings as .npz.

Modes:
* ``--estimate-only N`` -- encode N images, report throughput + ETA, write nothing.
* normal -- encode every sample in the split; with ``--collection`` upsert
  image vectors to Qdrant (``--resume`` skips points already present); with
  ``--save-npy DIR`` also write ``DIR/shard{k}.npz`` holding float32 image
  and caption embeddings plus image paths, in split order.

Multi-GPU is data-parallel: one process per GPU with
``CUDA_VISIBLE_DEVICES=<i> --shard <i> --num-shards <n>``. Point IDs derive
from the image path, so shards never collide in the collection.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/encode_to_qdrant.py \\
        --model bge_vl_large_ft1 --config configs/models/bge_vl_large.yaml \\
        --checkpoint /home/calypso/aic/runs/bge_vl_large_ft1/checkpoints/last.pt \\
        --split /home/calypso/aic/data/splits/finetune_train.jsonl \\
        --images-root /home/calypso/aic/data/keyframes \\
        --collection aic_bge_vl_ft1_train --save-npy /home/calypso/aic/data/emb/bge_vl_ft1_train \\
        --shard 0 --num-shards 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mm_bench.data.dataset import ImageCaptionDataset, load_samples_from_split_file
from mm_bench.data.types import infer_subset_and_ids
from mm_bench.indexing.qdrant_store import QdrantVectorStore, point_id_for_path
from mm_bench.models.registry import build_model
from mm_bench.training.checkpoint import load_checkpoint
from mm_bench.utils.config import apply_dotted_overrides, load_config
from mm_bench.utils.device import move_batch
from mm_bench.utils.logging_utils import get_console_logger

logger = get_console_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Run name for logs. The architecture comes from --config's `wrapper` key.")
    p.add_argument("--config", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--images-root", required=True)
    p.add_argument("--collection", default=None, help="Qdrant collection for image vectors.")
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--save-npy", default=None, metavar="DIR", help="Write DIR/shard{k}.npz with image+caption embeddings.")
    p.add_argument("--checkpoint", default=None, help="Fine-tuned checkpoint; omit to encode with pretrained weights.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--resume", action="store_true", help="Skip points already in the collection (incompatible with --save-npy).")
    p.add_argument("--recreate", action="store_true", help="Drop and rebuild the collection first (shard 0 only).")
    p.add_argument("--estimate-only", type=int, default=0, metavar="N")
    p.add_argument("--estimate-json", default=None)
    p.add_argument("--skip-text", action="store_true",
                    help="With --save-npy, skip the text-tower forward pass and write zero-filled "
                         "caption embeddings. Halves per-batch compute for image-only indexing "
                         "(the common case: search queries are encoded separately at eval time).")
    p.add_argument("--set", dest="overrides", nargs="*", default=[])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.collection or args.save_npy or args.estimate_only):
        raise SystemExit("Nothing to write: pass --collection and/or --save-npy (or --estimate-only).")
    if args.resume and args.save_npy:
        raise SystemExit("--save-npy needs the full shard in order; it cannot be combined with --resume.")

    model_cfg = load_config(args.config)
    if args.overrides:
        model_cfg = apply_dotted_overrides(model_cfg, args.overrides)
    model_cfg.setdefault("flash_attention", False)

    all_samples = load_samples_from_split_file(args.split)
    samples = [s for i, s in enumerate(all_samples) if i % args.num_shards == args.shard]
    logger.info(f"[shard {args.shard}/{args.num_shards}] split={len(all_samples)} shard_total={len(samples)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]

    t_load = time.time()
    model = build_model(model_cfg, dry_run=False, apply_adaptation=bool(args.checkpoint))
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, map_location="cpu", restore_rng=False)
        logger.info(f"Loaded fine-tuned checkpoint: {args.checkpoint}")
    model.to(device=device, dtype=dtype)
    model.eval()
    logger.info(f"Model ready in {time.time() - t_load:.1f}s")

    collate_fn = model.get_collate_fn()
    encode_text = bool(args.save_npy) and not args.skip_text
    if args.skip_text and args.save_npy:
        logger.info("--skip-text: writing zero-filled caption embeddings, image-only forward pass")

    # Probe the embedding dim through the real Dataset -> collate -> encode path.
    probe_ds = ImageCaptionDataset(samples[:1], images_root=args.images_root, image_transform=model.get_image_transform())
    with torch.no_grad():
        vector_size = int(model.encode_image(move_batch(collate_fn([probe_ds[0]])["pixel_values"], device, dtype)).shape[-1])
    logger.info(f"Embedding dim (probed): {vector_size}")

    store = None
    todo = samples[: args.estimate_only] if args.estimate_only else samples
    if args.collection and not args.estimate_only:
        store = QdrantVectorStore(args.collection, vector_size, url=args.qdrant_url)
        store.ensure_collection(recreate=args.recreate and args.shard == 0)
        if args.resume:
            ids = [point_id_for_path(s.image_path) for s in samples]
            present = store.existing_ids(ids)
            todo = [s for s, pid in zip(samples, ids) if pid not in present]
            logger.info(f"Resume: {len(samples) - len(todo)} already indexed, {len(todo)} remaining")
    if not todo:
        logger.info("Nothing to do.")
        return

    dataset = ImageCaptionDataset(
        todo,
        images_root=args.images_root,
        image_transform=model.get_image_transform(),
        text_tokenizer=model.get_text_tokenizer() if encode_text else None,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        collate_fn=collate_fn, pin_memory=device.type == "cuda")

    image_chunks: List[np.ndarray] = []
    text_chunks: List[np.ndarray] = []
    paths: List[str] = []
    processed, warmup_batches, t_start, t_all = 0, 2, None, time.time()

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi == warmup_batches:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_start, processed = time.time(), 0

            image_embeds = model.encode_image(move_batch(batch["pixel_values"], device, dtype)).float().cpu().numpy()
            text_embeds = None
            if encode_text:
                text_embeds = model.encode_text(move_batch(batch["text_tokens"], device, dtype)).float().cpu().numpy()
            elif args.save_npy and args.skip_text:
                text_embeds = np.zeros_like(image_embeds)

            if args.save_npy:
                image_chunks.append(image_embeds)
                text_chunks.append(text_embeds)
                paths.extend(batch["image_paths"])

            if store is not None:
                payloads: List[Dict[str, Any]] = []
                for i, path in enumerate(batch["image_paths"]):
                    ids = infer_subset_and_ids(path)
                    payloads.append({"path": path, "subset": batch["subsets"][i], "video_id": ids["video_id"],
                                     "frame_id": ids["frame_id"], "caption": batch["captions"][i]})
                store.upsert_batch(batch["image_paths"], image_embeds, payloads)

            processed += len(batch["image_paths"])
            if bi % 20 == 0 and t_start is not None and processed > 0:
                logger.info(f"[shard {args.shard}] batch {bi}/{len(loader)} | {processed / (time.time() - t_start):.1f} img/s")

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - (t_start or t_all)
    rate = processed / elapsed if elapsed > 0 else 0.0
    logger.info(f"[shard {args.shard}] DONE {len(todo)} imgs, {rate:.2f} img/s")

    if args.save_npy:
        out = Path(args.save_npy)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / f"shard{args.shard}.npz", image=np.concatenate(image_chunks), text=np.concatenate(text_chunks),
                 paths=np.array(paths))
        logger.info(f"Wrote {out / f'shard{args.shard}.npz'} ({len(paths)} rows)")

    if store is not None:
        store.finalize_index()
        logger.info(f"Collection '{args.collection}' now holds {store.count()} points")

    if args.estimate_only:
        full_n = len(all_samples)
        result = {"model": args.model, "embed_dim": vector_size, "precision": args.precision, "batch_size": args.batch_size,
                  "measured_imgs": processed, "measured_seconds": round(elapsed, 2), "img_per_sec_1gpu": round(rate, 2),
                  "eta_hours_1gpu_split": round(full_n / rate / 3600, 2) if rate else None,
                  "eta_hours_2gpu_split": round(full_n / (rate * 2) / 3600, 2) if rate else None, "split_size": full_n}
        print(json.dumps(result, indent=2))
        if args.estimate_json:
            with open(args.estimate_json, "a") as f:
                f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
