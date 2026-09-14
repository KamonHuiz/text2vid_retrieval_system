#!/usr/bin/env python3
"""Derive the fine-tuning training subset from the train/val split.

Two filters, both defined on each video folder's *on-disk* keyframe order
(files sorted by name == sorted by frame number):

* ``--keep-every K``: keep keyframes whose folder index is a multiple of K
  (K=2 -> index 0, 2, 4, ...). Adjacent keyframes are near-duplicates, so
  halving costs little signal while removing redundant pairs that act as
  false negatives inside a contrastive batch.
* ``--drop-head N --drop-head-subsets L26``: drop the first N keyframes of
  every video in the listed subsets (L26's videos open with a shared intro
  sequence; identical frames across hundreds of videos are pure false
  negatives for contrastive training).

Only the train/val split is filtered. The 5% benchmark split is untouched,
so post-fine-tuning scores stay comparable with the Phase-1 baseline.

Usage:
    python scripts/make_finetune_subset.py \\
        --trainval /home/calypso/aic/data/splits/trainval_split.jsonl \\
        --keyframes-root /home/calypso/aic/data/keyframes \\
        --output /home/calypso/aic/data/splits/finetune_train.jsonl \\
        --keep-every 2 --drop-head 50 --drop-head-subsets L26
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mm_bench.data.dataset import load_samples_from_split_file, save_samples_to_split_file
from mm_bench.utils.logging_utils import get_console_logger

logger = get_console_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trainval", required=True)
    p.add_argument("--keyframes-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--keep-every", type=int, default=2)
    p.add_argument("--drop-head", type=int, default=50)
    p.add_argument("--drop-head-subsets", nargs="*", default=["L26"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.keyframes_root)
    samples = load_samples_from_split_file(args.trainval)

    folder_index: dict[str, dict[str, int]] = {}

    def index_in_folder(rel_path: str) -> int:
        folder = str(Path(rel_path).parent)
        if folder not in folder_index:
            names = sorted(p.name for p in (root / folder).iterdir() if p.is_file())
            folder_index[folder] = {n: i for i, n in enumerate(names)}
        return folder_index[folder][Path(rel_path).name]

    drop_head_subsets = set(args.drop_head_subsets)
    kept, before = [], Counter()
    dropped_head, dropped_stride = Counter(), Counter()
    for s in samples:
        before[s.subset] += 1
        idx = index_in_folder(s.image_path)
        if s.subset in drop_head_subsets and idx < args.drop_head:
            dropped_head[s.subset] += 1
            continue
        if idx % args.keep_every != 0:
            dropped_stride[s.subset] += 1
            continue
        kept.append(s)

    after = Counter(s.subset for s in kept)
    logger.info(f"{'subset':<8}{'trainval':>10}{'drop_head':>11}{'drop_stride':>13}{'kept':>9}")
    for subset in sorted(before):
        logger.info(f"{subset:<8}{before[subset]:>10}{dropped_head[subset]:>11}{dropped_stride[subset]:>13}{after[subset]:>9}")
    logger.info(f"{'TOTAL':<8}{sum(before.values()):>10}{sum(dropped_head.values()):>11}{sum(dropped_stride.values()):>13}{len(kept):>9}")

    n = save_samples_to_split_file(args.output, kept)
    logger.info(f"Wrote {n} fine-tuning samples -> {args.output}")


if __name__ == "__main__":
    main()
