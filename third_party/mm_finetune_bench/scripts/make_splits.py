#!/usr/bin/env python3
"""Build the deterministic, per-subset 5% benchmark holdout.

This is the only script in the suite that never touches a model -- it is
pure dataset bookkeeping (read captions JSONL, group by subset, split,
write JSONL) and safe to run standalone before any model work begins.

Usage:
    python scripts/make_splits.py \\
        --keyframes-root /home/calypso/aic/data/keyframes \\
        --captions-glob "/home/calypso/aic/data/captions/captions_shard*.jsonl" \\
        --output-dir /home/calypso/aic/data/splits \\
        --benchmark-fraction 0.05 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mm_bench.data.dataset import load_samples_from_caption_jsonl, save_samples_to_split_file
from mm_bench.data.splits import split_by_subset, summarize_split
from mm_bench.utils.logging_utils import get_console_logger

logger = get_console_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--keyframes-root", required=True, help="Root directory containing the images referenced by captions JSONL (only checked for existence, not walked).")
    p.add_argument("--captions-glob", required=True, help="Glob matching one or more caption JSONL shards, e.g. 'data/captions/captions_shard*.jsonl'.")
    p.add_argument("--output-dir", required=True, help="Directory to write benchmark_split.jsonl and trainval_split.jsonl into.")
    p.add_argument("--benchmark-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-benchmark-per-subset", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    samples = load_samples_from_caption_jsonl(args.captions_glob, args.keyframes_root)
    if not samples:
        raise SystemExit(f"No valid (image, caption) samples found for glob '{args.captions_glob}'")
    logger.info(f"Loaded {len(samples)} captioned samples across {len({s.subset for s in samples})} subsets.")

    trainval, benchmark = split_by_subset(
        samples,
        benchmark_fraction=args.benchmark_fraction,
        seed=args.seed,
        min_benchmark_per_subset=args.min_benchmark_per_subset,
    )
    logger.info("\n" + summarize_split(split_by_subset.last_stats))  # type: ignore[attr-defined]

    output_dir = Path(args.output_dir)
    n_bench = save_samples_to_split_file(str(output_dir / "benchmark_split.jsonl"), benchmark)
    n_trainval = save_samples_to_split_file(str(output_dir / "trainval_split.jsonl"), trainval)
    logger.info(f"Wrote {n_bench} benchmark samples and {n_trainval} trainval samples to {output_dir}")


if __name__ == "__main__":
    main()
