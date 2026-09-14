#!/usr/bin/env python3
"""Run Phase 1 (baseline, zero-shot) or Phase 2 (fine-tuned) retrieval
evaluation for one model on the shared 5% benchmark holdout.

Usage:
    # Phase 1 -- zero-shot baseline, no checkpoint.
    python scripts/benchmark.py --model siglip2 --phase baseline \\
        --config configs/models/siglip2_vit_gopt16_384.yaml \\
        --benchmark-split /home/calypso/aic/data/splits/benchmark_split.jsonl \\
        --images-root /home/calypso/aic/data/keyframes \\
        --output-dir results/siglip2

    # Phase 2 -- fine-tuned checkpoint, same benchmark split.
    python scripts/benchmark.py --model siglip2 --phase finetuned \\
        --config configs/models/siglip2_vit_gopt16_384.yaml \\
        --checkpoint runs/siglip2_ft1/checkpoints/last.pt \\
        --benchmark-split /home/calypso/aic/data/splits/benchmark_split.jsonl \\
        --images-root /home/calypso/aic/data/keyframes \\
        --output-dir results/siglip2

    Add --save-embeddings to also write {phase}_embeddings.npz (image/text
    embeddings in split order) for offline fusion, e.g. scripts/fuse_rrf.py.

Dry run (synthetic model + synthetic data, no downloads):
    python scripts/benchmark.py --model siglip2 --phase baseline \\
        --config configs/models/siglip2_vit_gopt16_384.yaml \\
        --output-dir /tmp/dryrun_bench_siglip2 --dry-run
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mm_bench.data.dataset import load_samples_from_split_file
from mm_bench.data.dryrun_data import make_dry_run_dataset
from mm_bench.evaluation.evaluator import Evaluator
from mm_bench.models.registry import build_model
from mm_bench.utils.config import apply_dotted_overrides, load_config
from mm_bench.utils.logging_utils import get_console_logger

logger = get_console_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Result name (output subdir). The architecture itself comes from --config's `wrapper` key.")
    p.add_argument("--config", required=True)
    p.add_argument("--benchmark-config", default="configs/benchmark_default.yaml")
    p.add_argument("--phase", required=True, choices=["baseline", "finetuned"])
    p.add_argument("--checkpoint", default=None, help="Required for --phase finetuned; must be omitted for --phase baseline.")
    p.add_argument("--benchmark-split", help="Path to benchmark_split.jsonl (required unless --dry-run).")
    p.add_argument("--images-root", help="Root directory the split's image paths are relative to (required unless --dry-run).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=None, help="Overrides the benchmark config's per_device_batch_size.")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--precision", default=None, choices=["fp16", "bf16", "fp32"])
    p.add_argument("--save-embeddings", action="store_true", help="Also write {phase}_embeddings.npz next to the metrics.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--set", dest="overrides", nargs="*", default=[])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.phase == "finetuned" and not args.checkpoint and not args.dry_run:
        raise SystemExit("--phase finetuned requires --checkpoint (or pass --dry-run for a smoke test).")
    if args.phase == "baseline" and args.checkpoint:
        raise SystemExit("--phase baseline must not be given a --checkpoint (that would defeat the zero-shot comparison).")

    model_cfg = load_config(args.config)
    bench_cfg = load_config(args.benchmark_config)
    if args.overrides:
        model_cfg = apply_dotted_overrides(model_cfg, args.overrides)
        bench_cfg = apply_dotted_overrides(bench_cfg, args.overrides)

    if args.dry_run:
        tmp_dir = tempfile.mkdtemp(prefix="mm_bench_dryrun_bench_")
        samples, images_root = make_dry_run_dataset(tmp_dir, num_samples=16, image_size=224)
        checkpoint_path = args.checkpoint
    else:
        if not args.benchmark_split or not args.images_root:
            raise SystemExit("--benchmark-split and --images-root are required unless --dry-run is set.")
        samples = load_samples_from_split_file(args.benchmark_split)
        images_root = args.images_root
        checkpoint_path = args.checkpoint
        logger.info(f"Loaded {len(samples)} benchmark samples from {args.benchmark_split}")

    # Only build the PEFT wrapper when a fine-tuned checkpoint will be loaded
    # into it; the Phase-1 baseline must run on the raw pretrained weights.
    model = build_model(model_cfg, dry_run=args.dry_run, apply_adaptation=bool(checkpoint_path))

    evaluator = Evaluator(
        model=model,
        benchmark_samples=samples,
        images_root=images_root,
        batch_size=args.batch_size or bench_cfg.get("per_device_batch_size", 256),
        num_workers=args.num_workers or bench_cfg.get("num_workers", 8),
        precision=args.precision or bench_cfg.get("precision", "fp16"),
    )

    output_dir = Path(args.output_dir)
    output_path = output_dir / bench_cfg["output"]["metrics_filename_template"].format(phase=args.phase)
    result = evaluator.run(
        model_name=args.model,
        phase=args.phase,
        output_path=str(output_path),
        checkpoint_path=checkpoint_path,
        embeddings_path=str(output_dir / f"{args.phase}_embeddings.npz") if args.save_embeddings else None,
    )

    for direction, metrics in result["metrics"].items():
        logger.info(f"[{args.model}/{args.phase}] {direction}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "n_samples"))


if __name__ == "__main__":
    main()
