#!/usr/bin/env python3
"""Aggregate per-model Phase 1/Phase 2 metrics into the final comparison
report.

Expects ``{results_dir}/{model_name}/baseline_metrics.json`` and
``{results_dir}/{model_name}/finetuned_metrics.json`` to already exist for
each model (produced by running ``scripts/benchmark.py`` twice per model,
once per phase). Writes ``results_comparison.csv`` and
``results_comparison.md`` into ``--output-dir``.

Usage:
    python scripts/run_all_benchmarks.py --results-dir results --output-dir results/report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mm_bench.evaluation.compare import build_comparison_report
from mm_bench.utils.logging_utils import get_console_logger

logger = get_console_logger(__name__)

DEFAULT_MODELS = ["pe_core_bigg", "metaclip2", "siglip2", "jina_omni"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    df = build_comparison_report(
        results_dir=args.results_dir,
        model_names=args.models,
        output_csv=str(output_dir / "results_comparison.csv"),
        output_markdown=str(output_dir / "results_comparison.md"),
    )
    logger.info(f"Wrote comparison report for {df['model'].nunique()} model(s) -> {output_dir}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
