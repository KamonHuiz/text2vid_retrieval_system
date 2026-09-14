"""Deterministic, per-subset 5% benchmark holdout.

Requirement: for every subset "L" (L21..L30), exactly 5% of that subset's
samples are reserved for the benchmark/eval split, using a fixed seed, so
that re-running split generation is reproducible and every model/version
evaluates on the identical holdout.

Method: samples within a subset are sorted by ``image_path`` (a stable,
content-independent order), then permuted with a ``numpy.random.Generator``
seeded deterministically from ``(global_seed, subset)``. The first
``round(n * benchmark_fraction)`` permuted indices become the benchmark set
for that subset; everything else is train/val. Per-subset seeding (rather
than one global seed over the concatenated list) guarantees the 5% ratio
holds *within every subset independently*, not just in aggregate, and is
insensitive to how many subsets exist or the order samples are loaded in.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from .types import Sample


def _subset_seed(global_seed: int, subset: str) -> int:
    """Combine a global seed with a subset name into a stable 32-bit seed."""
    h = hashlib.sha256(f"{global_seed}:{subset}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], byteorder="big")


def split_by_subset(
    samples: List[Sample],
    benchmark_fraction: float = 0.05,
    seed: int = 42,
    min_benchmark_per_subset: int = 1,
) -> Tuple[List[Sample], List[Sample]]:
    """Return ``(trainval_samples, benchmark_samples)``.

    Raises:
        ValueError: if ``benchmark_fraction`` is not in ``(0, 1)``.
    """
    if not 0.0 < benchmark_fraction < 1.0:
        raise ValueError(f"benchmark_fraction must be in (0, 1), got {benchmark_fraction}")

    by_subset: Dict[str, List[Sample]] = defaultdict(list)
    for s in samples:
        by_subset[s.subset].append(s)

    trainval: List[Sample] = []
    benchmark: List[Sample] = []
    stats = {}

    for subset, items in sorted(by_subset.items()):
        items_sorted = sorted(items, key=lambda s: s.image_path)
        n = len(items_sorted)
        rng = np.random.default_rng(_subset_seed(seed, subset))
        perm = rng.permutation(n)
        n_bench = max(min_benchmark_per_subset, round(n * benchmark_fraction)) if n > 0 else 0
        n_bench = min(n_bench, n)
        bench_positions = set(perm[:n_bench].tolist())

        for i, item in enumerate(items_sorted):
            (benchmark if i in bench_positions else trainval).append(item)

        stats[subset] = {"total": n, "benchmark": n_bench, "trainval": n - n_bench}

    split_by_subset.last_stats = stats  # type: ignore[attr-defined]
    return trainval, benchmark


def summarize_split(stats: Dict[str, Dict[str, int]]) -> str:
    lines = [f"{'subset':<10}{'total':>10}{'trainval':>12}{'benchmark':>12}{'bench_pct':>12}"]
    total_all = total_bench = 0
    for subset, s in sorted(stats.items()):
        pct = 100.0 * s["benchmark"] / s["total"] if s["total"] else 0.0
        lines.append(f"{subset:<10}{s['total']:>10}{s['trainval']:>12}{s['benchmark']:>12}{pct:>11.2f}%")
        total_all += s["total"]
        total_bench += s["benchmark"]
    overall_pct = 100.0 * total_bench / total_all if total_all else 0.0
    lines.append("-" * 56)
    lines.append(f"{'TOTAL':<10}{total_all:>10}{total_all - total_bench:>12}{total_bench:>12}{overall_pct:>11.2f}%")
    return "\n".join(lines)
