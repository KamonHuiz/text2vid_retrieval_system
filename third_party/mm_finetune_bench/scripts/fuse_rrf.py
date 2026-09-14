#!/usr/bin/env python3
"""Reciprocal Rank Fusion (and z-score fusion) over several models'
benchmark embeddings.

Inputs are the ``{phase}_embeddings.npz`` files written by
``scripts/benchmark.py --save-embeddings`` (image/text embeddings in split
order, plus the image paths used to verify every model saw the same order).

RRF:  score(q, d) = sum_m 1 / (k + rank_m(q, d)),  rank 1-indexed over model
m's *full* candidate list -- computed exactly over all N candidates, not a
top-K truncation, in GPU row chunks. Scores are float64 because at deep ranks
the gap between neighbouring RRF values (~1/(k+r)^2 ~ 4e-9) is below float32
resolution, which would corrupt Mean Rank.

RRF ties are common by construction (rank tuples (1, 2) and (2, 1) sum to the
same value), so ties are broken by the models' mean cosine through a
perturbation smaller than any real RRF gap.

z-score fusion: average of per-query z-normalized cosines (raw cosines are not
comparable across models -- SigLIP and CLIP cosines live on different scales).

Usage:
    python scripts/fuse_rrf.py --emb-dir /home/calypso/aic/results/fusion_ft1 \\
        --models dfn5b_vith14_378 siglip2 bge_vl_large --k 60
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mm_bench.evaluation.retrieval_metrics import _metrics_from_ranks

DIRECTIONS = ("text_to_image", "image_to_text")
SHORT = {"dfn5b_vith14_378": "DFN5B", "siglip2": "SigLIP2", "bge_vl_large": "BGE-VL"}


def load_embeddings(emb_dir: Path, model: str, phase: str):
    z = np.load(emb_dir / model / f"{phase}_embeddings.npz", allow_pickle=False)
    return torch.from_numpy(z["image"]).float(), torch.from_numpy(z["text"]).float(), z["paths"]


def ranks_desc(scores: torch.Tensor) -> torch.Tensor:
    """1-indexed rank of every column within its row (highest score = 1)."""
    order = scores.argsort(dim=1, descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(1, scores.shape[1] + 1, device=scores.device).expand_as(order))
    return ranks


def zscore(scores: torch.Tensor) -> torch.Tensor:
    return (scores - scores.mean(dim=1, keepdim=True)) / scores.std(dim=1, keepdim=True)


def correct_ranks(queries: List[torch.Tensor], cands: List[torch.Tensor], k: float, chunk: int) -> Dict[str, torch.Tensor]:
    """Rank of the paired candidate (index i for query i) under RRF, z-fusion,
    and each single model."""
    n = queries[0].shape[0]
    device = queries[0].device
    out = {"rrf": torch.empty(n, dtype=torch.long), "zfusion": torch.empty(n, dtype=torch.long)}
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        rows = torch.arange(end - start, device=device)
        cols = torch.arange(start, end, device=device)
        rrf = torch.zeros(end - start, n, dtype=torch.float64, device=device)
        zsum = torch.zeros(end - start, n, dtype=torch.float64, device=device)
        cos_sum = torch.zeros(end - start, n, dtype=torch.float64, device=device)
        for q, c in zip(queries, cands):
            s = q[start:end] @ c.t()
            rrf += 1.0 / (k + ranks_desc(s).double())
            zsum += zscore(s).double()
            cos_sum += s.double()
        for name, fused in (("rrf", rrf + 1e-13 * cos_sum / len(queries)), ("zfusion", zsum)):
            diag = fused[rows, cols].unsqueeze(1)
            out[name][start:end] = ((fused > diag).sum(dim=1) + 1).cpu()
    return out


def single_ranks(q: torch.Tensor, c: torch.Tensor, chunk: int) -> torch.Tensor:
    n = q.shape[0]
    out = torch.empty(n, dtype=torch.long)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        s = q[start:end] @ c.t()
        diag = s[torch.arange(end - start), torch.arange(start, end)].unsqueeze(1)
        out[start:end] = ((s > diag).sum(dim=1) + 1).cpu()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--phase", default="finetuned")
    ap.add_argument("--k", type=float, default=60.0)
    ap.add_argument("--k-sweep", type=float, nargs="*", default=[1, 10, 30, 60, 100, 300],
                    help="Extra RRF k values evaluated for the all-model combination.")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--output", default=None, help="JSON output path (default: <emb-dir>/fusion_results.json)")
    args = ap.parse_args()

    emb_dir = Path(args.emb_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emb = {}
    ref_paths = None
    for m in args.models:
        img, txt, paths = load_embeddings(emb_dir, m, args.phase)
        if ref_paths is None:
            ref_paths = paths
        elif not np.array_equal(paths, ref_paths):
            raise SystemExit(f"Sample order of {m} differs from {args.models[0]}; fusion would pair the wrong rows.")
        emb[m] = (img.to(device), txt.to(device))
    n = len(ref_paths)
    print(f"Loaded {len(args.models)} models x {n} pairs on {device}\n")

    def qc(direction: str, models) -> tuple:
        # text_to_image: queries are captions, candidates are images.
        qi = 1 if direction == "text_to_image" else 0
        return [emb[m][qi] for m in models], [emb[m][1 - qi] for m in models]

    results: Dict[str, Dict] = {}
    for m in args.models:
        results[SHORT.get(m, m)] = {d: _metrics_from_ranks(single_ranks(*[x[0] for x in qc(d, [m])], args.chunk)) for d in DIRECTIONS}

    for size in range(2, len(args.models) + 1):
        for combo in itertools.combinations(args.models, size):
            label = "+".join(SHORT.get(m, m) for m in combo)
            per_dir = {d: correct_ranks(*qc(d, combo), k=args.k, chunk=args.chunk) for d in DIRECTIONS}
            results[f"RRF(k={args.k:g}) {label}"] = {d: _metrics_from_ranks(per_dir[d]["rrf"]) for d in DIRECTIONS}
            results[f"zFusion {label}"] = {d: _metrics_from_ranks(per_dir[d]["zfusion"]) for d in DIRECTIONS}

    sweep = {}
    for k in args.k_sweep:
        per_dir = {d: correct_ranks(*qc(d, args.models), k=k, chunk=args.chunk)["rrf"] for d in DIRECTIONS}
        sweep[k] = {d: _metrics_from_ranks(per_dir[d]) for d in DIRECTIONS}

    best_single = max((results[SHORT.get(m, m)]["text_to_image"]["recall@1"], SHORT.get(m, m)) for m in args.models)
    hdr = f"{'method':<34}" + "".join(f"{h:>8}" for h in ("R@1", "R@5", "R@10", "MRR", "MRank")) + "  |" + "".join(f"{h:>8}" for h in ("R@1", "R@5", "MRR", "MRank")) + f"{'Δt2iR@1':>10}"
    print(f"{'':<34}{'------------ text -> image ------------':<40}  |{'------ image -> text ------':<32}")
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        t, i = r["text_to_image"], r["image_to_text"]
        print(f"{name:<34}{t['recall@1']:>8.4f}{t['recall@5']:>8.4f}{t['recall@10']:>8.4f}{t['mrr']:>8.4f}{t['mean_rank']:>8.2f}  |"
              f"{i['recall@1']:>8.4f}{i['recall@5']:>8.4f}{i['mrr']:>8.4f}{i['mean_rank']:>8.2f}{t['recall@1'] - best_single[0]:>+10.4f}")
    print(f"\nΔ is vs the best single model on text->image R@1 ({best_single[1]} = {best_single[0]:.4f}).")

    print(f"\nRRF k sweep, all {len(args.models)} models:")
    print(f"{'k':>6}{'t2i R@1':>10}{'t2i MRR':>10}{'i2t R@1':>10}{'i2t MRR':>10}")
    for k, r in sweep.items():
        print(f"{k:>6g}{r['text_to_image']['recall@1']:>10.4f}{r['text_to_image']['mrr']:>10.4f}{r['image_to_text']['recall@1']:>10.4f}{r['image_to_text']['mrr']:>10.4f}")

    output = Path(args.output) if args.output else emb_dir / "fusion_results.json"
    with open(output, "w") as f:
        json.dump({"n_samples": n, "k": args.k, "results": results, "k_sweep": {str(k): v for k, v in sweep.items()}}, f, indent=2)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
