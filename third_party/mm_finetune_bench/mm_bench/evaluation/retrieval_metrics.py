"""Recall@K / MRR / Mean Rank for image<->text retrieval.

Assumes a 1:1 image-caption pairing within the evaluated batch: row ``i`` of
``image_embeds`` is the positive match for row ``i`` of ``text_embeds`` (this
holds for the benchmark split, which is built 1 caption per keyframe -- see
``mm_bench.data.splits``). Both embedding matrices must already be
L2-normalized (every model wrapper's ``encode_image``/``encode_text``
guarantees this).
"""

from __future__ import annotations

from typing import Dict

import torch


def _ranks_of_correct_match(similarity: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
    """``similarity[i, j]`` = score of query ``i`` against candidate ``j``.
    Returns the 1-indexed rank of the correct candidate (``j == i``) for
    every query, i.e. ``1`` means the top-1 result was correct.

    Computed by counting how many candidates outscore the correct one,
    rather than by sorting: an ``argsort`` over a 16k x 16k matrix costs
    O(N^2 log N) and materializes a 2 GB int64 index tensor, while the count
    is a single O(N^2) pass over one bool chunk at a time. Ties are broken
    pessimistically-but-fairly with a strict ``>`` (an exact tie with the
    correct candidate does not push it down a rank).
    """
    n = similarity.shape[0]
    correct_scores = similarity.diagonal().unsqueeze(1)  # (N, 1)
    ranks = torch.empty(n, dtype=torch.long, device=similarity.device)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        block = similarity[start:end]
        ranks[start:end] = (block > correct_scores[start:end]).sum(dim=1) + 1
    return ranks


def _metrics_from_ranks(ranks: torch.Tensor, ks=(1, 5, 10)) -> Dict[str, float]:
    n = ranks.shape[0]
    out = {f"recall@{k}": (ranks <= k).float().mean().item() for k in ks}
    out["mrr"] = (1.0 / ranks.float()).mean().item()
    out["mean_rank"] = ranks.float().mean().item()
    out["n_samples"] = n
    return out


def compute_retrieval_metrics(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    ks=(1, 5, 10),
    device: str = "auto",
) -> Dict[str, Dict[str, float]]:
    """Compute both retrieval directions.

    Args:
        device: where to build the (N, N) similarity matrix. ``"auto"`` uses
            CUDA when available -- for a 16k-sample benchmark the matmul is
            ~400 GFLOP, which is seconds on a GPU and minutes on CPU.

    Returns:
        {
          "image_to_text": {recall@1, recall@5, recall@10, mrr, mean_rank, n_samples},
          "text_to_image": {...same keys...},
        }
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    image_embeds = image_embeds.to(device=device, dtype=torch.float32)
    text_embeds = text_embeds.to(device=device, dtype=torch.float32)

    similarity = image_embeds @ text_embeds.t()  # (N, N)

    i2t_ranks = _ranks_of_correct_match(similarity)
    t2i_ranks = _ranks_of_correct_match(similarity.t())

    return {
        "image_to_text": _metrics_from_ranks(i2t_ranks, ks),
        "text_to_image": _metrics_from_ranks(t2i_ranks, ks),
    }


def flatten_metrics(metrics: Dict[str, Dict[str, float]], prefix_sep: str = "/") -> Dict[str, float]:
    """Flatten the nested dict above into ``{"image_to_text/recall@1": ...}``
    for easy CSV/DataFrame construction in ``evaluation.compare``.
    """
    flat = {}
    for direction, direction_metrics in metrics.items():
        for k, v in direction_metrics.items():
            flat[f"{direction}{prefix_sep}{k}"] = v
    return flat
