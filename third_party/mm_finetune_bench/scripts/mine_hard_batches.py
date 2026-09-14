#!/usr/bin/env python3
"""Mine hard-negative training batches from a reference model's embeddings.

Uses a fine-tuned reference model (BGE-VL-ft) as the judge of "what is
confusable": samples the reference still embeds close together are exactly
the pairs a random batch almost never puts side by side.

Pipeline
  1. kNN: top-K image neighbours of every training sample, via Qdrant
     query-by-point-id over the reference collection (HNSW).
  2. Duplicates: a neighbour pair whose image cosine >= --img-dup OR caption
     cosine >= --txt-dup is a near-duplicate -- each caption describes both
     images, so treating them as negatives would teach the model to push
     apart true matches. Duplicate pairs are union-found into ``dup_id``s;
     the trainer masks same-``dup_id`` pairs out of the contrastive loss.
  3. Groups: greedily grow groups of --group-size from random seeds using each
     seed's nearest *non-duplicate* unassigned neighbours.
  4. Batches: shuffle groups, lay them end to end, cut into --global-batch.
     Each sample therefore sees (group-size - 1) hard negatives plus
     (global-batch - group-size) ordinary ones.

The global batch is the all-gathered batch across GPUs (e.g. 2 x 128 = 256):
rank r trains on columns [r*local:(r+1)*local] of each row of ``batches``.

Pass --stats-only first to see the similarity distribution and pick the
duplicate thresholds; kNN results are cached so the second run is instant.

Usage:
    python scripts/mine_hard_batches.py \\
        --emb-dir /home/calypso/aic/data/emb/bge_vl_ft1_train \\
        --split /home/calypso/aic/data/splits/finetune_train.jsonl \\
        --collection aic_bge_vl_ft1_train --stats-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mm_bench.data.dataset import load_samples_from_split_file
from mm_bench.indexing.qdrant_store import point_id_for_path


def load_embeddings(emb_dir: Path, split_paths: list) -> tuple:
    parts = [np.load(p) for p in sorted(emb_dir.glob("shard*.npz"))]
    paths = np.concatenate([p["paths"] for p in parts])
    img = np.concatenate([p["image"] for p in parts])
    txt = np.concatenate([p["text"] for p in parts])
    row = {p: i for i, p in enumerate(paths)}
    missing = [p for p in split_paths if p not in row]
    if missing:
        raise SystemExit(f"{len(missing)} training samples have no embedding (e.g. {missing[:2]})")
    order = np.array([row[p] for p in split_paths])
    img, txt = img[order], txt[order]
    img /= np.linalg.norm(img, axis=1, keepdims=True)
    txt /= np.linalg.norm(txt, axis=1, keepdims=True)
    return torch.from_numpy(img), torch.from_numpy(txt)


def wait_for_index(client, collection: str, timeout: int = 1800) -> None:
    from qdrant_client import models

    t0 = time.time()
    while True:
        info = client.get_collection(collection)
        indexed, total = info.indexed_vectors_count or 0, info.points_count or 0
        if info.status == models.CollectionStatus.GREEN and indexed >= 0.99 * total:
            print(f"Qdrant index ready: {indexed}/{total} vectors indexed")
            return
        if time.time() - t0 > timeout:
            raise SystemExit(f"Qdrant index not ready after {timeout}s ({indexed}/{total})")
        print(f"  waiting for HNSW build: {indexed}/{total} indexed, status={info.status}")
        time.sleep(10)


def knn_qdrant(url: str, collection: str, split_paths: list, k: int, batch: int, hnsw_ef: int) -> np.ndarray:
    from qdrant_client import QdrantClient, models

    client = QdrantClient(url=url, timeout=300)
    wait_for_index(client, collection)
    pids = [point_id_for_path(p) for p in split_paths]
    idx_of = {pid: i for i, pid in enumerate(pids)}
    n = len(pids)
    nbrs = np.full((n, k), -1, dtype=np.int64)
    t0 = time.time()
    for start in range(0, n, batch):
        chunk = pids[start : start + batch]
        reqs = [models.QueryRequest(query=pid, limit=k + 1, params=models.SearchParams(hnsw_ef=hnsw_ef)) for pid in chunk]
        for i, res in enumerate(client.query_batch_points(collection_name=collection, requests=reqs)):
            ids = [idx_of[p.id] for p in res.points if p.id != chunk[i] and p.id in idx_of][:k]
            nbrs[start + i, : len(ids)] = ids
        if (start // batch) % 50 == 0:
            done = start + len(chunk)
            print(f"  kNN {done}/{n} ({done / (time.time() - t0):.0f} q/s)")
    return nbrs


def neighbour_cosines(emb: torch.Tensor, nbrs: np.ndarray, chunk: int = 8192) -> np.ndarray:
    n, k = nbrs.shape
    out = np.full((n, k), -2.0, dtype=np.float32)
    idx = torch.from_numpy(np.where(nbrs < 0, 0, nbrs))
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out[s:e] = torch.einsum("nd,nkd->nk", emb[s:e], emb[idx[s:e]]).numpy()
    out[nbrs < 0] = -2.0
    return out


def union_find_dups(nbrs: np.ndarray, dup_mask: np.ndarray) -> np.ndarray:
    parent = np.arange(nbrs.shape[0])

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(*np.nonzero(dup_mask)):
        a, b = find(i), find(int(nbrs[i, j]))
        if a != b:
            parent[a] = b
    return np.array([find(i) for i in range(len(parent))])


def build_groups(nbrs: np.ndarray, dup_id: np.ndarray, group_size: int, rng: np.random.Generator) -> list:
    n = nbrs.shape[0]
    assigned = np.zeros(n, dtype=bool)
    groups = []
    for seed in rng.permutation(n):
        if assigned[seed]:
            continue
        group, dups = [seed], {dup_id[seed]}
        assigned[seed] = True
        for j in nbrs[seed]:
            if len(group) == group_size:
                break
            if j < 0 or assigned[j] or dup_id[j] in dups:
                continue
            group.append(int(j))
            dups.add(dup_id[j])
            assigned[j] = True
        groups.append(group)
    return groups


def hardness(batches: np.ndarray, img: torch.Tensor, dup_id: np.ndarray, device: torch.device) -> float:
    """Mean over anchors of the highest image cosine to a non-duplicate batch-mate."""
    img = img.to(device)
    dup = torch.from_numpy(dup_id).to(device)
    vals = []
    for b in batches:
        ib = torch.from_numpy(b).to(device)
        s = img[ib] @ img[ib].t()
        same = dup[ib].unsqueeze(0) == dup[ib].unsqueeze(1)
        s.masked_fill_(same, -2.0)  # also removes the diagonal
        vals.append(s.max(dim=1).values.mean().item())
    return float(np.mean(vals))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--knn", type=int, default=32)
    ap.add_argument("--hnsw-ef", type=int, default=128)
    ap.add_argument("--query-batch", type=int, default=256)
    ap.add_argument("--global-batch", type=int, default=256)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--img-dup", type=float, default=0.95)
    ap.add_argument("--txt-dup", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--output", default=None, help="Default: <emb-dir>/hard_batches_b<B>_g<G>.npz")
    args = ap.parse_args()

    emb_dir = Path(args.emb_dir)
    split_paths = [s.image_path for s in load_samples_from_split_file(args.split)]
    n = len(split_paths)
    img, txt = load_embeddings(emb_dir, split_paths)
    print(f"Loaded {n} x {img.shape[1]}-d image/caption embeddings")

    cache = emb_dir / f"knn{args.knn}.npz"
    if cache.exists():
        nbrs = np.load(cache)["nbrs"]
        print(f"Using cached kNN: {cache}")
    else:
        nbrs = knn_qdrant(args.qdrant_url, args.collection, split_paths, args.knn, args.query_batch, args.hnsw_ef)
        np.savez(cache, nbrs=nbrs)
    img_cos = neighbour_cosines(img, nbrs)
    txt_cos = neighbour_cosines(txt, nbrs)

    valid = nbrs >= 0
    print(f"\nNeighbours found: {valid.sum() / n:.1f} of {args.knn} per sample on average")
    print("Nearest-neighbour cosine percentiles        p10     p25     p50     p75     p90     p99")
    for name, c in (("image (nn1)", img_cos[:, 0]), ("caption of image-nn1", txt_cos[:, 0])):
        print(f"  {name:<40}" + "".join(f"{np.percentile(c, q):8.3f}" for q in (10, 25, 50, 75, 90, 99)))
    print("\nShare of kNN pairs flagged duplicate at threshold:   0.90    0.93    0.95    0.97    0.99")
    for name, c in (("image cosine", img_cos), ("caption cosine", txt_cos)):
        print(f"  {name:<50}" + "".join(f"{100 * (c[valid] >= t).mean():7.2f}%" for t in (0.90, 0.93, 0.95, 0.97, 0.99)))
    if args.stats_only:
        return

    dup_mask = valid & ((img_cos >= args.img_dup) | (txt_cos >= args.txt_dup))
    dup_id = union_find_dups(nbrs, dup_mask)
    sizes = np.bincount(np.unique(dup_id, return_inverse=True)[1])
    print(f"\nDuplicate clusters: {len(sizes)} for {n} samples | largest {sizes.max()} | samples in multi-member clusters {sizes[sizes > 1].sum()}")

    rng = np.random.default_rng(args.seed)
    groups = build_groups(nbrs, dup_id, args.group_size, rng)
    gsize = np.bincount([len(g) for g in groups], minlength=args.group_size + 1)
    print(f"Groups: {len(groups)} | size histogram " + ", ".join(f"{s}:{gsize[s]}" for s in range(1, args.group_size + 1)))

    order = rng.permutation(len(groups))
    flat = np.concatenate([np.array(groups[i], dtype=np.int64) for i in order])
    num_batches = len(flat) // args.global_batch
    batches = flat[: num_batches * args.global_batch].reshape(num_batches, args.global_batch).astype(np.int32)
    print(f"Batches: {num_batches} x {args.global_batch} (dropped {len(flat) - batches.size} leftover samples)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rand = rng.permutation(n)[: batches.size].reshape(batches.shape)
    h_mined, h_rand = hardness(batches, img, dup_id, device), hardness(rand, img, dup_id, device)
    print(f"Hardness (mean max non-dup image cosine in batch): mined {h_mined:.3f} vs random {h_rand:.3f}")

    out = Path(args.output) if args.output else emb_dir / f"hard_batches_b{args.global_batch}_g{args.group_size}.npz"
    np.savez(out, batches=batches, dup_id=dup_id.astype(np.int64), paths=np.array(split_paths))
    stats = {"n": n, "knn": args.knn, "group_size": args.group_size, "global_batch": args.global_batch,
             "img_dup": args.img_dup, "txt_dup": args.txt_dup, "num_batches": int(num_batches),
             "dup_clusters": int(len(sizes)), "largest_dup_cluster": int(sizes.max()),
             "hardness_mined": h_mined, "hardness_random": h_rand}
    out.with_suffix(".json").write_text(json.dumps(stats, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
