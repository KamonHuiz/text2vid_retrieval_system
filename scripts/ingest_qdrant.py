#!/usr/bin/env python3
"""Nạp embedding ảnh (.npz, đã tính sẵn — KHÔNG đi kèm repo, xem SETUP.md §2) vào Qdrant.

Mỗi model có 1 thư mục `{model}_full/` chứa `shard0.npz` + `shard1.npz`, mỗi file có key
`image` (float32, embedding ảnh), `text` (một số model là số 0 — xem registry.py
`text_field_real`), `paths` (đường dẫn ảnh gốc dùng lúc encode).

Point id dùng ĐÚNG hàm `point_id_for_path` của mm_bench (hash blake2b của chuỗi path, xem
third_party/mm_finetune_bench/mm_bench/indexing/qdrant_store.py) — khớp với collection mà
`scripts/encode_to_qdrant.py` gốc của team tạo ra, nên nạp lại/nạp thêm sau này (vd chạy
encode_to_qdrant.py --resume trực tiếp) vẫn nhận đúng điểm đã có, không tạo trùng.

Dùng:
    python scripts/ingest_qdrant.py --model BGE_VL
    python scripts/ingest_qdrant.py --model ALL     # cả 4 model, bỏ qua model chưa có embedding
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app import config  # noqa: E402
from app.models.registry import MODELS  # noqa: E402

sys.path.insert(0, str(config.MM_BENCH_ROOT))
from mm_bench.indexing.qdrant_store import QdrantVectorStore, point_id_for_path  # noqa: E402

_VIDEO_FRAME_RE = re.compile(r"(L\d+_V\d+).*?(\d{6})\.webp")


def video_frame_of(path: str) -> tuple[str, str] | None:
    m = _VIDEO_FRAME_RE.search(path)
    return (m.group(1), m.group(2)) if m else None


def ingest_model(model_key: str, qdrant_url: str, recreate: bool):
    spec = MODELS[model_key]
    emb_dir = Path(spec["embeddings_dir"])
    shards = sorted(emb_dir.glob("shard*.npz"))
    if not shards:
        print(f"[{model_key}] không tìm thấy shard*.npz trong {emb_dir} — bỏ qua "
              f"(kiểm tra UTOTS_MODEL_PACKAGE có trỏ đúng chỗ đã giải nén "
              f"package_4models_export chưa)")
        return

    store = None
    total = 0
    for shard_path in shards:
        print(f"[{model_key}] đọc {shard_path} ...")
        data = np.load(shard_path, allow_pickle=True)
        images, paths = data["image"], data["paths"]
        if store is None:
            store = QdrantVectorStore(spec["collection"], int(images.shape[1]), url=qdrant_url)
            store.ensure_collection(recreate=recreate)

        payloads, batch_paths, batch_vecs = [], [], []
        for i in range(len(paths)):
            path = str(paths[i])
            vf = video_frame_of(path)
            payloads.append({"path": path, "video_id": vf[0] if vf else None,
                             "frame_id": vf[1] if vf else None})
            batch_paths.append(path)
            batch_vecs.append(images[i])
            if len(batch_paths) >= 512:
                store.upsert_batch(batch_paths, np.stack(batch_vecs), payloads)
                total += len(batch_paths)
                payloads, batch_paths, batch_vecs = [], [], []
        if batch_paths:
            store.upsert_batch(batch_paths, np.stack(batch_vecs), payloads)
            total += len(batch_paths)
    if store is not None:
        store.finalize_index()
        print(f"[{model_key}] xong: {total} điểm nạp, collection '{spec['collection']}' hiện có {store.count()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=[*MODELS.keys(), "ALL"])
    ap.add_argument("--qdrant-url", default=config.QDRANT_URL)
    ap.add_argument("--recreate", action="store_true", help="xoá và tạo lại collection trước khi nạp")
    args = ap.parse_args()

    targets = list(MODELS.keys()) if args.model == "ALL" else [args.model]
    for key in targets:
        ingest_model(key, args.qdrant_url, args.recreate)


if __name__ == "__main__":
    main()
