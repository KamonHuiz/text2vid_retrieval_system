#!/usr/bin/env python3
"""Nạp data/ocr_output/<lot>/<video>.json (đi kèm repo, Chandra OCR 2, 873/873 video) vào
Typesense. Cần fps để tính timestamp_sec — lấy từ data/video_meta.json (đi kèm repo).

Dùng:
    python scripts/ingest_ocr_typesense.py
"""
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config  # noqa: E402

COLLECTION = "utots_ocr_frames"
SCHEMA = {
    "name": COLLECTION,
    "fields": [
        {"name": "video_id", "type": "string", "facet": True},
        {"name": "lot", "type": "string", "facet": True},
        {"name": "frame_id", "type": "string"},
        {"name": "frame_index", "type": "int32", "sort": True},
        {"name": "timestamp_sec", "type": "float", "optional": True},
        {"name": "ocr_text", "type": "string"},
    ],
}


def main():
    headers = {"X-TYPESENSE-API-KEY": config.TYPESENSE_KEY}
    base = config.TYPESENSE_URL

    requests.delete(f"{base}/collections/{COLLECTION}", headers=headers)
    res = requests.post(f"{base}/collections", headers=headers, json=SCHEMA)
    res.raise_for_status()
    print(f"collection '{COLLECTION}' đã tạo")

    fps_map = json.loads(config.VIDEO_META_FILE.read_text()) if config.VIDEO_META_FILE.exists() else {}

    docs, total = [], 0
    for video_json in sorted(config.OCR_OUTPUT_DIR.glob("*/*.json")):
        video_id = video_json.stem
        lot = video_json.parent.name
        fps = fps_map.get(video_id)
        try:
            table = json.loads(video_json.read_text())
        except json.JSONDecodeError:
            continue
        for frame_id, text in table.items():
            if not text.strip():
                continue
            frame_index = int(frame_id)
            doc = {
                "id": f"{video_id}_{frame_id}",
                "video_id": video_id, "lot": lot, "frame_id": frame_id,
                "frame_index": frame_index, "ocr_text": text,
            }
            if fps:  # field "optional" trong schema -- Typesense muốn field VẮNG MẶT,
                doc["timestamp_sec"] = round(frame_index / fps, 2)  # không phải null
            docs.append(doc)
        if len(docs) >= 2000:
            _import(base, headers, docs)
            total += len(docs)
            docs = []
    if docs:
        _import(base, headers, docs)
        total += len(docs)
    print(f"xong: {total} document trong '{COLLECTION}'")


def _import(base, headers, docs):
    payload = "\n".join(json.dumps(d, ensure_ascii=False) for d in docs)
    res = requests.post(f"{base}/collections/{COLLECTION}/documents/import",
                        headers=headers, params={"action": "upsert"}, data=payload.encode("utf-8"))
    res.raise_for_status()


if __name__ == "__main__":
    main()
