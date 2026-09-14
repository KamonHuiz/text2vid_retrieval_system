#!/usr/bin/env python3
"""Nạp data/audio_output/<lot>/<video>.sentences.json (đi kèm repo, pipeline audio-aic-2025,
873/873 video) vào Typesense. Mỗi document là MỘT CÂU, đã kèm sẵn danh sách keyframe liên
quan (field `keyframes`) — không cần đoán khoảng frame bằng fps như hệ thống cũ.

Dùng:
    python scripts/ingest_asr_typesense.py
"""
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config  # noqa: E402

COLLECTION = "utots_asr_sentences"
SCHEMA = {
    "name": COLLECTION,
    "fields": [
        {"name": "video_id", "type": "string", "facet": True},
        {"name": "lot", "type": "string", "facet": True},
        {"name": "t_start", "type": "float"},
        {"name": "t_end", "type": "float"},
        {"name": "text", "type": "string"},
        {"name": "keyframes", "type": "string[]"},
    ],
}


def main():
    headers = {"X-TYPESENSE-API-KEY": config.TYPESENSE_KEY}
    base = config.TYPESENSE_URL

    requests.delete(f"{base}/collections/{COLLECTION}", headers=headers)
    res = requests.post(f"{base}/collections", headers=headers, json=SCHEMA)
    res.raise_for_status()
    print(f"collection '{COLLECTION}' đã tạo")

    docs, total = [], 0
    for video_json in sorted(config.AUDIO_OUTPUT_DIR.glob("*/*.sentences.json")):
        video_id = video_json.stem.removesuffix(".sentences")
        lot = video_json.parent.name
        try:
            doc = json.loads(video_json.read_text())
        except json.JSONDecodeError:
            continue
        for i, sent in enumerate(doc.get("sentences", [])):
            text = (sent.get("text") or "").strip()
            if not text:
                continue
            # keyframes trong sentences.json có đuôi .webp (vd "000067.webp") -- lưu KHÔNG
            # đuôi để khớp frame_id 6 chữ số dùng mọi nơi khác trong hệ thống.
            keyframes = [Path(k).stem for k in sent.get("keyframes", [])]
            docs.append({
                "id": f"{video_id}_{i}",
                "video_id": video_id, "lot": lot,
                "t_start": sent["start"], "t_end": sent["end"],
                "text": text, "keyframes": keyframes,
            })
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
