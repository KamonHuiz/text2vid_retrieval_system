"""Tìm chữ (OCR) / tìm lời nói (ASR) qua Typesense — cùng kỹ thuật KEYWORD/HYBRID/SEMANTIC
như hệ thống cũ (xem AIC_HCMC_2026_TONG_KET.md §5.2), nhưng nguồn dữ liệu khác:

  OCR   data/ocr_output/<lot>/<video>.json      (Chandra OCR 2, đã chạy xong 873/873)
  ASR   data/audio_output/<lot>/<video>.sentences.json  (pipeline audio-aic-2025)

Khác biệt quan trọng so với hệ thống cũ: bản ASR mới đã gán SẴN danh sách keyframe cho
từng câu (field `keyframes` trong sentences.json) — không cần đoán khoảng frame qua fps +
bisect như cũ nữa, chỉ cần đọc thẳng field đó ra. Đơn giản hơn và chính xác hơn.
"""
from __future__ import annotations

import requests
from fastapi import HTTPException

from . import config

TS_HEADERS = {"X-TYPESENSE-API-KEY": config.TYPESENSE_KEY}

OCR_COLLECTION = "utots_ocr_frames"
ASR_COLLECTION = "utots_asr_sentences"

# Ba "kỹ thuật" ánh xạ sang tham số Typesense — copy nguyên từ hệ thống cũ, đo thật vẫn còn
# đúng vì Typesense/schema chữ không đổi bản chất.
TECHNIQUES = {
    "KEYWORD": {"num_typos": 0, "drop_tokens_threshold": 0, "typo_tokens_threshold": 0},
    "HYBRID": {"num_typos": 2, "drop_tokens_threshold": 10, "typo_tokens_threshold": 10},
    "SEMANTIC": {"num_typos": 2, "drop_tokens_threshold": 30, "typo_tokens_threshold": 30},
}


def typesense_search(collection: str, field: str, query: str, technique: str, limit: int):
    params = {"q": query, "query_by": field, "per_page": min(limit, 250), "page": 1,
              "prefix": "false", **TECHNIQUES[technique]}
    docs, page = [], 1
    while len(docs) < limit:
        params["page"] = page
        try:
            res = requests.get(f"{config.TYPESENSE_URL}/collections/{collection}/documents/search",
                               headers=TS_HEADERS, params=params, timeout=30)
            res.raise_for_status()
            body = res.json()
        except requests.RequestException as exc:
            raise HTTPException(status_code=503, detail=f"Typesense unreachable: {exc}")
        hits = body.get("hits", [])
        if not hits:
            break
        docs.extend(hits)
        if len(docs) >= body.get("found", 0):
            break
        page += 1
        if page > 20:
            break
    return docs[:limit]


def ocr_search(query: str, technique: str, top_k: int) -> list[dict]:
    """-> [{video_id, frame_id, score, ocr_text}], score = RRF theo thứ hạng Typesense."""
    docs = typesense_search(OCR_COLLECTION, "ocr_text", query, technique, top_k)
    out = []
    for rank, hit in enumerate(docs, start=1):
        d = hit["document"]
        out.append({
            "video_id": d["video_id"], "frame_id": d["frame_id"],
            "score": 1.0 / (config.RRF_K + rank), "ocr_text": d.get("ocr_text", ""),
        })
    return out


def asr_search(query: str, technique: str, top_k: int) -> tuple[list[dict], list[dict]]:
    """-> (frames, segments).

    frames   mỗi keyframe mà câu khớp có gán sẵn (field `keyframes`), điểm = RRF theo thứ
             hạng câu; một keyframe rơi vào nhiều câu thì giữ điểm cao nhất.
    segments câu gốc đã khớp, để UI hiển thị context — giống hệt hệ thống cũ.
    """
    docs = typesense_search(ASR_COLLECTION, "text", query, technique, top_k)
    best: dict[tuple[str, str], dict] = {}
    segments = []
    for rank, hit in enumerate(docs, start=1):
        d = hit["document"]
        segments.append(d)
        score = 1.0 / (config.RRF_K + rank)
        for frame_id in d.get("keyframes", []):
            key = (d["video_id"], frame_id)
            if key not in best or score > best[key]["score"]:
                best[key] = {"score": score, "seg": d}
    frames = [{"video_id": v, "frame_id": f, "score": item["score"],
               "asr_text": item["seg"]["text"],
               "asr_t_start": item["seg"]["t_start"], "asr_t_end": item["seg"]["t_end"]}
              for (v, f), item in best.items()]
    frames.sort(key=lambda r: r["score"], reverse=True)
    return frames, segments
