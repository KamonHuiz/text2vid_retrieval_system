"""Trộn nhiều bảng xếp hạng thành một — copy nguyên công thức từ hệ thống cũ
(APITesting/main.py), xem AIC_HCMC_2026_TONG_KET.md §5.2 để đọc giải thích đầy đủ. Không
đổi gì cả, chỉ đổi kiểu key (video_id, frame_id) là chung cho mọi model vì cả 4 model đều
index cùng 329.931 keyframe.
"""
from __future__ import annotations

import statistics
from collections import defaultdict

from . import config


def rrf(rankings: dict[str, list[tuple[str, str, float]]], weights: dict[str, float]):
    """Reciprocal rank fusion — dùng khi điểm giữa các model không cùng thang, chỉ THỨ TỰ
    là so sánh được."""
    fused: dict[tuple[str, str], float] = defaultdict(float)
    for leg, hits in rankings.items():
        w = weights.get(leg, 1.0)
        for rank, (video, frame, _) in enumerate(hits, start=1):
            fused[(video, frame)] += w / (config.RRF_K + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


def zscore_fuse(rankings: dict[str, list[tuple[str, str, float]]], weights: dict[str, float]):
    """z = (điểm - trung bình) / độ lệch chuẩn — giữ được khoảng cách điểm, nhạy với outlier
    hơn rrf. Vắng mặt trong một bảng bị tính bằng z thấp nhất của bảng đó trừ thêm 0.5."""
    fused: dict[tuple[str, str], float] = defaultdict(float)
    for leg, hits in rankings.items():
        if not hits:
            continue
        scores = [h[2] for h in hits]
        mean = statistics.fmean(scores)
        sd = statistics.pstdev(scores) or 1e-6
        w = weights.get(leg, 1.0)
        floor = (min(scores) - mean) / sd - 0.5
        seen = set()
        for video, frame, score in hits:
            fused[(video, frame)] += w * (score - mean) / sd
            seen.add((video, frame))
        for key in list(fused):
            if key not in seen:
                fused[key] += w * floor
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
