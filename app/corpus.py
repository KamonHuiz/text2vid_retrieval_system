"""Metadata của corpus: fps, cú máy (shot), danh sách keyframe từng video, caption, OCR,
ASR. Mọi hàm ở đây đọc dữ liệu NHẸ đi kèm trong `data/` (xem SETUP.md), trừ danh sách
keyframe (`build_frame_index`) phải liệt kê thư mục KEYFRAME_ROOT bên ngoài.

Chuyển thể trực tiếp từ APITesting/main.py (hệ thống cũ, PE-Core/BEiT3) — cùng công thức,
chỉ đổi nguồn OCR/ASR sang data/ocr_output, data/audio_output (Chandra OCR 2 + pipeline ASR
mới, xem AIC_HCMC_2026_TONG_KET.md §3.2 để biết vì sao khác nguồn cũ).
"""
from __future__ import annotations

import bisect
import json
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from . import config

STATE: dict = {}


def load_fps_map() -> dict[str, float]:
    if config.VIDEO_META_FILE.exists():
        try:
            return json.loads(config.VIDEO_META_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def load_youtube_map() -> dict[str, str]:
    if config.YOUTUBE_MAP_FILE.exists():
        try:
            return json.loads(config.YOUTUBE_MAP_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def load_shot_index() -> dict[str, list[tuple]]:
    """video_id -> [(start_frame, end_frame, shot_index), ...] tăng dần theo start_frame.

    Dùng để gộp keyframe theo ĐÚNG cú cắt cảnh thay vì đoán bằng khoảng cách thời gian.
    """
    shots_dir = config.DATA_ROOT / "omnishotcut_shots"
    index: dict[str, list[tuple]] = {}
    if not shots_dir.is_dir():
        return index
    for path in sorted(shots_dir.glob("*/*.json")):
        try:
            rec = json.loads(path.read_text())
            shots = [(int(sh["start_frame"]), int(sh["end_frame"]), int(sh["shot_index"]))
                     for sh in rec.get("shots", [])]
        except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
            continue
        if shots:
            shots.sort()
            index[rec.get("video") or path.stem] = shots
    return index


def shot_of(video_id: str, frame_index: int) -> Optional[int]:
    shots = STATE.get("shots", {}).get(video_id)
    if not shots:
        return None
    pos = bisect.bisect_right(shots, (frame_index, float("inf"), float("inf"))) - 1
    if pos < 0:
        return None
    start, end, idx = shots[pos]
    return idx if start <= frame_index <= end else None


def build_frame_index() -> dict[str, list[int]]:
    """video_id -> danh sách số frame (keyframe) tăng dần, đọc từ thư mục ảnh THẬT trên đĩa.

    Cần KEYFRAME_ROOT (bên ngoài repo) — xem SETUP.md §1. API vẫn khởi động được nếu thư
    mục này rỗng/không tồn tại, chỉ là mọi search sẽ trả về rỗng cho tới khi trỏ đúng.
    """
    index: dict[str, list[int]] = {}
    if not config.KEYFRAME_ROOT.is_dir():
        return index
    for lot_dir in sorted(p for p in config.KEYFRAME_ROOT.iterdir() if p.is_dir()):
        for video_dir in sorted(p for p in lot_dir.iterdir() if p.is_dir()):
            frames = sorted(int(f.stem) for f in video_dir.iterdir()
                            if f.suffix == ".webp" and f.stem.isdigit())
            if frames:
                index[video_dir.name] = frames
    return index


def load_captions() -> dict[str, str]:
    """(video_id, frame_id) -> caption tiếng Anh, nạp một lần lúc khởi động.

    329.931 dòng ~ 220MB text, nạp hết vào RAM (một dict {video_id}/{frame_id} -> str) tốn
    khoảng vài trăm MB — chấp nhận được, rẻ hơn hẳn so với mở lại file mỗi request.
    """
    out: dict[str, str] = {}
    if not config.CAPTIONS_FILE.exists():
        return out
    with config.CAPTIONS_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out[f'{rec["video_id"]}/{rec["frame_id"]}'] = rec.get("caption", "") or ""
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def caption_of(video_id: str, frame_id: str) -> str:
    return STATE.get("captions", {}).get(f"{video_id}/{frame_id}", "")


def frame_path(video_id: str, frame_id: str) -> str:
    return str(config.KEYFRAME_ROOT / video_id.split("_")[0] / video_id / f"{frame_id}.webp")


def to_url(path: str) -> str:
    try:
        return "/images/" + str(Path(path).relative_to(config.KEYFRAME_ROOT))
    except ValueError:
        return path


def video_path(video_id: str) -> Path:
    return config.VIDEO_ROOT / video_id.split("_")[0] / f"{video_id}.mp4"


def video_url(video_id: str) -> Optional[str]:
    """URL local (qua StaticFiles /videos) nếu video gốc có trên máy này, None nếu không --
    UI rơi về link YouTube trong trường hợp đó."""
    if not video_path(video_id).is_file():
        return None
    return f"/videos/{video_id.split('_')[0]}/{video_id}.mp4"


# ---------------------------------------------------------------------------
# OCR — data/ocr_output/<lot>/<video>.json = {frame_id: text} (Chandra OCR 2)
# ---------------------------------------------------------------------------

def ocr_of(video_id: str) -> dict[str, str]:
    """frame_id -> chữ OCR, nạp theo video và LRU-cache. Xem data/ocr_output đi kèm repo —
    không cần external gì thêm, đã chạy xong 873/873 video."""
    cache: OrderedDict = STATE.setdefault("ocr_cache", OrderedDict())
    if video_id in cache:
        cache.move_to_end(video_id)
        return cache[video_id]
    lot = video_id.split("_")[0]
    path = config.OCR_OUTPUT_DIR / lot / f"{video_id}.json"
    table = {}
    try:
        table = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        table = {}
    cache[video_id] = table
    while len(cache) > 128:
        cache.popitem(last=False)
    return table


# ---------------------------------------------------------------------------
# ASR — data/audio_output/<lot>/<video>.sentences.json (đã có sẵn field `keyframes` cho
# từng câu, khác hệ thống cũ: không cần đoán khoảng frame bằng fps nữa, chiếu THẲNG bằng
# danh sách keyframe mà pipeline nguồn đã gán sẵn cho câu đó).
# ---------------------------------------------------------------------------

def asr_of(video_id: str) -> list[dict]:
    """-> [{text, t_start, t_end, frame_start, frame_end, keyframes}, ...].

    Đổi tên field từ sentences.json gốc (`start`/`end`/`start_frame`/`end_frame`) sang tên
    mà UI + /ocr_asr_search đều dùng chung (`t_start`/`t_end`/`frame_start`/`frame_end`) --
    KHÁC tên gốc trong file nguồn, đổi ở ĐÚNG MỘT CHỖ này để hai đường (video_meta và
    Typesense) không bao giờ lệch tên field với nhau nữa.
    """
    cache: OrderedDict = STATE.setdefault("asr_cache", OrderedDict())
    if video_id in cache:
        cache.move_to_end(video_id)
        return cache[video_id]
    lot = video_id.split("_")[0]
    path = config.AUDIO_OUTPUT_DIR / lot / f"{video_id}.sentences.json"
    sentences = []
    try:
        doc = json.loads(path.read_text())
        for s in doc.get("sentences", []):
            sentences.append({
                "text": s.get("text", ""),
                "t_start": s.get("start"), "t_end": s.get("end"),
                "frame_start": s.get("start_frame"), "frame_end": s.get("end_frame"),
                "keyframes": [Path(k).stem for k in s.get("keyframes", [])],
            })
    except (json.JSONDecodeError, OSError, KeyError):
        sentences = []
    cache[video_id] = sentences
    while len(cache) > 128:
        cache.popitem(last=False)
    return sentences


def video_meta_of(video_id: str) -> dict:
    """fps, YouTube URL, OCR theo frame, câu ASR — mọi thứ Frame Inspector cần cho 1 video."""
    fps = STATE["fps"].get(video_id, 0.0)
    return {
        "fps": fps,
        "video_url": STATE.get("youtube", {}).get(video_id, ""),
        "ocr": ocr_of(video_id),
        "asr": asr_of(video_id),
    }
