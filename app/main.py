"""UTOTS_SYSTEM search API — 4 model fine-tune (BGE-VL, DFN5B, SigLIP2, PE-Core-ft2) qua
Qdrant + OCR (Chandra OCR 2) / ASR (audio-aic-2025) qua Typesense.

Kiến trúc/công thức trộn điểm giống hệ thống cũ (APITesting/main.py trong repo gốc) — xem
AIC_HCMC_2026_TONG_KET.md §5 để đọc giải thích RRF/zscore đầy đủ. Khác biệt chính:

  * 4 model thay vì 2 (PE-Core gốc + BEiT3), tất cả cùng một giao diện TextEncoder nên
    main_search không cần nhánh code riêng cho từng model như bản cũ.
  * OCR/ASR đọc từ data/ đi kèm repo (Chandra OCR 2 + audio-aic-2025), không phải nguồn
    outsourced cũ — xem app/ocr_asr.py.
  * Qdrant/Typesense là instance RIÊNG (port khác hệ thống cũ, xem app/config.py) để không
    đụng vào API đang host — mỗi máy tự dựng, xem SETUP.md.

Run:  uv run uvicorn app.main:app --host 0.0.0.0 --port 8091
"""
from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, corpus, fusion, ocr_asr
from .models.base import ModelUnavailable
from .models.registry import COMBOS, MODELS

STATE: dict = {}


# ---------------------------------------------------------------------------
# dựng frame / view (envelope) -- copy nguyên công thức hệ thống cũ
# ---------------------------------------------------------------------------

def make_frame(video_id: str, frame_id: str, score: float, extra: dict = None) -> dict:
    fps = corpus.STATE["fps"].get(video_id, 0.0)
    try:
        frame_index = int(frame_id)
    except (TypeError, ValueError):
        frame_index = 0
    path = corpus.frame_path(video_id, frame_id)
    row = {
        "frame_id": frame_id, "video_id": video_id, "frame_index": frame_index,
        "timestamp_sec": round(frame_index / fps, 2) if fps else None,
        "shot_index": corpus.shot_of(video_id, frame_index),
        "image_path": path, "image_url": corpus.to_url(path),
        "score": round(float(score), 6),
        "caption": corpus.caption_of(video_id, frame_id),
    }
    if extra:
        row.update(extra)
    return row


def group_by_video(frames: list[dict]) -> "OrderedDict[str, list[dict]]":
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for f in frames:
        groups.setdefault(f["video_id"], []).append(f)
    return groups


def _payload(video_id: str, hits: list[dict]) -> dict:
    ordered = sorted(hits, key=lambda f: f["frame_id"])
    best = max(hits, key=lambda f: f["score"])
    return {"video_id": video_id, "frame_ids": [f["frame_id"] for f in ordered],
            "images_path": [f["image_path"] for f in ordered],
            "images_url": [f["image_url"] for f in ordered],
            "num_frames": len(ordered), "caption": best.get("caption", "")}


def make_views(frames: list[dict]) -> list[dict]:
    groups = group_by_video(frames)
    return [{"rank": i, **_payload(v, h)} for i, (v, h) in enumerate(groups.items(), 1)]


def envelope(frames: list[dict], **fields) -> dict:
    for i, f in enumerate(frames, 1):
        f["rank"] = i
    return {"status": "success", **fields, "total_frames": len(frames),
            "imageBased": frames, "videoBasedType1": make_views(frames)}


# ---------------------------------------------------------------------------
# dense retrieval -- một model bất kỳ trong registry
# ---------------------------------------------------------------------------

def query_qdrant(collection: str, vector, top_k: int) -> list[tuple[str, str, float]]:
    hits = STATE["qdrant"].query_points(
        collection_name=collection, query=vector.tolist(), limit=top_k,
        search_params={"exact": True}).points
    out = []
    for h in hits:
        p = h.payload or {}
        video_id, frame_id = p.get("video_id"), p.get("frame_id")
        if video_id and frame_id:
            out.append((video_id, frame_id, float(h.score)))
    return out


def encoder_of(model_key: str):
    enc = STATE["encoders"].get(model_key)
    if enc is None:
        raise HTTPException(status_code=400, detail=f"model không tồn tại: {model_key}")
    return enc


def require_ready(model_key: str):
    enc = encoder_of(model_key)
    if not enc.ready:
        raise HTTPException(status_code=503, detail=enc.error)
    return enc


# ---------------------------------------------------------------------------
# dịch câu truy vấn Việt -> Anh -- copy nguyên hệ thống cũ (4 model đều học tiếng Anh)
# ---------------------------------------------------------------------------
TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"
TRANSLATE_CACHE: "OrderedDict[str, str]" = OrderedDict()
TRANSLATE_CACHE_MAX = 4000
_translate_lock = threading.Lock()
_VI_CHARS = set("ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụưừửữứựỳỷỹýỵ")
_VI_WORDS = {"nguoi", "cua", "mot", "dang", "tren", "trong", "voi", "khong", "nhung",
             "duoc", "nhieu", "canh", "quay", "dan", "ong", "cai", "chiec", "dung", "ngoi"}


def looks_vietnamese(text: str) -> bool:
    low = text.lower()
    if any(c in _VI_CHARS for c in low):
        return True
    return bool(_VI_WORDS & set(re.findall(r"[a-z]+", low)))


GOOGLE_CLIENTS = ("gtx", "dict-chrome-ex", "at")
_google_client = 0


def _google_translate(text: str) -> str:
    global _google_client
    last = None
    for step in range(len(GOOGLE_CLIENTS)):
        idx = (_google_client + step) % len(GOOGLE_CLIENTS)
        try:
            res = requests.get(TRANSLATE_URL, params={"client": GOOGLE_CLIENTS[idx], "sl": "vi",
                                                       "tl": "en", "dt": "t", "q": text}, timeout=6)
            res.raise_for_status()
            out = "".join(part[0] for part in res.json()[0] if part and part[0]).strip()
            if out:
                _google_client = idx
                return out
            last = RuntimeError("empty")
        except Exception as exc:
            last = exc
    raise last if last else RuntimeError("google: no client worked")


def translate_vi_en(text: str) -> dict:
    text = text.strip()
    if not text:
        return {"text": text, "backend": "none", "error": None}
    key = f"google|{text}"
    with _translate_lock:
        hit = TRANSLATE_CACHE.get(key)
    if hit is not None:
        return {"text": hit, "backend": "cache", "error": None}
    try:
        out = _google_translate(text)
        with _translate_lock:
            TRANSLATE_CACHE[key] = out
            while len(TRANSLATE_CACHE) > TRANSLATE_CACHE_MAX:
                TRANSLATE_CACHE.popitem(last=False)
        return {"text": out, "backend": "google", "error": None}
    except Exception as exc:
        # KHÔNG trả về nguyên câu tiếng Việt: encode thẳng tiếng Việt ra vector vô nghĩa
        # -- search "chạy" nhưng hỏng âm thầm, tệ hơn báo lỗi rõ ràng.
        return {"text": text, "backend": "none", "error": f"{type(exc).__name__}: {exc}"}


def maybe_translate(query: str, mode: str) -> dict:
    if mode == "off" or (mode == "auto" and not looks_vietnamese(query)):
        return {"text": query, "backend": "skip", "error": None}
    return translate_vi_en(query)


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    from qdrant_client import QdrantClient

    from .models.mm_encoder import load_all

    STATE["qdrant"] = QdrantClient(url=config.QDRANT_URL)
    STATE["fps"] = corpus.load_fps_map()
    STATE["youtube"] = corpus.load_youtube_map()
    corpus.STATE["fps"] = STATE["fps"]
    corpus.STATE["youtube"] = STATE["youtube"]
    corpus.STATE["shots"] = corpus.load_shot_index()
    print("[startup] đang nạp captions ...")
    corpus.STATE["captions"] = corpus.load_captions()
    print(f"[startup] {len(corpus.STATE['captions'])} caption")
    print("[startup] đang liệt kê keyframe từ KEYFRAME_ROOT ...")
    corpus.STATE["frames"] = corpus.build_frame_index()
    print(f"[startup] {len(corpus.STATE['frames'])} video có keyframe")

    print("[startup] đang load 4 model (mất vài phút mỗi model LoRA) ...")
    STATE["encoders"] = load_all()

    yield
    STATE.clear()


app = FastAPI(title="UTOTS_SYSTEM", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

if config.KEYFRAME_ROOT.is_dir():
    app.mount("/images", StaticFiles(directory=str(config.KEYFRAME_ROOT)), name="images")


class TranslateRequest(BaseModel):
    text: str


class VisualSearchRequest(BaseModel):
    query: str
    top_k: int = 200
    # tên model trong registry (BGE_VL/DFN5B/SIGLIP2/PE_CORE_FT2), hoặc ALL = trộn RRF mọi
    # model đang ready
    search_mode: str = "BGE_VL"
    weights: dict[str, float] = {}     # dùng khi search_mode=ALL, mặc định 1.0 mỗi model
    translate: Literal["auto", "on", "off"] = "auto"
    debug_timing: bool = False


class ImageSearchRequest(BaseModel):
    image: Optional[str] = None
    video_id: Optional[str] = None
    frame_id: Optional[str] = None
    text: str = ""
    text_weight: float = 0.7
    image_weight: float = 0.3
    top_k: int = 200
    search_mode: str = "BGE_VL"
    translate: Literal["auto", "on", "off"] = "auto"


class TextSearchRequest(BaseModel):
    query: str
    search_mode: Literal["OCR", "ASR"] = "ASR"
    search_technique: Literal["KEYWORD", "SEMANTIC", "HYBRID"] = "HYBRID"
    top_k: int = 200


def decode_image(data: str):
    import base64
    import io

    from PIL import Image

    data = (data or "").strip()
    if data.startswith("http://") or data.startswith("https://"):
        res = requests.get(data, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        res.raise_for_status()
        return Image.open(io.BytesIO(res.content))
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data)))


@app.get("/")
async def root():
    return {"message": "UTOTS_SYSTEM"}


@app.get("/ui", include_in_schema=False)
async def ui():
    from fastapi.responses import FileResponse

    page = Path(__file__).resolve().parent.parent / "ui.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="ui.html not found")
    return FileResponse(page, media_type="text/html")


@app.get("/videos")
def videos():
    lots: dict[str, list] = {}
    for video_id, frames in sorted(corpus.STATE["frames"].items()):
        lots.setdefault(video_id.split("_")[0], []).append(
            {"video_id": video_id, "frames": len(frames)})
    return {"status": "success", "total_videos": len(corpus.STATE["frames"]),
            "lots": [{"lot": lot, "videos": vids} for lot, vids in sorted(lots.items())]}


@app.get("/video_meta/{video_id}")
def video_meta(video_id: str, center: Optional[int] = None, radius: int = 100,
                     mode: Literal["neighbors", "range"] = "neighbors"):
    import bisect

    frames = corpus.STATE["frames"].get(video_id)
    if not frames:
        raise HTTPException(status_code=404, detail=f"unknown video_id '{video_id}'")

    info = corpus.video_meta_of(video_id)
    fps = info["fps"]
    window = frames
    if center is not None:
        if mode == "range":
            lo, hi = center - radius, center + radius
            window = [n for n in frames if lo <= n <= hi]
        else:
            pos = bisect.bisect_left(frames, center)
            window = frames[max(0, pos - radius): pos + radius + 1]

    url = info["video_url"]
    youtube_id = ""
    if "v=" in url:
        youtube_id = url.split("v=", 1)[1].split("&", 1)[0]
    elif "youtu.be/" in url:
        youtube_id = url.split("youtu.be/", 1)[1].split("?", 1)[0]

    rows = []
    for number in window:
        frame_id = f"{number:06d}"
        rows.append({
            "frame_id": frame_id, "frame_index": number,
            "timestamp_sec": round(number / fps, 2) if fps else None,
            "image_url": corpus.to_url(corpus.frame_path(video_id, frame_id)),
            "ocr": info["ocr"].get(frame_id, ""),
            "caption": corpus.caption_of(video_id, frame_id),
            "shot_index": corpus.shot_of(video_id, number),
        })

    return {"video_id": video_id, "fps": fps, "video_url": url, "youtube_id": youtube_id,
            "total_frames": len(frames), "first_frame": frames[0], "last_frame": frames[-1],
            "returned": len(rows), "frames": rows, "asr": info["asr"]}


@app.get("/health")
async def health():
    out = {"status": "ok", "keyframe_root": str(config.KEYFRAME_ROOT),
           "videos": len(corpus.STATE.get("frames", {})),
           "captions": len(corpus.STATE.get("captions", {}))}
    out["models"] = {name: ("ready" if enc.ready else enc.error)
                     for name, enc in STATE.get("encoders", {}).items()}
    try:
        for name, spec in MODELS.items():
            coll = spec["collection"]
            if STATE["qdrant"].collection_exists(coll):
                out[coll] = STATE["qdrant"].count(collection_name=coll, exact=True).count
            else:
                out[coll] = "collection chưa tạo -- chạy scripts/ingest_qdrant.py"
    except Exception as exc:
        out["qdrant_error"] = str(exc)
    try:
        for name in (ocr_asr.OCR_COLLECTION, ocr_asr.ASR_COLLECTION):
            info = requests.get(f"{config.TYPESENSE_URL}/collections/{name}",
                                headers=ocr_asr.TS_HEADERS, timeout=10).json()
            out[name] = info.get("num_documents", "collection chưa tạo")
    except requests.RequestException as exc:
        out["typesense_error"] = str(exc)
    return out


@app.post("/main_search")
def main_search(request: VisualSearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    if request.top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be >= 1")

    clock = time.perf_counter
    t = {}
    t0 = clock()
    tr = maybe_translate(request.query, request.translate)
    text = tr["text"]
    t["translate"] = clock() - t0; t0 = clock()

    combo = COMBOS.get(request.search_mode)
    if combo:
        missing = [k for k in combo["models"] if not STATE["encoders"][k].ready]
        if missing:
            reasons = "; ".join(f"{k}: {STATE['encoders'][k].error}" for k in missing)
            raise HTTPException(status_code=503,
                                detail=f"{request.search_mode} cần {missing} nhưng chưa sẵn sàng — {reasons}")
        legs = {}
        for key in combo["models"]:
            vec = STATE["encoders"][key].encode_text(text)
            legs[key] = query_qdrant(MODELS[key]["collection"], vec, request.top_k)
        t["encode_and_search"] = clock() - t0; t0 = clock()
        weights = {k: request.weights.get(k, 1.0) for k in legs}
        fuse = fusion.zscore_fuse if combo["method"] == "zscore" else fusion.rrf
        frames = [make_frame(v, f, s) for (v, f), s in fuse(legs, weights)[:request.top_k]]
    else:
        enc = require_ready(request.search_mode)
        vec = enc.encode_text(text)
        t["encode"] = clock() - t0; t0 = clock()
        hits = query_qdrant(MODELS[request.search_mode]["collection"], vec, request.top_k)
        t["qdrant"] = clock() - t0; t0 = clock()
        frames = [make_frame(v, f, s) for v, f, s in hits]

    body = envelope(frames, query=request.query, search_mode=request.search_mode)
    if tr["backend"] not in ("skip", "none"):
        body["query_en"] = text
        body["translate_backend"] = tr["backend"]
    if tr["error"]:
        body["translate_error"] = tr["error"]
    if request.debug_timing:
        t["total_server"] = sum(t.values())
        body["timings"] = {k: round(v, 4) for k, v in t.items()}
    return body


@app.post("/image_search")
def image_search(request: ImageSearchRequest):
    if not request.image and not (request.video_id and request.frame_id):
        raise HTTPException(status_code=400, detail="cần image (base64/URL) hoặc video_id + frame_id")
    if request.search_mode not in MODELS:
        raise HTTPException(status_code=400,
                            detail=f"image_search cần đúng 1 model thật ({list(MODELS)}), "
                                   f"không dùng tổ hợp ({request.search_mode})")
    enc = require_ready(request.search_mode)

    pil = None
    if not (request.video_id and request.frame_id):
        try:
            pil = decode_image(request.image)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"không đọc được ảnh: {exc}")

    tr = maybe_translate(request.text, request.translate) if request.text.strip() \
        else {"text": "", "backend": "skip", "error": None}
    text = tr["text"]

    if request.video_id and request.frame_id:
        # KHÁC hệ thống cũ: point id ở đây hash theo PATH ảnh gốc lúc encode (blake2b, xem
        # mm_bench.indexing.qdrant_store.point_id_for_path), không phải uuid5(video/frame_id)
        # nên không tra ngược được vector đã lưu chỉ từ (video_id, frame_id) mà không biết
        # đúng path string gốc -- xem MODEL_INTEGRATION.md mục "TODO". Chưa implement, báo rõ
        # thay vì âm thầm trả sai.
        raise HTTPException(status_code=501,
                            detail="Lấy vector đã lưu theo video_id/frame_id: CHƯA IMPLEMENT "
                                   "(point id hash theo path ảnh lúc encode, xem "
                                   "MODEL_INTEGRATION.md). Dùng image=<base64/URL> thay thế.")

    tw, iw = float(request.text_weight), float(request.image_weight)
    if not text:
        tw, iw = 0.0, 1.0
    img_vec = enc.encode_image(pil)
    if text:
        txt_vec = enc.encode_text(text)
        import numpy as np
        v = tw * txt_vec + iw * img_vec
        vec = v / (np.linalg.norm(v) or 1.0)
    else:
        vec = img_vec

    hits = query_qdrant(MODELS[request.search_mode]["collection"], vec, request.top_k)
    frames = [make_frame(v, f, s) for v, f, s in hits]
    body = envelope(frames, query=request.text, search_mode=request.search_mode)
    body["weights"] = {"text": tw, "image": iw}
    if tr["backend"] not in ("skip", "none"):
        body["query_en"] = text
    return body


@app.post("/translate")
def translate(request: TranslateRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    t0 = time.perf_counter()
    vi = looks_vietnamese(request.text)
    out = translate_vi_en(request.text) if vi else {"text": request.text, "backend": "skip", "error": None}
    return {"status": "success", "source": request.text, "translated": out["text"],
            "backend": out["backend"], "vietnamese": vi, "error": out["error"],
            "seconds": round(time.perf_counter() - t0, 3)}


@app.post("/ocr_asr_search")
def ocr_asr_search(request: TextSearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    if request.search_mode == "OCR":
        hits = ocr_asr.ocr_search(request.query, request.search_technique, request.top_k)
        frames = [make_frame(h["video_id"], h["frame_id"], h["score"], {"ocr": h["ocr_text"]})
                 for h in hits]
        return envelope(frames, query=request.query, search_mode="OCR",
                        search_technique=request.search_technique)

    frames_raw, segments = ocr_asr.asr_search(request.query, request.search_technique, request.top_k)
    frames = [make_frame(h["video_id"], h["frame_id"], h["score"],
                         {"asr_text": h["asr_text"], "asr_t_start": h["asr_t_start"],
                          "asr_t_end": h["asr_t_end"]})
             for h in frames_raw]
    body = envelope(frames, query=request.query, search_mode="ASR",
                    search_technique=request.search_technique)
    body["total_segments"] = len(segments)
    body["segments"] = [{"rank": i, "video_id": s["video_id"], "text": s["text"],
                         "t_start": s["t_start"], "t_end": s["t_end"]}
                        for i, s in enumerate(segments, start=1)]
    return body
