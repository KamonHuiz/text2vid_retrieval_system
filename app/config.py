"""Đường dẫn/URL cấu hình được — mọi thứ nặng (ảnh, weight, embedding) nằm NGOÀI repo này,
trỏ tới bằng biến môi trường. Xem SETUP.md để biết cách đặt từng biến trên một máy mới.
"""
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent  # thư mục UTOTS_SYSTEM/

# --- Dữ liệu nhẹ, đi kèm trong repo (data/) ---
DATA_ROOT = _HERE / "data"
OCR_OUTPUT_DIR = Path(os.environ.get("UTOTS_OCR_DIR", DATA_ROOT / "ocr_output"))
AUDIO_OUTPUT_DIR = Path(os.environ.get("UTOTS_AUDIO_DIR", DATA_ROOT / "audio_output"))
VIDEO_META_FILE = Path(os.environ.get("UTOTS_VIDEO_META", DATA_ROOT / "video_meta.json"))
YOUTUBE_MAP_FILE = Path(os.environ.get("UTOTS_YOUTUBE_MAP", DATA_ROOT / "youtube_map.json"))

# --- Dữ liệu NẶNG, KHÔNG đi kèm trong repo — mỗi máy tự trỏ tới chỗ mình để ---
# Ảnh keyframe WebP gốc của corpus (317,961+ file, ~33GB). Xem SETUP.md §1.
KEYFRAME_ROOT = Path(os.environ.get("UTOTS_KEYFRAME_ROOT", "/mnt/aic/data/keyframes"))
# Video gốc .mp4, layout <LOT>/<VIDEO_ID>.mp4 (873 video, ~78GB) -- dùng để phát trực tiếp
# trong popup thay vì nhúng YouTube (không phụ thuộc mạng ngoài/link YouTube còn sống hay
# không, tua mượt hơn nhờ HTTP Range). Thiếu thư mục này API vẫn chạy, popup chỉ rơi về
# link YouTube trong metadata (nếu có). Xem SETUP.md.
VIDEO_ROOT = Path(os.environ.get("UTOTS_VIDEO_ROOT", "/mnt/aic/data/videos"))
# Thư mục đã giải nén package_4models_export.zip — chứa weights/ và embeddings/.
# Xem SETUP.md §2.
MODEL_PACKAGE_ROOT = Path(os.environ.get(
    "UTOTS_MODEL_PACKAGE", "/mnt/aic/AIC/AIC2026_DATA/package_4models_export"))
WEIGHTS_DIR = MODEL_PACKAGE_ROOT / "weights"
EMBEDDINGS_DIR = MODEL_PACKAGE_ROOT / "embeddings"
CAPTIONS_FILE = MODEL_PACKAGE_ROOT / "captions" / "all_keyframes.jsonl"

# mm_finetune_bench — framework train/load 4 model, vendor nguyên trong repo (code, không
# phải data/model) tại third_party/. Không cần biến môi trường vì luôn đi kèm.
MM_BENCH_ROOT = _HERE / "third_party" / "mm_finetune_bench"

# --- Database ngoài (mỗi máy tự dựng, xem SETUP.md §3-4) ---
QDRANT_URL = os.environ.get("UTOTS_QDRANT_URL", "http://localhost:6335")
TYPESENSE_URL = os.environ.get("UTOTS_TYPESENSE_URL", "http://localhost:8109")
TYPESENSE_KEY = os.environ.get("UTOTS_TYPESENSE_KEY", "utots-local-key")

# --- Thiết bị chạy model (encode query lúc search) ---
DEVICE = os.environ.get("UTOTS_DEVICE", "cpu")  # vd "cuda:1" để tránh đụng GPU đang host API khác
# Nhiều GPU: đặt UTOTS_DEVICES="cuda:0,cuda:1" để chia 4 model round-robin qua các GPU thay
# vì dồn hết vào một cái (UTOTS_DEVICE). Không đặt thì dùng nguyên UTOTS_DEVICE cho cả 4.
DEVICES = [d.strip() for d in os.environ.get("UTOTS_DEVICES", "").split(",") if d.strip()] or [DEVICE]

# RRF constant — giống hệ thống cũ, xem AIC_HCMC_2026_TONG_KET.md §5.2
RRF_K = 60
