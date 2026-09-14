# UTOTS_SYSTEM

Hệ thống tìm kiếm video/keyframe cho AI Challenge HCMC 2026 — bản dùng **4 model fine-tune
mới** thay cho PE-Core/BEiT3 gốc:

| Model | Kiểu fine-tune |
|---|---|
| BGE-VL-large | full fine-tune (ft4b) |
| DFN5B ViT-H-14-378 | LoRA (ft2) |
| SigLIP2 giant-opt-384 | LoRA (ft2) |
| PE-Core-bigG-14-448 | LoRA (ft2) |

Cộng OCR (Chandra OCR 2, chạy xong 873/873 video) và ASR (pipeline `audio-aic-2025`, chạy
xong 873/873 video).

## Repo này là gì

Một bản **code portable**, không kèm model weight / embedding (quá nặng, xem
[SETUP.md](SETUP.md) để biết cách trỏ tới nơi để các thứ đó trên từng máy). Chỉ kèm data
NHẸ: OCR/ASR/caption/shot-index đã tính sẵn (`data/`, ~220MB) và framework train/load model
(`third_party/mm_finetune_bench/`, code thuần, ~400KB).

```
UTOTS_SYSTEM/
  README.md                    -- file này
  SETUP.md                     -- cách trỏ tới ảnh/weight/Qdrant/Typesense trên máy mới
  MODEL_INTEGRATION.md         -- chi tiết build lại đúng 4 model (config/override khớp checkpoint)
  pyproject.toml
  app/                         -- FastAPI backend
    main.py                    -- endpoint: /main_search /image_search /ocr_asr_search /video_meta ...
    config.py                  -- MỌI đường dẫn/URL cấu hình được qua biến môi trường
    corpus.py                  -- fps/shot/caption/OCR/ASR helper
    fusion.py                  -- RRF + zscore fusion (trộn điểm nhiều model)
    ocr_asr.py                 -- tìm Typesense
    models/
      registry.py               -- 4 model: collection Qdrant, checkpoint, config, override
      mm_encoder.py              -- bọc mm_bench thành giao diện encode_text/encode_image chung
      base.py                    -- giao diện TextEncoder
  scripts/
    ingest_qdrant.py            -- nạp embedding .npz (bên ngoài) vào Qdrant
    ingest_ocr_typesense.py     -- nạp data/ocr_output vào Typesense
    ingest_asr_typesense.py     -- nạp data/audio_output vào Typesense
  third_party/mm_finetune_bench/ -- framework train/load 4 model (vendor nguyên, xem README riêng)
  data/
    ocr_output/                 -- Chandra OCR 2, 873/873 video
    audio_output/                -- ASR (audio-aic-2025), 873/873 video, có sẵn "keyframes" per câu
    captions/all_keyframes.jsonl -- caption tiếng Anh, 329.931 keyframe
    omnishotcut_shots/           -- shot index (để gộp keyframe theo cú máy trên UI)
    video_meta.json              -- fps từng video
    youtube_map.json             -- video_id -> URL YouTube
  ui.html                       -- frontend, một file không build
```

## Chạy nhanh (đã setup xong theo SETUP.md)

```bash
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8091
# http://localhost:8091/ui
```

## Liên quan tới hệ thống cũ (PE-Core/BEiT3, `APITesting/` trong repo gốc)

Kiến trúc API (envelope, RRF/zscore fusion, temporal search phía UI, dịch VI->EN) giữ
NGUYÊN so với hệ thống cũ — xem `AIC_HCMC_2026_TONG_KET.md` ở repo gốc để đọc giải thích đầy
đủ các quyết định thiết kế đó, tài liệu này không lặp lại. Khác biệt chính nằm ở §"1. Việc
cần làm" phía trên: 4 model mới thay 2 model cũ, nguồn OCR/ASR mới đi kèm sẵn trong repo
thay vì đọc từ annotation ngoài.

Hai hệ thống dùng Qdrant/Typesense **RIÊNG** (port khác nhau, xem `SETUP.md` §3-4) — chạy
song song trên cùng máy không đụng nhau.
