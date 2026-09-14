# Setup trên một máy mới

Repo này chỉ có CODE + dữ liệu nhẹ (OCR/ASR/caption/shot-index, ~220MB, xem `data/`). Ba
thứ NẶNG dưới đây mỗi máy tự chuẩn bị lấy, KHÔNG đi kèm repo.

## 0. Cài Python deps

```bash
cd UTOTS_SYSTEM
uv sync            # hoặc: pip install -e .
```

## 1. Ảnh keyframe gốc (~33GB, 317.961+ file WebP)

Copy/mount thư mục `keyframes/<LOT>/<VIDEO_ID>/<frame_id>.webp` của corpus vào máy, rồi trỏ:

```bash
export UTOTS_KEYFRAME_ROOT=/đường/dẫn/tới/keyframes
```

Không có bước này API vẫn khởi động được nhưng mọi kết quả search sẽ không có ảnh/không
tính được `timestamp_sec` đúng (frame index vẫn liệt kê được từ tên file).

## 2. Model weights + embedding đã tính sẵn (~21GB, `package_4models_export.zip`)

Giải nén gói (xin từ người giữ gói, không đi kèm repo vì quá nặng):

```bash
unzip package_4models_export.zip -d /đường/dẫn/tới
export UTOTS_MODEL_PACKAGE=/đường/dẫn/tới/package_4models_export
```

Phải thấy đúng cấu trúc:

```
package_4models_export/
  weights/{bge_vl_large_ft4b_best.pt, dfn5b_vith14_378_ft2_best.pt,
           siglip2_ft2_best.pt, pe_core_bigg_ft2_best.pt}
  embeddings/{bge_vl_ft4b_full, dfn5b_ft2_full, siglip2_ft2_full,
              pe_core_bigg_ft2_full, qwen3_embed_4b_captions}/shard{0,1}.npz
  captions/all_keyframes.jsonl   (đã copy sẵn vào data/captions/ trong repo -- có thể bỏ qua)
```

## 3. Qdrant (vector search 4 model)

```bash
docker run -d --name utots_qdrant -p 6335:6333 -p 6336:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant:latest
export UTOTS_QDRANT_URL=http://localhost:6335
```

> Cổng 6335/6336 (khác 6333/6334 của hệ thống cũ) để không đụng vào Qdrant đang phục vụ
> API khác trên cùng máy — đổi nếu máy bạn không có xung đột đó.

Nạp embedding (đọc trực tiếp từ `.npz`, không cần encode lại ảnh):

```bash
python scripts/ingest_qdrant.py --model ALL
```

Từng model ~330k điểm; theo dõi tiến độ qua log in ra, ~vài phút/model tuỳ tốc độ đĩa.

## 4. Typesense (OCR + ASR)

```bash
docker run -d --name utots_typesense -p 8109:8108 \
  -v $(pwd)/typesense_data:/data typesense/typesense:27.1 \
  --data-dir /data --api-key=utots-local-key --enable-cors
export UTOTS_TYPESENSE_URL=http://localhost:8109
export UTOTS_TYPESENSE_KEY=utots-local-key
```

Nạp (dữ liệu OCR/ASR đã có sẵn trong `data/`, không cần gì thêm):

```bash
python scripts/ingest_ocr_typesense.py
python scripts/ingest_asr_typesense.py
```

## 5. Chọn thiết bị chạy model (GPU/CPU)

Một GPU:

```bash
export UTOTS_DEVICE=cuda:1    # hoặc cuda:0, cpu -- tuỳ máy có mấy GPU
```

Nhiều GPU — chia 4 model round-robin qua từng GPU thay vì dồn hết vào một cái (đã test
thật: BGE_VL+SIGLIP2 -> cuda:0, DFN5B+PE_CORE_FT2 -> cuda:1):

```bash
export UTOTS_DEVICES=cuda:0,cuda:1
```

> Nếu máy đang chạy sẵn một hệ thống search khác chiếm GPU nào đó, dùng GPU còn lại hoặc
> CPU cho UTOTS_SYSTEM để không tranh VRAM/compute với hệ thống đang host.

Load cả 4 model (3 model LoRA build lại qua `mm_bench` + load checkpoint, BGE-VL full
fine-tune) tốn vài phút lúc khởi động API — bình thường, xem log `[models] đang load ...`.
Lần đầu chạy trên máy mới sẽ chậm hơn vì 2 model `transformers` (BGE_VL, SIGLIP2) phải tải
checkpoint gốc từ HuggingFace Hub (~vài phút, một lần, sau đó nằm trong cache HF).

## 6. Chạy API + UI

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8091
# mở http://<ip-máy>:8091/ui
```

Kiểm tra `curl http://localhost:8091/health` phải thấy đủ 4 model `"ready"` và 6 collection
(`utots_bge_vl_ft4b`, `utots_dfn5b_ft2`, `utots_siglip2_ft2`, `utots_pe_core_bigg_ft2`,
`utots_ocr_frames`, `utots_asr_sentences`) đều có ~330k / ~298k / ~46k document — thiếu cái
nào nghĩa là bước ingest tương ứng ở §3/§4 chưa chạy hoặc chạy lỗi.

Kiểm tra nhanh mọi thứ đã lên chưa: `curl http://localhost:8091/health` — model nào lỗi sẽ
thấy message lý do cụ thể trong `models`, KHÔNG bị giấu.

## Biến môi trường đầy đủ (tham khảo `app/config.py`)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `UTOTS_KEYFRAME_ROOT` | `/mnt/aic/data/keyframes` | ảnh gốc, §1 |
| `UTOTS_MODEL_PACKAGE` | `/mnt/aic/AIC/AIC2026_DATA/package_4models_export` | weights+embeddings, §2 |
| `UTOTS_QDRANT_URL` | `http://localhost:6335` | §3 |
| `UTOTS_TYPESENSE_URL` | `http://localhost:8109` | §4 |
| `UTOTS_TYPESENSE_KEY` | `utots-local-key` | §4 |
| `UTOTS_DEVICE` | `cpu` | §5 |
| `UTOTS_OCR_DIR` / `UTOTS_AUDIO_DIR` | `data/ocr_output` / `data/audio_output` | đổi nếu di chuyển `data/` ra ngoài repo |
| `UTOTS_VIDEO_META` / `UTOTS_YOUTUBE_MAP` | `data/video_meta.json` / `data/youtube_map.json` | fps map / link YouTube |
