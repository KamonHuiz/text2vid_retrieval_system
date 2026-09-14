# OCR output — AIC HCMC 2026

Nguồn: OCR toàn bộ keyframe corpus bằng model **Chandra OCR 2**
(`datalab-to/chandra-ocr-2`) — F1 có dấu 78,3% trên bake-off nội bộ (so với
55,9% pipeline EasyOCR+VietOCR+LLM cũ, 64,9% Qwen3-VL 8B). **Đã chạy xong
toàn bộ 873/873 video, 329.938 ảnh keyframe** của corpus.

## Cấu trúc

```
<LOT>/<VIDEO_ID>.json   # 1 file / video, dict phẳng {frame_id: text}
```
`LOT` = L21..L30, `VIDEO_ID` = `<LOT>_V###`. Đủ 873 file trong gói này —
mỗi video có đúng 1 file.

## Format 1 file `<VIDEO_ID>.json`

```json
{
  "000000": "HTV9 HD 06:30:11\ngiây",
  "000067": "HTV9 HD 06:30:13",
  "000458": "TIN CHÍNH\nHTV9 HD 06:30:26\n16 giờ\nTÌNH TRẠNG SỰ LÚN Ở ĐBSCL..."
}
```
- **Key** — `frame_id`, 6 chữ số, khớp *chính xác* basename file `.webp`
  trong `keyframes/<LOT>/<VIDEO_ID>/` của corpus gốc (vd frame_id `000067`
  ↔ file `000067.webp`). Đây cũng là **số thứ tự frame tuyệt đối trong
  video gốc** (không phải số thứ tự keyframe) → đổi sang giây bằng
  `int(frame_id) / fps` (fps lấy trong `media-info/<VIDEO_ID>.json` hoặc
  cột `fps` trong `map-keyframes/<VIDEO_ID>.csv`, không kèm trong gói này).
- **Value** — text OCR được, mỗi block chữ Chandra nhận diện tách ra nằm
  trên **một dòng riêng** (nối bằng `\n`). Đã bỏ các block nhãn
  `Image`/`Figure`/`Blank-Page` (mô tả ảnh do model tự sinh, không phải
  chữ trên màn hình). Giữ `Page-Header`/`Page-Footer` (logo kênh, đồng hồ
  góc màn hình hay bị gán nhãn này).
- **Frame không có chữ → không xuất hiện trong dict** (không phải
  `""`) — đừng coi thiếu key là lỗi.

## Lưu ý quan trọng

- Đây là **OCR mới (Chandra 2)**, chưa phải nguồn OCR đang phục vụ search UI
  hiện tại của corpus gốc (UI dùng OCR từ nguồn annotation khác). Nếu bạn
  ingest gói này vào hệ thống search, coi đây là nguồn OCR chất lượng cao
  hơn để thay thế/bổ sung, không phải để merge field-by-field với OCR cũ.
- Schema đề xuất khi ingest (vd Typesense/Elasticsearch...):
  `video_id, lot, frame_id, frame_index, timestamp_sec, ocr_text` — trong đó
  `frame_index`/`timestamp_sec` bạn tự tính từ `frame_id` + fps như trên.
- Không có ảnh kèm theo gói này (chỉ text OCR) — cần ảnh gốc thì lấy từ
  `keyframes/<LOT>/<VIDEO_ID>/<frame_id>.webp` của corpus.

## Gợi ý dùng nhanh

- Full-text search theo video → nối toàn bộ value trong 1 file `.json`.
- Tìm theo frame cụ thể / hiển thị chữ chồng lên đúng ảnh → tra trực tiếp
  bằng key `frame_id`.
