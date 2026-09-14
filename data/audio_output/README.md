# Audio (ASR) output — AIC HCMC 2026

Nguồn: pipeline `audio-aic-2025` (Whisper ASR → flagger → LLM refiner, xem
`AGENTS.md` của project đó nếu cần chi tiết pipeline). **Đã chạy xong toàn bộ
873/873 video** của corpus.

## Cấu trúc

```
<LOT>/<VIDEO_ID>.json             # transcript đầy đủ, 4 giai đoạn xử lý
<LOT>/<VIDEO_ID>.sentences.json   # transcript đã tách câu + gắn keyframe
<LOT>/<VIDEO_ID>.srt              # phụ đề chuẩn SRT (giống nội dung .sentences.json)
```
`LOT` = L21..L30, `VIDEO_ID` = `<LOT>_V###` (vd `L21_V001`). Mỗi video có
đúng 3 file trên — đủ 873×3 = 2619 file trong gói này.

## 1. `<VIDEO_ID>.json` — transcript 4 giai đoạn

```json
{
  "unfiltered": {"text": "...", "word_count": N},
  "flagged":    {"text": "...", "word_count": N},
  "refined":    {"text": "...", "word_count": N, "source": "..."},
  "proofread":  {"text": "...", "word_count": N}
}
```
- `unfiltered` — text thô từ Whisper, **không dấu câu, không viết hoa**, có thể có lỗi ASR.
- `flagged` — như trên nhưng các span nghi vấn (số, ngày giờ, đơn vị, tên riêng...)
  được bọc `@flag_type(nội_dung)` để LLM refiner xử lý tiếp — không dùng trực
  tiếp, chỉ là bước trung gian.
- `refined` — bản LLM đã sửa dấu câu/viết hoa/chính tả lần 1.
- `proofread` — **bản cuối cùng, khuyến nghị dùng bản này** cho mọi việc đọc
  hiểu / tìm kiếm văn bản. Đây cũng là text đã dùng để sinh `.sentences.json`
  và `.srt`.

## 2. `<VIDEO_ID>.sentences.json` — câu + timestamp + keyframe liên quan

```json
{
  "video_id": "L21_V001",
  "fps": 30.0,
  "n_sentences": 137,
  "total_words": 4391,
  "sentences": [
    {
      "text": "Chào mừng quý vị...",
      "start": 1.64, "end": 8.2,          // giây, theo timeline video gốc
      "n_words": 19,
      "start_frame": 49, "end_frame": 246, // = round(start*fps), round(end*fps)
      "keyframes": ["000067.webp", "000238.webp"],
      "keyframe_count": 2
    },
    ...
  ]
}
```
- Mỗi câu được gán các **keyframe rơi trong khoảng [start_frame, end_frame]**
  của câu đó. Tên file trong `keyframes` khớp *chính xác* basename các file
  `.webp` trong thư mục `keyframes/<LOT>/<VIDEO_ID>/` của corpus gốc (nếu bạn
  có quyền truy cập `/data/AIC/keyframes/`) — dùng để join text ↔ ảnh.
- Muốn đổi 1 frame_id (số trong tên file, vd `000067`) sang giây: `int(frame_id) / fps`.
  `fps` lấy ở field `fps` trong chính file này (không cần file nào khác).
- Đây là **nguồn giàu thông tin nhất** để index tìm kiếm theo câu + ảnh liên quan.

## 3. `<VIDEO_ID>.srt` — phụ đề chuẩn

Sinh trực tiếp từ `sentences.json` (cùng text/timestamp), định dạng SRT chuẩn
— dùng để hiển thị phụ đề hoặc tiện xem nhanh nội dung video, không có thêm
thông tin gì so với `.sentences.json`.

## Việc KHÔNG có trong gói này

Các file `.progress_shard*.json` / `.progress_refine.json` (state nội bộ để
resume pipeline khi bị ngắt giữa chừng) đã bị loại — không có giá trị sử
dụng, chỉ gây nhiễu.

## Gợi ý dùng nhanh

- Chỉ cần text sạch để index full-text search → đọc `proofread.text` trong
  `<VIDEO_ID>.json`.
- Cần tìm theo câu + trỏ tới ảnh cụ thể (frame-level retrieval) → dùng
  `sentences.json`, mỗi sentence tự mang theo `keyframes` liên quan.
- Cần hiển thị phụ đề đồng bộ video → dùng `.srt`.
