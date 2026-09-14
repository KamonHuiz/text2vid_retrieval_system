# Tích hợp 4 model fine-tune

Framework train/load (`mm_bench`) được vendor nguyên trong `third_party/mm_finetune_bench/`
— không cần cài gì thêm ngoài `pip install -r third_party/mm_finetune_bench/requirements.txt`
(đã gộp vào `pyproject.toml` của repo này rồi, `uv sync` là đủ).

## Cách build lại đúng 1 model

```python
import sys
sys.path.insert(0, "third_party/mm_finetune_bench")

from mm_bench.models.registry import build_model
from mm_bench.training.checkpoint import load_checkpoint
from mm_bench.utils.config import load_config, apply_dotted_overrides

cfg = load_config("third_party/mm_finetune_bench/configs/models/<base_config>.yaml")
cfg = apply_dotted_overrides(cfg, ["lora.r=32", "lora.alpha=64", ...])  # xem bảng dưới
model = build_model(cfg, dry_run=False, apply_adaptation=<True/False, xem bảng>)
load_checkpoint("<checkpoint>.pt", model, map_location="cpu")
model.to(device=..., dtype=...)
model.eval()

text_vec = model.encode_text(model.get_text_tokenizer()("một câu tiếng Anh"))
```

Đã bọc sẵn thành `app/models/mm_encoder.py: MMBenchEncoder` — bình thường không cần đụng
tới đoạn trên, chỉ cần biết ĐÚNG override nào khớp checkpoint nào (bảng dưới, đã điền sẵn
trong `app/models/registry.py`, đừng sửa trừ khi có checkpoint mới).

## Bảng model <-> config <-> override (nguồn: package_4models_export/README.md)

| Model | `base_config` | `overrides` | `apply_adaptation` |
|---|---|---|---|
| BGE_VL | `configs/models/bge_vl_large_full.yaml` | (không có, full fine-tune) | `False` |
| DFN5B | `configs/models/dfn5b_vith14_378.yaml` | `lora.r=32 lora.alpha=64 lora.target_module_depth_fraction=0.75` | `True` |
| SIGLIP2 | `configs/models/siglip2_vit_gopt16_384.yaml` | `lora.r=16 lora.alpha=32 lora.target_module_depth_fraction=0.75` | `True` |
| PE_CORE_FT2 | `configs/models/pe_core_bigg.yaml` | `lora.r=32 lora.alpha=64` (depth_fraction giữ mặc định 0.5 trong yaml) | `True` |

**Vì sao `apply_adaptation` khác nhau:** 3 model LoRA cần build lại đúng cấu trúc
`PeftModel` (đúng r/alpha/target modules) TRƯỚC khi `load_checkpoint` mới khớp được tên
tham số đã lưu — sai bất kỳ số nào ở trên, `load_checkpoint` sẽ tự raise lỗi rõ ràng
(`unexpected_keys`) chứ không load nhầm âm thầm (xem
`third_party/mm_finetune_bench/mm_bench/training/checkpoint.py`). BGE-VL là full
fine-tune, checkpoint là `state_dict` thường của `transformers.CLIPModel`, không cần bọc
PeftModel.

**Vì sao SIGLIP2/PE_CORE_FT2 không tra được vector text thật cho keyframe có sẵn:** gói
export chạy `--skip-text` khi encode 2 model này (xem `text_field_real: False` trong
`registry.py`) — field `text` trong `.npz` là số 0, không dùng được. Không ảnh hưởng gì
tới search bình thường (query luôn encode MỚI qua `encode_text` lúc chạy, không đọc field
`text` trong `.npz`), chỉ ảnh hưởng nếu sau này muốn tính similarity caption<->caption cho
2 model đó mà không encode lại.

## Điểm chưa xong (TODO)

`POST /image_search` với `video_id` + `frame_id` (lấy vector ảnh ĐÃ LƯU của một keyframe
trong index, không cần encode lại — hệ thống cũ dùng cách này để BEiT3 "tìm bằng ảnh có
sẵn" chạy được dù không có tháp ảnh) **chưa implement** ở đây: point id trong Qdrant hash
theo blake2b của chuỗi path ảnh lúc encode (`mm_bench.indexing.qdrant_store.point_id_for_path`),
không phải `uuid5(video_id/frame_id)` như hệ thống cũ, nên cần biết đúng path string gốc
(dạng `"{lot}/{lot}/{video_id}/{frame_id}.webp"`, xem README của
`package_4models_export`) để tính lại đúng id rồi `client.retrieve()`. Muốn bật tính năng
này: viết hàm tái tạo path string đúng công thức, hash bằng `point_id_for_path`, gọi
`STATE["qdrant"].retrieve(collection_name=..., ids=[pid], with_vectors=True)` — xem
`frame_vector()` trong `APITesting/main.py` của hệ thống cũ để tham khảo cấu trúc tương tự.

## Model text-only chưa dùng: Qwen3-Embedding-4B

`data` (`embeddings/qwen3_embed_4b_captions/shard*.npz`) chỉ có field `embedding` (2560d)
của CAPTION, không có ảnh — không index vào `main_search` được (không có ảnh để trả kết
quả). Có thể dùng riêng cho một tính năng "tìm caption tương tự" nếu cần sau này, chưa
implement ở bản này.
