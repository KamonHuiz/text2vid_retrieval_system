"""Danh sách 4 model — nguồn sự thật duy nhất cho tên collection Qdrant, thư mục embedding,
và cấu hình cần build lại đúng model (config yaml + override LoRA khớp checkpoint đã train).
`scripts/ingest_qdrant.py`, `app/models/mm_encoder.py` và `app/main.py` đều đọc từ đây.

`base_config` là đường dẫn tới file trong `third_party/mm_finetune_bench/configs/models/` —
xem MODEL_INTEGRATION.md để biết vì sao mỗi override khớp đúng với checkpoint đã nhận.
"""
from __future__ import annotations

from .. import config

_MM_BENCH_CONFIGS = config.MM_BENCH_ROOT / "configs" / "models"

MODELS = {
    "BGE_VL": {
        "label": "BGE-VL-large (full fine-tune ft4b)",
        "embeddings_dir": config.EMBEDDINGS_DIR / "bge_vl_ft4b_full",
        "collection": "utots_bge_vl_ft4b",
        "checkpoint": config.WEIGHTS_DIR / "bge_vl_large_ft4b_best.pt",
        "base_config": _MM_BENCH_CONFIGS / "bge_vl_large_full.yaml",
        "overrides": [],                 # full fine-tune, dùng nguyên yaml (adaptation: full)
        "apply_adaptation": False,        # full FT: kiến trúc không đổi, load thẳng state_dict
        "text_field_real": True,          # embeddings/text/ trong npz là caption embedding thật
    },
    "DFN5B": {
        "label": "DFN5B ViT-H-14-378 (LoRA ft2)",
        "embeddings_dir": config.EMBEDDINGS_DIR / "dfn5b_ft2_full",
        "collection": "utots_dfn5b_ft2",
        "checkpoint": config.WEIGHTS_DIR / "dfn5b_vith14_378_ft2_best.pt",
        "base_config": _MM_BENCH_CONFIGS / "dfn5b_vith14_378.yaml",
        "overrides": ["lora.r=32", "lora.alpha=64", "lora.target_module_depth_fraction=0.75"],
        "apply_adaptation": True,
        "text_field_real": True,
    },
    "SIGLIP2": {
        "label": "SigLIP2 giant-opt-384 (LoRA ft2)",
        "embeddings_dir": config.EMBEDDINGS_DIR / "siglip2_ft2_full",
        "collection": "utots_siglip2_ft2",
        "checkpoint": config.WEIGHTS_DIR / "siglip2_ft2_best.pt",
        "base_config": _MM_BENCH_CONFIGS / "siglip2_vit_gopt16_384.yaml",
        "overrides": ["lora.r=16", "lora.alpha=32", "lora.target_module_depth_fraction=0.75"],
        "apply_adaptation": True,
        "text_field_real": False,  # export dùng --skip-text, field "text" trong npz toàn số 0
    },
    "PE_CORE_FT2": {
        "label": "PE-Core-bigG-14-448 (LoRA ft2, KHÁC bản gốc PE-Core đang host ở hệ thống cũ)",
        "embeddings_dir": config.EMBEDDINGS_DIR / "pe_core_bigg_ft2_full",
        "collection": "utots_pe_core_bigg_ft2",
        "checkpoint": config.WEIGHTS_DIR / "pe_core_bigg_ft2_best.pt",
        "base_config": _MM_BENCH_CONFIGS / "pe_core_bigg.yaml",
        "overrides": ["lora.r=32", "lora.alpha=64"],  # depth_fraction giữ mặc định 0.5 trong yaml
        "apply_adaptation": True,
        "text_field_real": False,
    },
}

# Model text-only (Qwen3-Embedding-4B), không có tháp ảnh — chỉ để so caption<->caption,
# KHÔNG index vào main_search vì không có ảnh để trả kết quả gốc.
QWEN3_CAPTION_EMBEDDINGS_DIR = config.EMBEDDINGS_DIR / "qwen3_embed_4b_captions"

# Tổ hợp trộn nhiều model sẵn có, chọn trực tiếp qua `search_mode` giống một "model ảo" —
# main.py không cần biết đây là tổ hợp hay 1 model thật, chỉ tra COMBOS trước MODELS.
# method: "rrf" (chỉ so được THỨ HẠNG, xem fusion.rrf) hay "zscore" (giữ khoảng cách điểm,
# nhạy hơn với outlier, xem fusion.zscore_fuse) — cả hai đọc trong AIC_HCMC_2026_TONG_KET.md §5.2.
COMBOS = {
    "ALL_RRF": {"label": "Cả 4 model (RRF)", "method": "rrf",
                "models": ["BGE_VL", "DFN5B", "SIGLIP2", "PE_CORE_FT2"]},
    "ALL_ZSCORE": {"label": "Cả 4 model (z-score)", "method": "zscore",
                   "models": ["BGE_VL", "DFN5B", "SIGLIP2", "PE_CORE_FT2"]},
    "ZSCORE_BGE_SIGLIP": {"label": "BGE-VL + SigLIP2 (z-score)", "method": "zscore",
                          "models": ["BGE_VL", "SIGLIP2"]},
    "RRF_DFN5B_PECORE": {"label": "DFN5B + PE-Core-ft2 (RRF)", "method": "rrf",
                         "models": ["DFN5B", "PE_CORE_FT2"]},
}
# "ALL" là alias cũ của ALL_RRF, giữ để không phá client đã gọi search_mode="ALL" trước đó.
COMBOS["ALL"] = COMBOS["ALL_RRF"]
