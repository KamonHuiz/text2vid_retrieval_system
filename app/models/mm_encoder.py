"""Load 1 trong 4 model đã fine-tune qua framework `mm_bench` (vendor tại
third_party/mm_finetune_bench/) và bọc lại theo giao diện `TextEncoder` chung.

Cách build đúng NGUYÊN VĂN theo README của gói xuất (xem package_4models_export/README.md,
copy lại vào MODEL_INTEGRATION.md):

    model = build_model(load_config(base_config, overrides), apply_adaptation=True)
    load_checkpoint(checkpoint_path, model)

`apply_adaptation` PHẢI đúng (True cho 3 model LoRA để dựng lại đúng cấu trúc PeftModel
khớp tên tham số đã lưu; False cho BGE-VL full fine-tune vì checkpoint là state_dict thường,
không phải state_dict của PeftModel) — sai chỗ này thì `load_checkpoint` tự raise lỗi rõ
ràng (unexpected_keys) chứ không load nhầm âm thầm, xem checkpoint.py trong mm_bench.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from .. import config
from .base import ModelUnavailable, TextEncoder

if str(config.MM_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(config.MM_BENCH_ROOT))


class MMBenchEncoder(TextEncoder):
    def __init__(self, name: str, spec: dict, device: str = None):
        self.name = name
        self.spec = spec
        self.device_str = device or config.DEVICE
        self._net = None
        self._error: Optional[str] = None
        self.dim = 0
        self._load()

    def _load(self):
        checkpoint = Path(self.spec["checkpoint"])
        base_config = Path(self.spec["base_config"])
        if not checkpoint.exists():
            self._error = (f"{self.name}: không tìm thấy checkpoint tại {checkpoint} — "
                           f"kiểm tra UTOTS_MODEL_PACKAGE đã trỏ đúng chỗ giải nén "
                           f"package_4models_export chưa (xem SETUP.md §2)")
            return
        if not base_config.exists():
            self._error = f"{self.name}: không tìm thấy config {base_config} (third_party/mm_finetune_bench thiếu file?)"
            return
        try:
            import torch

            from mm_bench.models.registry import build_model
            from mm_bench.training.checkpoint import load_checkpoint
            from mm_bench.utils.config import apply_dotted_overrides, load_config

            cfg = load_config(str(base_config))
            cfg.setdefault("flash_attention", False)
            if self.spec.get("overrides"):
                cfg = apply_dotted_overrides(cfg, self.spec["overrides"])

            model = build_model(cfg, dry_run=False, apply_adaptation=self.spec["apply_adaptation"])
            load_checkpoint(str(checkpoint), model, map_location="cpu", restore_rng=False)
            device = torch.device(self.device_str)
            dtype = torch.float16 if device.type == "cuda" else torch.float32
            model.to(device=device, dtype=dtype)
            model.eval()

            self._net = model
            self._device = device
            self._dtype = dtype
            self._tokenizer = model.get_text_tokenizer()
            self.dim = int(cfg.get("embed_dim") or 0)
        except Exception as exc:  # noqa: BLE001 — muốn bắt MỌI lỗi load để báo rõ, không crash cả API
            self._error = f"{self.name}: lỗi load model — {type(exc).__name__}: {exc}"

    @property
    def ready(self) -> bool:
        return self._net is not None

    @property
    def error(self) -> Optional[str]:
        return self._error

    def encode_text(self, text: str):
        if self._net is None:
            raise ModelUnavailable(self._error)
        import torch

        tokens = self._tokenizer(text)
        tokens = {k: (v.to(self._device) if hasattr(v, "to") else v) for k, v in tokens.items()}
        with torch.no_grad():
            vec = self._net.encode_text(tokens)
        return vec.float().cpu().numpy()[0]

    def encode_image(self, image):
        if self._net is None:
            raise ModelUnavailable(self._error)
        import torch

        transform = self._net.get_image_transform()
        pixel = transform(image.convert("RGB")).unsqueeze(0).to(self._device, dtype=self._dtype)
        with torch.no_grad():
            vec = self._net.encode_image(pixel)
        return vec.float().cpu().numpy()[0]


def load_all(devices: list[str] = None) -> dict[str, MMBenchEncoder]:
    """Load cả 4 model, chia round-robin qua `devices` (vd ["cuda:0", "cuda:1"]) nếu có
    nhiều hơn 1 -- mặc định [config.DEVICE]. Model nào lỗi vẫn nằm trong dict
    (`.ready == False`, `.error` có lý do) — main.py không được loại nó ra, phải để endpoint
    báo lỗi rõ ràng cho từng model."""
    from .registry import MODELS

    devices = devices or config.DEVICES
    out = {}
    for i, (name, spec) in enumerate(MODELS.items()):
        device = devices[i % len(devices)]
        print(f"[models] đang load {name} ({spec['label']}) trên {device} ...")
        enc = MMBenchEncoder(name, spec, device=device)
        if enc.ready:
            print(f"[models] {name} sẵn sàng trên {device}, dim={enc.dim}")
        else:
            print(f"[models] {name} KHÔNG sẵn sàng: {enc.error}")
        out[name] = enc
    return out
