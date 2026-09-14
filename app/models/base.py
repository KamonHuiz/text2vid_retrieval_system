"""Giao diện chung mọi model phải theo, để app/main.py không cần biết model nào load
được, model nào chưa — chỉ gọi qua giao diện này và bắt lỗi rõ ràng khi chưa sẵn sàng.
"""
from __future__ import annotations

from typing import Optional


class ModelUnavailable(RuntimeError):
    """Model chưa load được (thiếu framework, thiếu checkpoint, load lỗi...).

    Luôn kèm lý do CỤ THỂ trong message — endpoint trả thẳng message này ra cho UI hiển
    thị, không được nuốt lỗi rồi âm thầm trả kết quả rỗng/sai.
    """


class TextEncoder:
    """Một model có thể encode CÂU CHỮ (và, nếu có tháp ảnh, encode ẢNH) thành vector.

    Implement `encode_text` là bắt buộc để search bằng câu chữ hoạt động. `encode_image`
    là tuỳ chọn (mặc định raise ModelUnavailable) — model nào không có tháp ảnh sống được
    (ví dụ vì checkpoint không kèm) thì cứ để mặc định, `/image_search` sẽ báo rõ lý do
    thay vì crash.
    """

    name: str = "base"
    dim: int = 0

    def encode_text(self, text: str):
        raise NotImplementedError

    def encode_image(self, image):
        raise ModelUnavailable(f"{self.name}: encode_image chưa được implement")

    @property
    def ready(self) -> bool:
        return True

    @property
    def error(self) -> Optional[str]:
        return None
