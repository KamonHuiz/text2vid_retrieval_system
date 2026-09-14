"""Device/dtype movement for heterogeneous batches.

Shared by the evaluator and the encoder so both handle the same subtlety:
a batch can be a bare tensor (CLIP-style pixel_values) or a nested dict of
tensors (a unified decoder's processor output, mixing float pixel values
with integer token ids), and only the floating-point ones may be cast.
"""

from __future__ import annotations

from typing import Any

import torch


def move_batch(obj: Any, device: torch.device, dtype: torch.dtype | None = None) -> Any:
    """Recursively move tensors to ``device``, casting only float tensors.

    Integer tensors (``input_ids``, ``attention_mask``, ``image_grid_thw``)
    keep their dtype -- casting token ids to fp16 silently corrupts them.
    """
    if isinstance(obj, dict):
        return {k: move_batch(v, device, dtype) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], torch.Tensor):
        return type(obj)(move_batch(v, device, dtype) for v in obj)
    if isinstance(obj, torch.Tensor):
        if dtype is not None and obj.is_floating_point():
            return obj.to(device=device, dtype=dtype, non_blocking=True)
        return obj.to(device=device, non_blocking=True)
    return obj
