"""BAAI/BGE-VL-large adapter.

BGE-VL-large is a CLIP ViT-L/14 checkpoint (MegaPairs-trained) whose custom
``modeling_MMRet_CLIP.py`` is a verbatim copy of transformers' CLIP modeling
file plus convenience helpers (``encode_image`` / ``encode_text`` /
``encode_multimodal``). Its ``get_image_features`` / ``get_text_features``
are line-for-line identical to stock CLIP, so we load it as a stock
``transformers.CLIPModel`` (``trust_remote_code=False``) instead of running
that remote code: identical embeddings, and no exposure to remote code
written against an older transformers API.

That makes it exactly the "HF dual encoder with get_*_features + a
processor" shape ``SigLIP2Wrapper`` already implements (its logit-bias hook
returns ``None`` for CLIP), so this class only exists as BGE-VL's named,
documented entry point in the registry.
"""

from __future__ import annotations

from .siglip2 import SigLIP2Wrapper


class BGEVLWrapper(SigLIP2Wrapper):
    pass
