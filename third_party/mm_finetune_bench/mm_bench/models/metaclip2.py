"""MetaCLIP 2 (worldwide, multilingual) adapter.

Loaded via ``open_clip_torch`` as ``hf-hub:facebook/MetaCLIP-2-worldwide-huge``
(see ``configs/models/metaclip2.yaml``). Shares its implementation with PE
Core BigG in ``open_clip_common.OpenCLIPDualEncoderWrapper``, since both are
standard open_clip CLIP-style dual encoders.
"""

from __future__ import annotations

from .open_clip_common import OpenCLIPDualEncoderWrapper


class MetaClip2Wrapper(OpenCLIPDualEncoderWrapper):
    pass
