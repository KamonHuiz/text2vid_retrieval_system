"""PE Core BigG (Meta Perception Encoder) adapter, via open_clip
``('PE-Core-bigG-14-448', 'meta')``.

Unlike DFN5B/MetaCLIP2 (plain open_clip ``CLIP``), open_clip builds PE-Core
as a ``CustomTextCLIP`` (verified on the loaded checkpoint):

* vision: ``visual.trunk`` is a timm ``Eva`` (50 blocks, width 1536) with
  fused ``attn.qkv`` / ``attn.proj`` / ``mlp.fc1`` / ``mlp.fc2`` -- all plain
  ``nn.Linear`` -- pooled by ``visual.trunk.attn_pool`` (AttentionPoolLatent)
  and projected by ``visual.trunk.head``; ``visual.head`` is an empty
  Sequential.
* text: ``text`` is an open_clip ``TextTransformer`` (24 blocks) whose
  attention is ``nn.MultiheadAttention`` (LoRA must target the ``attn``
  module itself, see dfn5b config) and whose ``text_projection`` is a
  Parameter.

The two towers therefore need different LoRA target names, read from
``lora.vision_target_modules`` / ``lora.text_target_modules``. Loading,
encoding and tokenization are inherited unchanged from the shared open_clip
wrapper -- ``CustomTextCLIP`` exposes the same ``encode_image`` /
``encode_text`` / ``logit_scale`` surface.
"""

from __future__ import annotations

from typing import Dict, List

from .open_clip_common import OpenCLIPDualEncoderWrapper

VISION_BLOCKS = "visual.trunk.blocks"
TEXT_BLOCKS = "text.transformer.resblocks"


class PECoreBigGWrapper(OpenCLIPDualEncoderWrapper):
    def _tower_block_counts(self) -> Dict[str, int]:
        counts = {}
        trunk = getattr(getattr(self.net, "visual", None), "trunk", None)
        if trunk is not None and hasattr(trunk, "blocks"):
            counts[VISION_BLOCKS] = len(trunk.blocks)
        text = getattr(self.net, "text", None)
        if text is not None and hasattr(text, "transformer"):
            counts[TEXT_BLOCKS] = len(text.transformer.resblocks)
        return counts

    def resolve_lora_target_modules(self) -> List[str]:
        lora_cfg = self.config["lora"]
        depth_fraction = lora_cfg.get("target_module_depth_fraction", 0.5)
        per_tower = {
            VISION_BLOCKS: lora_cfg["vision_target_modules"],
            TEXT_BLOCKS: lora_cfg["text_target_modules"],
        }
        targets: List[str] = []
        for prefix, n_blocks in self._tower_block_counts().items():
            for i in range(int(n_blocks * (1 - depth_fraction)), n_blocks):
                targets.extend(f"{prefix}.{i}.{name}" for name in per_tower[prefix])
        return targets

    def _unfreeze_last_n_blocks(self, tower: str, n: int) -> None:
        if n <= 0:
            return
        prefix = VISION_BLOCKS if tower == "vision" else TEXT_BLOCKS
        start = max(0, self._tower_block_counts().get(prefix, 0) - n)
        for name, p in self.net.named_parameters():
            if name.startswith(prefix + ".") and int(name[len(prefix) + 1 :].split(".")[0]) >= start:
                p.requires_grad_(True)
