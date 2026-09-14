"""Shared implementation for any CLIP-style dual encoder loaded through
``open_clip_torch`` (PE Core BigG and MetaCLIP 2 both fit this shape
exactly: ``model.visual.transformer.resblocks[i]`` /
``model.transformer.resblocks[i]``, a single fused ``logit_scale``, and an
``open_clip.create_model_and_transforms`` / ``get_tokenizer`` loading path).

Model-specific files (``pe_core_bigg.py``, ``metaclip2.py``) subclass this
and only exist so each architecture has its own importable, documented
entry point in ``models.registry`` -- keep the actual logic here so a fix
or feature (e.g. supporting a 5th open_clip model) lands once.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import torch
import torch.nn as nn

from .base import DualEncoderWrapper
from ._dryrun import TinyDualEncoder, dry_run_image_transform, dry_run_tokenizer


class OpenCLIPDualEncoderWrapper(DualEncoderWrapper):
    def _build_real_network(self) -> nn.Module:
        import open_clip  # lazy import

        cfg = self.config["open_clip"]
        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg["model_name"], pretrained=cfg["pretrained"]
        )
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(cfg["model_name"])
        return model

    def _build_dry_run_network(self) -> nn.Module:
        self._preprocess = dry_run_image_transform(self.config.get("image_size", 224))
        self._tokenizer = dry_run_tokenizer(max_length=self.config.get("context_length", 77))
        return TinyDualEncoder(
            embed_dim=self.config.get("embed_dim", 128),
            image_size=self.config.get("image_size", 224),
        )

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.net.encode_image(pixel_values)
        return nn.functional.normalize(feats, dim=-1)

    def encode_text(self, text_tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        ids = text_tokens["input_ids"] if isinstance(text_tokens, dict) else text_tokens
        feats = self.net.encode_text(ids)
        return nn.functional.normalize(feats, dim=-1)

    def get_logit_scale(self) -> torch.Tensor:
        return self.net.logit_scale.exp()

    def get_image_transform(self) -> Callable:
        return self._preprocess

    def get_text_tokenizer(self) -> Callable[[str], Dict[str, torch.Tensor]]:
        tokenizer = self._tokenizer

        def _tok(text: str) -> Dict[str, torch.Tensor]:
            return {"input_ids": tokenizer([text])}

        return _tok

    def _tower_block_counts(self) -> Dict[str, int]:
        counts = {}
        visual = getattr(self.net, "visual", None)
        visual_transformer = getattr(visual, "transformer", None)
        if visual_transformer is not None:
            counts["visual.transformer"] = len(visual_transformer.resblocks)
        text_transformer = getattr(self.net, "transformer", None)
        if text_transformer is not None:
            counts["transformer"] = len(text_transformer.resblocks)
        return counts

    def resolve_lora_target_modules(self) -> List[str]:
        lora_cfg = self.config["lora"]
        depth_fraction = lora_cfg.get("target_module_depth_fraction", 0.5)
        base_names = lora_cfg["target_modules"]

        targets: List[str] = []
        for tower_prefix, n_blocks in self._tower_block_counts().items():
            start = int(n_blocks * (1 - depth_fraction))
            for i in range(start, n_blocks):
                for base in base_names:
                    targets.append(f"{tower_prefix}.resblocks.{i}.{base}")
        return targets

    def _unfreeze_last_n_blocks(self, tower: str, n: int) -> None:
        if n <= 0:
            return
        prefix = "visual.transformer" if tower == "vision" else "transformer"
        total = self._tower_block_counts().get(prefix, 0)
        start = max(0, total - n)
        for name, p in self.net.named_parameters():
            if name.startswith(f"{prefix}.resblocks."):
                block_idx = int(name.split(".")[2])
                if block_idx >= start:
                    p.requires_grad_(True)
