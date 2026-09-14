"""ViT-gopt-16-SigLIP2-384 adapter, via ``transformers.Siglip2Model``.

SigLIP(2) differs from CLIP-style models in two structural ways this wrapper
has to account for:

1. Both towers end in an attention-pooling ``.head`` submodule (not a plain
   linear projection tacked onto a [CLS]/EOS token), so ``modules_to_save``
   in the config targets ``vision_model.head`` / ``text_model.head``.
2. The contrastive objective is a *pairwise sigmoid* loss with a learnable
   ``(logit_scale, logit_bias)`` pair rather than softmax InfoNCE with just
   ``logit_scale`` -- see ``losses/siglip_loss.py`` and
   ``get_logit_bias()`` below.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from .base import DualEncoderWrapper
from ._dryrun import TinyDualEncoder, dry_run_image_transform, dry_run_tokenizer


def _as_embedding_tensor(out) -> torch.Tensor:
    """Normalize what ``get_{image,text}_features`` returns into a tensor.

    transformers changed this return type across versions: older releases
    hand back a plain ``Tensor``, 5.x can hand back a
    ``BaseModelOutputWithPooling`` whose ``pooler_output`` is the pooled
    embedding. Accept either rather than pinning a transformers version.
    """
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("pooler_output", "image_embeds", "text_embeds", "last_hidden_state"):
        value = getattr(out, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    raise TypeError(f"Cannot extract an embedding tensor from {type(out)}")


class SigLIP2Wrapper(DualEncoderWrapper):
    def _build_real_network(self) -> nn.Module:
        from transformers import AutoModel, AutoProcessor  # lazy import

        cfg = self.config["transformers"]
        attn_impl = "flash_attention_2" if self.config.get("flash_attention") else "sdpa"
        quant_kwargs = {}
        if self.config.get("quantization") == "4bit":
            from mm_bench.peft_utils.lora_setup import build_bnb_4bit_config

            quant_kwargs["quantization_config"] = build_bnb_4bit_config()
        model = AutoModel.from_pretrained(
            cfg["model_id"],
            trust_remote_code=cfg.get("trust_remote_code", False),
            attn_implementation=attn_impl,
            **quant_kwargs,
        )
        if self.config.get("quantization") == "4bit":
            from mm_bench.peft_utils.lora_setup import prepare_for_kbit_training

            model = prepare_for_kbit_training(model)
        self._processor = AutoProcessor.from_pretrained(cfg["model_id"])
        return model

    def _build_dry_run_network(self) -> nn.Module:
        self._processor = None
        self._dry_transform = dry_run_image_transform(self.config.get("image_size", 384))
        self._dry_tokenizer = dry_run_tokenizer(max_length=self.config.get("max_text_length", 64))
        return TinyDualEncoder(
            embed_dim=self.config.get("embed_dim", 128),
            image_size=self.config.get("image_size", 384),
        )

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self._processor is not None:
            feats = _as_embedding_tensor(self.net.get_image_features(pixel_values=pixel_values))
        else:
            feats = self.net.encode_image(pixel_values)
        return nn.functional.normalize(feats, dim=-1)

    def encode_text(self, text_tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self._processor is not None:
            feats = _as_embedding_tensor(
                self.net.get_text_features(
                    input_ids=text_tokens["input_ids"],
                    attention_mask=text_tokens.get("attention_mask"),
                )
            )
        else:
            feats = self.net.encode_text(text_tokens["input_ids"])
        return nn.functional.normalize(feats, dim=-1)

    def get_logit_scale(self) -> torch.Tensor:
        return self.net.logit_scale.exp()

    def get_logit_bias(self) -> Optional[torch.Tensor]:
        return getattr(self.net, "logit_bias", None)

    def get_image_transform(self) -> Callable:
        if self._processor is None:
            return self._dry_transform
        processor = self._processor

        def _transform(image):
            # Drop the processor's leading batch dim: the Dataset yields one
            # sample at a time and the collate_fn re-stacks them, so leaving
            # it on would produce (B, 1, 3, H, W).
            return processor(images=image, return_tensors="pt")["pixel_values"][0]

        return _transform

    def get_text_tokenizer(self) -> Callable[[str], Dict[str, torch.Tensor]]:
        if self._processor is None:
            return self._dry_tokenizer
        processor = self._processor
        max_length = self.config.get("max_text_length", 64)

        def _tok(text: str) -> Dict[str, torch.Tensor]:
            return processor(
                text=[text], padding="max_length", truncation=True,
                max_length=max_length, return_tensors="pt",
            )

        return _tok

    def _tower_block_counts(self) -> Dict[str, int]:
        counts = {}
        vision_model = getattr(self.net, "vision_model", None)
        if vision_model is not None:
            counts["vision_model.encoder.layers"] = len(vision_model.encoder.layers)
        text_model = getattr(self.net, "text_model", None)
        if text_model is not None:
            counts["text_model.encoder.layers"] = len(text_model.encoder.layers)
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
                    targets.append(f"{tower_prefix}.{i}.{base}")
        return targets

    def _unfreeze_last_n_blocks(self, tower: str, n: int) -> None:
        if n <= 0:
            return
        prefix = "vision_model.encoder.layers" if tower == "vision" else "text_model.encoder.layers"
        total = self._tower_block_counts().get(prefix, 0)
        start = max(0, total - n)
        for name, p in self.net.named_parameters():
            if name.startswith(f"{prefix}."):
                block_idx = int(name.split(".")[3])
                if block_idx >= start:
                    p.requires_grad_(True)
