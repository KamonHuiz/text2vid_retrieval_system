"""``jinaai/jina-embeddings-v5-omni-small-retrieval`` adapter.

.. important::
    **This model is not a dual encoder.** It is a unified Qwen3-VL-Audio
    decoder (``Qwen3VLAudioModel``, custom code, ``trust_remote_code=True``)
    that embeds *every* modality through one forward pass and pools the
    **last token** of the decoder's final hidden state. Verified against the
    actual checkpoint (not the model card alone):

    * image: ``model.embed(**proc(images=img,
      text="<|vision_start|><|image_pad|><|vision_end|>", return_tensors="pt"))``
    * text:  ``model.embed(**proc(text="Query: ...", return_tensors="pt"))``

    Two consequences the other three wrappers don't have:

    1. Preprocessing is **batch-level**, not per-sample: the number of image
       placeholder tokens depends on each image's grid, so the processor has
       to pad a whole batch together. Hence ``get_image_transform()`` returns
       ``None`` (the Dataset yields raw PIL) and ``get_collate_fn()`` runs
       the processor over the batch.
    2. The checkpoint's own ``embed()`` wraps the forward in
       ``torch.no_grad()``, which would silently break fine-tuning. We
       therefore re-implement its pooling in :meth:`_embed` (identical
       math: last non-padded token, optional Matryoshka truncation, L2
       normalize) so gradients flow during training while inference
       results stay bit-identical to ``embed()``.

    Asymmetric retrieval: the model was trained with ``Query:`` /
    ``Document:`` prefixes on the text side. Captions are indexed as
    documents; a user's search string should be encoded as a query. See
    ``text_prefix`` / ``query_prefix`` in the config.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from .base import DualEncoderWrapper
from ._dryrun import TinyDualEncoder, dry_run_image_transform, dry_run_tokenizer

IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


class JinaOmniWrapper(DualEncoderWrapper):
    def _build_real_network(self) -> nn.Module:
        from transformers import AutoModel, AutoProcessor  # lazy import

        cfg = self.config["transformers"]
        quant_kwargs = {}
        if self.config.get("quantization") == "4bit":
            from mm_bench.peft_utils.lora_setup import build_bnb_4bit_config

            quant_kwargs["quantization_config"] = build_bnb_4bit_config()
        model = AutoModel.from_pretrained(
            cfg["model_id"],
            trust_remote_code=cfg.get("trust_remote_code", True),
            **quant_kwargs,
        )
        if self.config.get("quantization") == "4bit":
            from mm_bench.peft_utils.lora_setup import prepare_for_kbit_training

            model = prepare_for_kbit_training(model)
        self._processor = AutoProcessor.from_pretrained(
            cfg["model_id"], trust_remote_code=cfg.get("trust_remote_code", True)
        )
        self._truncate_dim = cfg.get("truncate_dim")
        self._doc_prefix = cfg.get("document_prefix", "Document: ")
        self._query_prefix = cfg.get("query_prefix", "Query: ")
        self._max_text_length = self.config.get("max_text_length", 512)
        return model

    def _build_dry_run_network(self) -> nn.Module:
        self._processor = None
        self._truncate_dim = None
        self._doc_prefix = ""
        self._query_prefix = ""
        self._max_text_length = self.config.get("max_text_length", 512)
        self._dry_transform = dry_run_image_transform(self.config.get("image_size", 384))
        self._dry_tokenizer = dry_run_tokenizer(max_length=64)
        return TinyDualEncoder(
            embed_dim=self.config.get("embed_dim", 128),
            image_size=self.config.get("image_size", 384),
        )

    # -- pooling -----------------------------------------------------------
    def _embed(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Gradient-preserving reimplementation of the checkpoint's ``embed()``.

        Pools the last non-padded token of the decoder's final hidden state,
        applies the optional Matryoshka truncation, then L2-normalizes.
        """
        attention_mask = inputs.get("attention_mask")
        out = self.net(**inputs)
        hidden = out.last_hidden_state
        if attention_mask is not None and attention_mask.dim() == 2:
            idx = attention_mask.sum(dim=1) - 1
        else:
            idx = torch.full((hidden.shape[0],), hidden.shape[1] - 1, device=hidden.device, dtype=torch.long)
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]
        if self._truncate_dim is not None:
            pooled = pooled[:, : self._truncate_dim]
        return nn.functional.normalize(pooled, dim=-1)

    def encode_image(self, inputs: Any) -> torch.Tensor:
        """``inputs`` is the processor dict produced by this model's collate."""
        if self._processor is None:
            return nn.functional.normalize(self.net.encode_image(inputs), dim=-1)
        return self._embed(inputs)

    def encode_text(self, text_tokens: Any) -> torch.Tensor:
        if self._processor is None:
            return nn.functional.normalize(self.net.encode_text(text_tokens["input_ids"]), dim=-1)
        return self._embed(text_tokens)

    def encode_queries(self, queries: List[str], device: torch.device) -> torch.Tensor:
        """Encode search strings with the asymmetric ``Query:`` prefix."""
        batch = self._processor(
            text=[self._query_prefix + q for q in queries],
            padding=True, truncation=True, max_length=self._max_text_length,
            return_tensors="pt",
        ).to(device)
        return self._embed(dict(batch))

    def get_logit_scale(self) -> torch.Tensor:
        logit_scale = getattr(self.net, "logit_scale", None)
        if logit_scale is None:
            # The checkpoint ships no learnable temperature (it was trained
            # with a fixed one); expose a constant so the InfoNCE loss and
            # the Trainer's clamp both remain well-defined.
            device = next(self.net.parameters()).device
            return torch.tensor(1.0 / 0.07, device=device)
        return logit_scale.exp()

    def get_image_transform(self) -> Optional[Callable]:
        # None => ImageCaptionDataset yields raw PIL; the processor runs at
        # batch level in get_collate_fn().
        return None if self._processor is not None else self._dry_transform

    def get_text_tokenizer(self) -> Callable[[str], Dict[str, torch.Tensor]]:
        if self._processor is None:
            return self._dry_tokenizer
        processor = self._processor
        prefix = self._doc_prefix
        max_length = self._max_text_length

        def _tok(text: str) -> Dict[str, torch.Tensor]:
            return processor(
                text=[prefix + text], padding="max_length", truncation=True,
                max_length=max_length, return_tensors="pt",
            )

        return _tok

    def get_collate_fn(self) -> Callable:
        if self._processor is None:
            return super().get_collate_fn()

        processor = self._processor
        doc_prefix = self._doc_prefix
        max_length = self._max_text_length

        def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            images = [b["image"] for b in batch]
            captions = [b["caption"] for b in batch]

            image_inputs = processor(
                images=images,
                text=[IMAGE_PLACEHOLDER] * len(images),
                padding=True,
                return_tensors="pt",
            )
            out: Dict[str, Any] = {
                "pixel_values": dict(image_inputs),
                "captions": captions,
                "image_paths": [b["image_path"] for b in batch],
                "subsets": [b["subset"] for b in batch],
            }
            if "text_tokens" in batch[0]:
                out["text_tokens"] = dict(
                    processor(
                        text=[doc_prefix + c for c in captions],
                        padding=True, truncation=True, max_length=max_length,
                        return_tensors="pt",
                    )
                )
            return out

        return _collate

    # -- LoRA targeting -----------------------------------------------------
    def _find_module(self, name_fragment: str) -> Optional[nn.Module]:
        for name, module in self.net.named_modules():
            if name == name_fragment:
                return module
        return None

    def _tower_block_counts(self) -> Dict[str, int]:
        counts = {}
        for prefix, attr in (("visual", "blocks"), ("model.visual", "blocks"),
                             ("language_model", "layers"), ("model.language_model", "layers")):
            module = self._find_module(prefix)
            if module is not None and hasattr(module, attr):
                counts[f"{prefix}.{attr}"] = len(getattr(module, attr))
        return counts

    def resolve_lora_target_modules(self) -> List[str]:
        lora_cfg = self.config["lora"]
        depth_fraction = lora_cfg.get("target_module_depth_fraction", 0.3)
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
        wanted = "visual" if tower == "vision" else "language_model"
        for prefix, total in self._tower_block_counts().items():
            if wanted not in prefix:
                continue
            start = max(0, total - n)
            for name, p in self.net.named_parameters():
                if name.startswith(f"{prefix}."):
                    try:
                        block_idx = int(name[len(prefix) + 1 :].split(".")[0])
                    except ValueError:
                        continue
                    if block_idx >= start:
                        p.requires_grad_(True)
