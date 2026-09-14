"""Common interface every model adapter (PE-Core, MetaCLIP2, SigLIP2, Jina
Omni) implements, so ``training.trainer.Trainer`` and
``evaluation.evaluator.Evaluator`` never branch on architecture.

Design notes
------------
* Each concrete wrapper owns exactly one underlying HF/open_clip module in
  ``self.net`` and exposes ``encode_image`` / ``encode_text`` over it.
* LoRA application is delegated to ``mm_bench.peft_utils.lora_setup`` but
  invoked through ``self.apply_lora(...)`` so the wrapper can pass in its own
  architecture-specific target-module resolution (regex over block index,
  since "last N blocks" means a different module-name pattern per backbone).
* ``named_parameter_groups`` implements the differential-LR requirement:
  every named parameter is bucketed into exactly one of
  ``{vision_encoder, text_encoder, projection, lora, logit_scale, other}``
  by matching (in that priority order) the regexes in
  ``config["differential_lr_groups"]``, with any parameter whose name
  contains ``"lora_"`` always routed to the ``lora`` bucket regardless of
  which tower it lives in (adapter parameters get their own, much higher,
  learning rate -- see docs/HYPERPARAMETERS.md).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class EncoderOutput:
    image_embeds: Optional[torch.Tensor] = None   # (B, D), L2-normalized
    text_embeds: Optional[torch.Tensor] = None     # (B, D), L2-normalized
    logit_scale: Optional[torch.Tensor] = None      # scalar, already exp()'d
    logit_bias: Optional[torch.Tensor] = None        # scalar, SigLIP only


class DualEncoderWrapper(nn.Module, ABC):
    """Abstract base for a contrastive image-text model under fine-tuning."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.net: nn.Module  # set by subclass in build_network()

    # -- construction -----------------------------------------------------
    @classmethod
    def build(cls, config: Dict[str, Any], dry_run: bool = False) -> "DualEncoderWrapper":
        """Factory used by ``models.registry``.

        ``dry_run=True`` must construct a tiny, randomly-initialized stand-in
        network with the same interface (no download, no real checkpoint),
        so the rest of the pipeline (loss, optimizer, checkpoint, eval) can
        be exercised in CI/local smoke tests. Concrete subclasses implement
        this branch explicitly rather than trying to intercept
        ``from_pretrained`` -- see each wrapper's ``_build_dry_run_network``.
        """
        instance = cls(config)
        if dry_run:
            instance.net = instance._build_dry_run_network()
        else:
            instance.net = instance._build_real_network()
        instance.post_build()
        return instance

    @abstractmethod
    def _build_real_network(self) -> nn.Module:
        """Download/load the real pretrained network. Only called when
        ``dry_run=False`` -- this is the one place actual weight I/O happens.
        """

    @abstractmethod
    def _build_dry_run_network(self) -> nn.Module:
        """Build a tiny synthetic network with a matching call signature."""

    def post_build(self) -> None:
        """Optional hook subclasses can override (e.g. to cache tokenizer)."""
        return None

    # -- core interface -----------------------------------------------------
    @abstractmethod
    def encode_image(self, pixel_values: Any) -> torch.Tensor:
        """Return L2-normalized image embeddings, shape (B, D)."""

    @abstractmethod
    def encode_text(self, text_tokens: Any) -> torch.Tensor:
        """Return L2-normalized text embeddings, shape (B, D)."""

    @abstractmethod
    def get_logit_scale(self) -> torch.Tensor:
        """Return the (already-exponentiated) contrastive temperature scalar."""

    def get_logit_bias(self) -> Optional[torch.Tensor]:
        """Return the SigLIP-style additive bias, or ``None`` if not used."""
        return None

    @abstractmethod
    def get_image_transform(self) -> Callable:
        """Return a callable(PIL.Image) -> tensor/dict matching this model's
        expected preprocessing (resize, normalize, patchify, ...)."""

    @abstractmethod
    def get_text_tokenizer(self) -> Callable[[str], Dict[str, torch.Tensor]]:
        """Return a callable(str) -> dict of tokenizer tensors."""

    def get_collate_fn(self) -> Callable:
        """Return the DataLoader collate for this model.

        Defaults to the shared image/caption collate, which assumes each
        sample's image has already been turned into a fixed-shape tensor by
        ``get_image_transform()``. Models whose preprocessing is inherently
        batch-level -- e.g. a unified decoder that interleaves image
        placeholder tokens with text and therefore needs the processor to
        pad a whole batch at once -- override this and do their own
        processor call over the raw PIL images.
        """
        from mm_bench.data.dataset import collate_image_caption_batch

        return collate_image_caption_batch

    def forward(self, pixel_values: Any, text_tokens: Any) -> EncoderOutput:
        return EncoderOutput(
            image_embeds=self.encode_image(pixel_values),
            text_embeds=self.encode_text(text_tokens),
            logit_scale=self.get_logit_scale(),
            logit_bias=self.get_logit_bias(),
        )

    # -- PEFT / freezing -----------------------------------------------------
    def apply_lora(self) -> None:
        """Apply LoRA to ``self.net`` per ``self.config['lora']``.

        No-op if ``config['lora']['enabled']`` is falsy. Delegates the actual
        ``peft.get_peft_model`` call to ``peft_utils.lora_setup`` so the
        target-module resolution logic (depth-fraction -> concrete module
        names) is shared and unit-testable independent of any one backbone.
        """
        from mm_bench.peft_utils.lora_setup import apply_lora_to_module

        lora_cfg = self.config.get("lora", {})
        if not lora_cfg.get("enabled", False):
            return
        self.net = apply_lora_to_module(
            self.net,
            target_modules=self.resolve_lora_target_modules(),
            r=lora_cfg["r"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg.get("dropout", 0.05),
            bias=lora_cfg.get("bias", "none"),
            modules_to_save=lora_cfg.get("modules_to_save"),
            # peft freezes everything outside the adapter; the contrastive
            # temperature (and SigLIP's bias) must keep learning.
            extra_trainable_params=["logit_scale", "logit_bias"],
        )

    @abstractmethod
    def resolve_lora_target_modules(self) -> List[str]:
        """Expand ``config['lora']['target_modules']`` +
        ``target_module_depth_fraction`` into concrete dotted module-name
        substrings/regexes for ``peft.LoraConfig(target_modules=...)``.
        Architecture-specific because block-indexing/naming differs per
        backbone (open_clip ``visual.transformer.resblocks.{i}`` vs HF
        ``vision_model.encoder.layers.{i}`` vs a Qwen-style decoder).
        """

    def apply_partial_unfreeze(self) -> None:
        """Alternative to LoRA: freeze everything, then unfreeze the last N
        blocks of each tower plus any ``always_unfreeze`` name fragments.
        Only used when ``config['adaptation'] == 'partial_unfreeze'``.
        """
        cfg = self.config.get("partial_unfreeze", {})
        for p in self.net.parameters():
            p.requires_grad_(False)

        always = cfg.get("always_unfreeze", [])
        for name, p in self.net.named_parameters():
            if any(frag in name for frag in always):
                p.requires_grad_(True)

        self._unfreeze_last_n_blocks(
            "vision", cfg.get("unfreeze_last_n_vision_blocks", 0)
        )
        self._unfreeze_last_n_blocks(
            "text", cfg.get("unfreeze_last_n_text_blocks", 0)
        )

    def _unfreeze_last_n_blocks(self, tower: str, n: int) -> None:
        """Default no-op; overridden by subclasses that know their own
        transformer-block naming convention."""
        return None

    def configure_adaptation(self) -> None:
        """Dispatch to LoRA / partial-unfreeze / full fine-tune based on
        ``config['adaptation']``. Called once after ``build()``."""
        strategy = self.config.get("adaptation", "lora")
        if strategy == "lora":
            self.apply_lora()
        elif strategy == "partial_unfreeze":
            self.apply_partial_unfreeze()
        elif strategy == "full":
            for p in self.net.parameters():
                p.requires_grad_(True)
        else:
            raise ValueError(f"Unknown adaptation strategy: {strategy}")

    # -- differential LR -----------------------------------------------------
    def named_parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        """Bucket trainable parameters for differential-LR optimizer groups.

        Priority: explicit 'lora' bucket (by name substring) > regex groups
        from ``config['differential_lr_groups']`` (vision_encoder,
        text_encoder, projection, logit_scale) > 'other' catch-all.
        """
        patterns: Dict[str, str] = self.config.get("differential_lr_groups", {})
        compiled = {k: re.compile(v) for k, v in patterns.items()}
        groups: Dict[str, List[nn.Parameter]] = {
            "lora": [], "vision_encoder": [], "text_encoder": [],
            "projection": [], "logit_scale": [], "other": [],
        }
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_" in name.lower():
                groups["lora"].append(param)
                continue
            placed = False
            for group_name in ("logit_scale", "projection", "vision_encoder", "text_encoder"):
                pattern = compiled.get(group_name)
                if pattern and pattern.search(name):
                    groups[group_name].append(param)
                    placed = True
                    break
            if not placed:
                groups["other"].append(param)
        return {k: v for k, v in groups.items() if len(v) > 0}

    def trainable_parameters_report(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pct = 100.0 * trainable / total if total else 0.0
        return f"trainable params: {trainable:,} / {total:,} ({pct:.3f}%)"
