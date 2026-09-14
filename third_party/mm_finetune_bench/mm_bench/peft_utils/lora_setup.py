"""LoRA / QLoRA application helpers built on top of ``peft``.

Kept separate from ``models.base`` so the target-module *resolution* logic
(architecture-specific, lives in each wrapper) is decoupled from the
*application* logic (generic, lives here) -- a new backbone only needs to
implement ``resolve_lora_target_modules()``, never touch this file.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

import torch.nn as nn


def build_bnb_4bit_config():
    """Return a ``transformers.BitsAndBytesConfig`` for QLoRA (NF4, double
    quantization, bf16 compute dtype). Only meaningful for
    ``transformers``-backed wrappers; open_clip has no bitsandbytes k-bit
    loading path, so open_clip configs must leave ``quantization: null``.
    """
    from transformers import BitsAndBytesConfig
    import torch

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def prepare_for_kbit_training(model: nn.Module) -> nn.Module:
    """Call before :func:`apply_lora_to_module` when ``quantization: 4bit``."""
    from peft import prepare_model_for_kbit_training

    return prepare_model_for_kbit_training(model)


def _matches(name: str, wanted: str) -> bool:
    return name == wanted or name.endswith("." + wanted)


def apply_lora_to_module(
    module: nn.Module,
    target_modules: List[str],
    r: int,
    alpha: int,
    dropout: float = 0.05,
    bias: str = "none",
    modules_to_save: Optional[List[str]] = None,
    extra_trainable_params: Optional[Iterable[str]] = None,
) -> nn.Module:
    """Wrap ``module`` with LoRA on exactly ``target_modules``.

    ``target_modules`` are fully-qualified module names from each wrapper's
    ``resolve_lora_target_modules()``. They are compiled into one anchored
    regex rather than passed as a list, because peft matches list entries by
    *suffix*: ``transformer.resblocks.12.attn`` (text tower) would also
    select ``visual.transformer.resblocks.12.attn`` (vision tower).

    ``modules_to_save`` entries are split by what they actually name:
    submodules go to peft's ``modules_to_save`` (trained in full, saved with
    the adapter); bare ``nn.Parameter``s -- e.g. open_clip's ``visual.proj``
    and ``text_projection``, which are parameters, not modules -- cannot go
    through peft and are simply unfrozen after wrapping.
    ``extra_trainable_params`` (logit scale/bias) are unfrozen the same way.
    """
    from peft import LoraConfig, get_peft_model

    if len(target_modules) == 0:
        raise ValueError(
            "resolve_lora_target_modules() returned an empty list -- LoRA "
            "would be a no-op. Check the model's _tower_block_counts() against "
            "the actually-loaded checkpoint."
        )

    module_names = [n for n, _ in module.named_modules()]
    param_names = [n for n, _ in module.named_parameters()]
    missing = [t for t in target_modules if t not in set(module_names)]
    if missing:
        raise ValueError(f"LoRA targets not found in model: {missing[:5]} (+{max(0, len(missing) - 5)} more)")

    save_modules, save_params = [], []
    for name in modules_to_save or []:
        if any(_matches(m, name) for m in module_names):
            save_modules.append(name)
        elif any(_matches(p, name) for p in param_names):
            save_params.append(name)
        else:
            raise ValueError(f"modules_to_save entry '{name}' matches neither a submodule nor a parameter")

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias,
        target_modules="|".join(re.escape(t) for t in target_modules),
        modules_to_save=save_modules or None,
    )
    peft_model = get_peft_model(module, lora_config)

    unfreeze = save_params + list(extra_trainable_params or [])
    for name, param in peft_model.named_parameters():
        if any(_matches(name, w) for w in unfreeze):
            param.requires_grad_(True)
    return peft_model


def merge_and_unload(peft_module: nn.Module) -> nn.Module:
    """Merge LoRA weights into the base model for inference/export."""
    return peft_module.merge_and_unload()
