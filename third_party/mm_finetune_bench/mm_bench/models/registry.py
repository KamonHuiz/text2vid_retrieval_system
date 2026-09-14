"""Factory mapping a config's ``wrapper`` key to a concrete
``DualEncoderWrapper`` subclass, and applying the configured adaptation
strategy after construction.

Importing this module does not import any of the heavy backend libraries
(``open_clip_torch``, ``transformers``) at module scope for the wrappers
that don't end up being used -- each wrapper module performs its own lazy
imports inside ``_build_real_network`` so ``--dry-run`` runs never require
those packages to even be installed.
"""

from __future__ import annotations

from typing import Any, Dict, Type

from .base import DualEncoderWrapper

_REGISTRY: Dict[str, str] = {
    # wrapper key -> "module_path:ClassName", resolved lazily in build_model
    "pe_core_bigg": "mm_bench.models.pe_core_bigg:PECoreBigGWrapper",
    "metaclip2": "mm_bench.models.metaclip2:MetaClip2Wrapper",
    "siglip2": "mm_bench.models.siglip2:SigLIP2Wrapper",
    "jina_omni": "mm_bench.models.jina_omni:JinaOmniWrapper",
    # Any other plain open_clip CLIP-style checkpoint (e.g. DFN5B ViT-H-378)
    # needs only a config, not a new wrapper class.
    "open_clip": "mm_bench.models.open_clip_common:OpenCLIPDualEncoderWrapper",
    "bge_vl": "mm_bench.models.bge_vl:BGEVLWrapper",
}


def _resolve(dotted: str) -> Type[DualEncoderWrapper]:
    module_path, class_name = dotted.split(":")
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def build_model(
    config: Dict[str, Any],
    dry_run: bool = False,
    apply_adaptation: bool = True,
) -> DualEncoderWrapper:
    """Construct a wrapper from its config, apply the configured adaptation
    strategy (LoRA / partial-unfreeze / full), and return it in ``train()``
    mode with gradients enabled only on the intended parameters.

    Args:
        apply_adaptation: set False for pure inference over the *pretrained*
            weights (baseline eval, corpus encoding). LoRA is a mathematical
            no-op at init (B is zero-initialized), but wrapping in a
            ``PeftModel`` still costs wall-clock per forward and changes the
            module tree, so skip it when no adapter is going to be trained
            or loaded.
    """
    wrapper_key = config["wrapper"]
    if wrapper_key not in _REGISTRY:
        raise KeyError(f"Unknown model wrapper '{wrapper_key}'. Known: {list(_REGISTRY)}")
    cls = _resolve(_REGISTRY[wrapper_key])
    model = cls.build(config, dry_run=dry_run)
    if apply_adaptation:
        model.configure_adaptation()
    return model


def available_models() -> Dict[str, str]:
    return dict(_REGISTRY)
