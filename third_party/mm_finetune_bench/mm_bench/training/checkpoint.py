"""Atomic checkpoint save/load with full resume semantics.

A checkpoint bundles: model weights, optimizer state, LR scheduler state, AMP
``GradScaler`` state, and RNG state (python/numpy/torch/cuda), plus
``{epoch, global_step, best_metric}``. Writing goes through a temp-file +
``os.replace`` so a crash mid-save can never leave a truncated checkpoint at
the canonical path.

Weights: under PEFT we save *every parameter with ``requires_grad``* by
name -- LoRA A/B, peft ``modules_to_save`` copies, and parameters unfrozen by
hand (open_clip's ``visual.proj``/``text_projection``, ``logit_scale``). The
last group is invisible to ``peft.get_peft_model_state_dict``, which is why
that helper is not used here.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from mm_bench.utils.seed import capture_rng_state, restore_rng_state


def _unwrap_net(model_or_module: Any) -> torch.nn.Module:
    return model_or_module.net if hasattr(model_or_module, "net") else model_or_module


def _is_peft_model(net: torch.nn.Module) -> bool:
    try:
        from peft import PeftModel

        return isinstance(net, PeftModel)
    except ImportError:
        return False


def save_checkpoint(
    path: str,
    model: Any,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    scaler: Optional[Any],
    epoch: int,
    global_step: int,
    best_metric: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    net = _unwrap_net(model)
    is_peft = _is_peft_model(net)

    if is_peft:
        model_state = {n: p.detach().cpu() for n, p in net.named_parameters() if p.requires_grad}
    else:
        model_state = net.state_dict()

    checkpoint = {
        "is_peft": is_peft,
        "model_state": model_state,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "rng_state": dataclasses.asdict(capture_rng_state()),
        "extra": extra or {},
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(
    path: str,
    model: Any,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    map_location: str = "cpu",
    strict: bool = True,
    restore_rng: bool = True,
) -> Dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint` and restore every
    piece of state onto the objects passed in. Returns the training-loop
    bookkeeping dict ``{epoch, global_step, best_metric, extra}``.

    For PEFT checkpoints the model must already be wrapped with the same
    LoRA config (``build_model(..., apply_adaptation=True)``) so parameter
    names line up; any saved key that doesn't land on a model parameter is
    an error rather than a silent no-op.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    net = _unwrap_net(model)

    if checkpoint["is_peft"]:
        result = net.load_state_dict(checkpoint["model_state"], strict=False)
        if result.unexpected_keys:
            raise RuntimeError(
                f"{len(result.unexpected_keys)} checkpoint tensors matched no model parameter "
                f"(e.g. {result.unexpected_keys[:3]}). Was the model built with the same LoRA config?"
            )
    else:
        net.load_state_dict(checkpoint["model_state"], strict=strict)

    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    if restore_rng and checkpoint.get("rng_state") is not None:
        restore_rng_state(checkpoint["rng_state"])

    return {
        "epoch": checkpoint["epoch"],
        "global_step": checkpoint["global_step"],
        "best_metric": checkpoint.get("best_metric"),
        "extra": checkpoint.get("extra", {}),
    }


def rotate_checkpoints(checkpoint_dir: str, keep_last_n: int, pattern: str = "step_*.pt") -> None:
    """Delete all but the ``keep_last_n`` most-recent step checkpoints
    (by mtime). ``last.pt`` / ``best.pt`` live outside the glob and are
    never touched.
    """
    paths = sorted(Path(checkpoint_dir).glob(pattern), key=lambda p: p.stat().st_mtime)
    for stale in paths[:-keep_last_n] if keep_last_n > 0 else paths:
        stale.unlink(missing_ok=True)
