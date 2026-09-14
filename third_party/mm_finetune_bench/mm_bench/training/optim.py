"""Optimizer and LR-scheduler factories implementing the differential
learning-rate scheme (vision encoder / text encoder / projection heads /
LoRA adapters / logit_scale each get their own base LR -- see
``docs/HYPERPARAMETERS.md`` for the rationale).
"""

from __future__ import annotations

import math
from typing import Any, Dict

import torch

from mm_bench.models.base import DualEncoderWrapper


def build_optimizer(model: DualEncoderWrapper, train_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    lr_map: Dict[str, float] = train_cfg["learning_rate"]
    weight_decay = train_cfg.get("weight_decay", 0.05)

    param_groups = []
    for group_name, params in model.named_parameter_groups().items():
        lr = lr_map.get(group_name, lr_map.get("other", 1e-5))
        # No weight decay on 1-D params (biases, norms, logit_scale/bias) --
        # standard practice, avoids pulling the temperature toward zero.
        decay_params = [p for p in params if p.ndim > 1]
        no_decay_params = [p for p in params if p.ndim <= 1]
        if decay_params:
            param_groups.append({"params": decay_params, "lr": lr, "weight_decay": weight_decay, "name": f"{group_name}_decay"})
        if no_decay_params:
            param_groups.append({"params": no_decay_params, "lr": lr, "weight_decay": 0.0, "name": f"{group_name}_no_decay"})

    if not param_groups:
        raise ValueError(
            "No trainable parameters found -- check config['adaptation'] "
            "and that configure_adaptation() ran before build_optimizer()."
        )

    return torch.optim.AdamW(param_groups, betas=(0.9, 0.98), eps=1e-6)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: Dict[str, Any],
    num_training_steps: int,
):
    """Thin wrapper over ``transformers.get_scheduler`` so we don't
    re-implement warmup+decay math, while keeping this module's own
    ``state_dict``/``load_state_dict`` contract (delegated straight through)
    for the checkpoint code to rely on.
    """
    from transformers import get_scheduler

    warmup_steps = int(train_cfg.get("warmup_ratio", 0.03) * num_training_steps)
    return get_scheduler(
        name=train_cfg.get("lr_scheduler", "cosine"),
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )


def clamp_logit_scale(model: DualEncoderWrapper, logit_scale_max: float = math.log(100.0)) -> None:
    """In-place clamp of the *underlying learnable log-parameter* (not the
    exponentiated value returned by ``get_logit_scale()``). Must be called
    after each optimizer step, before the next forward pass, to prevent the
    temperature from exploding early in training when gradients are large.
    """
    raw_param = getattr(model.net, "logit_scale", None)
    if raw_param is not None:
        with torch.no_grad():
            raw_param.clamp_(max=logit_scale_max)
