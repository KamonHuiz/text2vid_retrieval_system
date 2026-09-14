"""Deterministic seeding and RNG state (de)serialization.

RNG state capture is used by ``training.checkpoint`` so that a resumed run
draws the exact same sequence of random numbers (data shuffling order,
dropout masks, augmentation choices, ...) it would have drawn had it never
been interrupted.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch (CPU + all visible CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class RngState:
    python_state: Any
    numpy_state: Any
    torch_state: torch.Tensor
    cuda_state: Any  # list[torch.Tensor] | None


def capture_rng_state() -> RngState:
    return RngState(
        python_state=random.getstate(),
        numpy_state=np.random.get_state(),
        torch_state=torch.get_rng_state(),
        cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def restore_rng_state(state: RngState | Dict[str, Any]) -> None:
    """Restore RNG state produced by :func:`capture_rng_state`.

    Accepts either an ``RngState`` instance or the plain dict form it is
    serialized to inside a checkpoint (see ``training.checkpoint``).
    """
    if isinstance(state, dict):
        state = RngState(**state)
    random.setstate(state.python_state)
    np.random.set_state(state.numpy_state)
    torch.set_rng_state(state.torch_state.cpu() if isinstance(state.torch_state, torch.Tensor) else state.torch_state)
    if state.cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() if isinstance(s, torch.Tensor) else s for s in state.cuda_state])
