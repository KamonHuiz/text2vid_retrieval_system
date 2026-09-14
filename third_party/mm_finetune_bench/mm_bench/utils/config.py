"""YAML config loading with shallow-merge overrides.

Every script loads a *model* config (architecture, LoRA target modules,
per-model default loss) and, for training, a *train* config (optimizer,
schedule, precision). ``load_config`` also supports ``--set key.path=value``
style CLI overrides via :func:`apply_dotted_overrides`.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

PathLike = Union[str, Path]


def load_yaml(path: PathLike) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(*paths: PathLike) -> Dict[str, Any]:
    """Load and deep-merge one or more YAML files, later files taking precedence."""
    cfg: Dict[str, Any] = {}
    for p in paths:
        cfg = deep_merge(cfg, load_yaml(p))
    return cfg


def apply_dotted_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply ``key.path=value`` CLI overrides (e.g. ``lora.r=32``).

    Values are parsed with ``yaml.safe_load`` so ``true``/``42``/``3.0``
    round-trip to their native python types; anything that doesn't parse as
    YAML scalar falls back to a plain string.
    """
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Malformed override '{item}', expected key.path=value")
        key_path, raw_value = item.split("=", 1)
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError:
            value = raw_value
        keys = key_path.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return cfg
