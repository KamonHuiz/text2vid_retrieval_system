"""Lightweight structured logging shared by all scripts.

Deliberately avoids pulling in a heavyweight experiment tracker; swap
``JsonlLogger`` for a W&B/TensorBoard logger later without touching call
sites, since every script only ever calls ``logger.log(dict)``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union


def get_console_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class JsonlLogger:
    """Appends one JSON object per ``log()`` call to a ``.jsonl`` file.

    Safe to reopen across a resumed run: it opens in append mode and never
    truncates, so metrics from before a crash are preserved alongside
    metrics from after the resume.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any], step: Optional[int] = None) -> None:
        payload = {"timestamp": time.time()}
        if step is not None:
            payload["step"] = step
        payload.update(record)
        with open(self.path, "a") as f:
            f.write(json.dumps(payload) + "\n")
