"""Shared dataset record schema.

A ``Sample`` is one (image, caption) pair. ``subset`` is the top-level
lesson/category id ("L21".."L30" for the AIC keyframe corpus) that the 5%
benchmark split is stratified over -- see ``mm_bench.data.splits``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional


@dataclass(frozen=True)
class Sample:
    image_path: str        # path relative to the configured images root
    caption: str
    subset: str             # e.g. "L21" -- first path component
    video_id: Optional[str] = None    # e.g. "L21_V009", if resolvable
    frame_id: Optional[str] = None    # e.g. "000663", if resolvable
    extra: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Sample":
        return Sample(
            image_path=d["image_path"],
            caption=d["caption"],
            subset=d["subset"],
            video_id=d.get("video_id"),
            frame_id=d.get("frame_id"),
            extra=d.get("extra"),
        )


def infer_subset_and_ids(relative_image_path: str) -> Dict[str, Optional[str]]:
    """Derive ``subset``/``video_id``/``frame_id`` from a keyframes-relative path.

    Expected layout (as produced by the AIC extraction pipeline):
        L21/L21/L21_V009/000663.webp
        ^^^ subset   ^^^ video_id     ^^^^^^ frame_id (stem)
    Falls back gracefully (``None``) for paths that don't match, so this
    utility never raises on unexpected layouts -- callers decide whether a
    missing subset is fatal.
    """
    parts = Path(relative_image_path).parts
    subset = parts[0] if len(parts) >= 1 else None
    video_id = None
    for part in parts:
        if "_V" in part:
            video_id = part
            break
    frame_id = Path(relative_image_path).stem
    return {"subset": subset, "video_id": video_id, "frame_id": frame_id}


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n
