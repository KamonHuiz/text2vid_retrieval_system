"""Synthetic on-disk dataset used by ``--dry-run`` in ``scripts/train.py`` /
``scripts/benchmark.py``.

Writes a handful of tiny, randomly-generated RGB images to a temp directory
and pairs them with short random captions, then returns real ``Sample``
records pointing at real files -- so the *entire* real data pipeline
(``ImageCaptionDataset``, ``DataLoader``, collate, tokenizer/transform calls)
gets exercised end-to-end, rather than special-cased/mocked at the dataset
layer.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

from PIL import Image

from .types import Sample

_WORDS = [
    "person", "walking", "street", "red", "car", "blue", "sky", "market",
    "vendor", "basket", "smiling", "holding", "table", "chair", "river",
    "boat", "field", "green", "shirt", "hat", "crowd", "building", "sign",
]


def _random_caption(rng: random.Random, min_words: int = 6, max_words: int = 12) -> str:
    n = rng.randint(min_words, max_words)
    return " ".join(rng.choice(_WORDS) for _ in range(n)).capitalize() + "."


def make_dry_run_dataset(
    root_dir: str,
    num_samples: int = 32,
    image_size: int = 224,
    num_subsets: int = 2,
    seed: int = 0,
) -> Tuple[List[Sample], str]:
    """Returns ``(samples, images_root)``. ``images_root`` == ``root_dir``.
    """
    rng = random.Random(seed)
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    samples: List[Sample] = []
    for i in range(num_samples):
        subset = f"L{i % num_subsets:02d}"
        video_id = f"{subset}_V{(i // num_subsets) % 5:03d}"
        frame_id = f"{i:06d}"
        rel_path = f"{subset}/{video_id}/{frame_id}.png"
        abs_path = root / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        pixels = bytes(rng.getrandbits(8) for _ in range(image_size * image_size * 3))
        Image.frombytes("RGB", (image_size, image_size), pixels).save(abs_path)

        samples.append(
            Sample(
                image_path=rel_path,
                caption=_random_caption(rng),
                subset=subset,
                video_id=video_id,
                frame_id=frame_id,
            )
        )
    return samples, str(root)
