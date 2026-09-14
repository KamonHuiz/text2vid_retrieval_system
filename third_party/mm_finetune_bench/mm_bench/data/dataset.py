"""Torch ``Dataset`` for image-caption contrastive training/eval, plus the
loader that turns the captioning pipeline's JSONL output into ``Sample``
records.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from .types import Sample, infer_subset_and_ids, read_jsonl, write_jsonl


def load_samples_from_caption_jsonl(
    captions_glob: str,
    keyframes_root: str,
    skip_missing_caption: bool = True,
) -> List[Sample]:
    """Build ``Sample`` records from one or more captioning-output JSONL files.

    Each line is expected to look like
    ``{"path": "L21/L21/L21_V009/000663.webp", "caption": "...", ["error": ...]}``
    (the schema produced by ``caption_keyframes_vllm.py``). Lines with a null
    caption or an ``error`` field are skipped by default.
    """
    samples: List[Sample] = []
    for shard_path in sorted(glob.glob(captions_glob)):
        for rec in read_jsonl(shard_path):
            caption = rec.get("caption")
            if skip_missing_caption and not caption:
                continue
            rel_path = rec["path"]
            ids = infer_subset_and_ids(rel_path)
            if ids["subset"] is None:
                continue
            samples.append(
                Sample(
                    image_path=rel_path,
                    caption=caption,
                    subset=ids["subset"],
                    video_id=ids["video_id"],
                    frame_id=ids["frame_id"],
                )
            )
    # ``keyframes_root`` is not embedded in the Sample itself (paths stay
    # relative so splits are portable across machines); dataset consumers
    # join it back on at __getitem__ time. We accept it here only to fail
    # fast if it doesn't exist.
    if not Path(keyframes_root).exists():
        raise FileNotFoundError(f"keyframes_root does not exist: {keyframes_root}")
    return samples


def load_samples_from_split_file(path: str) -> List[Sample]:
    return [Sample.from_dict(d) for d in read_jsonl(path)]


def save_samples_to_split_file(path: str, samples: List[Sample]) -> int:
    return write_jsonl(path, (s.__dict__ for s in samples))


class ImageCaptionDataset(Dataset):
    """Yields ``(image_path, PIL.Image, caption, subset)`` for one sample.

    Image decoding happens lazily in ``__getitem__`` (not eagerly at
    construction) so the dataset is cheap to instantiate even for the full
    ~330k-image corpus. Model-specific resizing/normalization is applied by
    the ``image_transform`` callable injected by the caller (each model
    wrapper exposes its own preprocessing transform so this class stays
    architecture-agnostic).
    """

    def __init__(
        self,
        samples: List[Sample],
        images_root: str,
        image_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        text_tokenizer: Optional[Callable[[str], Dict[str, torch.Tensor]]] = None,
    ):
        self.samples = samples
        self.images_root = Path(images_root)
        self.image_transform = image_transform
        self.text_tokenizer = text_tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        image_path = self.images_root / sample.image_path
        with Image.open(image_path) as im:
            image = im.convert("RGB")
            if self.image_transform is not None:
                image = self.image_transform(image)

        item: Dict[str, object] = {
            "image": image,
            "caption": sample.caption,
            "image_path": sample.image_path,
            "subset": sample.subset,
        }
        if self.text_tokenizer is not None:
            item["text_tokens"] = self.text_tokenizer(sample.caption)
        return item


def collate_image_caption_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Default collate: stacks pre-transformed image tensors, keeps captions
    as a list of strings (tokenization, if not already done per-sample, is
    left to the training loop so it can batch-tokenize efficiently).
    """
    images = batch[0]["image"]
    if isinstance(images, Image.Image):
        # No per-sample transform was configured: hand the raw PIL images
        # through untouched, for models whose processor runs batch-level
        # (see DualEncoderWrapper.get_collate_fn).
        pixel_values = [b["image"] for b in batch]
    elif isinstance(images, torch.Tensor):
        pixel_values = torch.stack([b["image"] for b in batch], dim=0)
    else:
        # Some HF processors return dicts (e.g. {"pixel_values": ..., ...});
        # in that case each sample's "image" is itself a dict of tensors.
        keys = batch[0]["image"].keys()
        pixel_values = {k: torch.stack([b["image"][k].squeeze(0) for b in batch], dim=0) for k in keys}

    out: Dict[str, object] = {
        "pixel_values": pixel_values,
        "captions": [b["caption"] for b in batch],
        "image_paths": [b["image_path"] for b in batch],
        "subsets": [b["subset"] for b in batch],
    }
    if "text_tokens" in batch[0]:
        keys = batch[0]["text_tokens"].keys()
        out["text_tokens"] = {k: torch.cat([b["text_tokens"][k] for b in batch], dim=0) for k in keys}
    return out
