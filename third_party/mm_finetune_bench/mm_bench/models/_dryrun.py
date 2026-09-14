"""Tiny synthetic dual-encoder used by every wrapper's ``--dry-run`` path.

Deliberately architecture-agnostic and CPU-friendly: a couple of Linear
layers standing in for the vision/text towers, sized from the model's own
config (``embed_dim``, ``image_size``) so downstream code (loss, metrics,
checkpoint shapes) exercises realistic tensor shapes without downloading or
instantiating the real multi-billion-parameter network.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class TinyDualEncoder(nn.Module):
    def __init__(self, embed_dim: int = 128, image_size: int = 224, vocab_size: int = 1000):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        patch = 16
        n_patches = (image_size // patch) ** 2
        self.vision_tower = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * image_size * image_size, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
        )
        self.text_embedding = nn.Embedding(vocab_size, 64)
        self.text_tower = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
        )
        self.visual_proj = nn.Linear(embed_dim, embed_dim)
        self.text_projection = nn.Linear(embed_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(4.6052))
        self.logit_bias = nn.Parameter(torch.tensor(-10.0))
        self._n_patches = n_patches  # unused, kept to mirror real ViT config surface

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.vision_tower(pixel_values)
        return self.visual_proj(feats)

    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.text_embedding(input_ids).mean(dim=1)
        feats = self.text_tower(emb)
        return self.text_projection(feats)


def dry_run_image_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def dry_run_tokenizer(max_length: int = 16, vocab_size: int = 1000):
    def _tokenize(text: str) -> Dict[str, torch.Tensor]:
        ids = [abs(hash((text, i))) % vocab_size for i in range(max_length)]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    return _tokenize
