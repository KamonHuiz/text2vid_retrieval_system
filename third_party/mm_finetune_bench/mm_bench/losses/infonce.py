"""Symmetric InfoNCE (CLIP-style) contrastive loss with a learnable,
clamped logit scale.

Every other sample in the batch is an in-batch negative, so the number of
negatives per anchor is ``batch - 1`` for the batch the loss actually sees.
``grad_accum_steps`` does not raise that number (each micro-batch's loss is
computed on its own); all-gathering embeddings across GPUs does -- see
``training.trainer``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, logit_scale_max: float = math.log(100.0)):
        super().__init__()
        self.logit_scale_max = logit_scale_max

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        logit_scale: torch.Tensor,
        neg_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            image_embeds / text_embeds: (B, D), L2-normalized; row i of each
                is a positive pair.
            logit_scale: scalar, already exponentiated.
            neg_mask: optional (B, B) bool, True for off-diagonal pairs that
                must NOT count as negatives (near-duplicates whose captions
                describe each other's image). Those logits are dropped from
                the softmax; the diagonal must be False.
        """
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        if neg_mask is not None:
            logits_per_image = logits_per_image.masked_fill(neg_mask, float("-inf"))
        targets = torch.arange(image_embeds.shape[0], device=image_embeds.device)
        loss_i2t = F.cross_entropy(logits_per_image, targets)
        loss_t2i = F.cross_entropy(logits_per_image.t(), targets)
        return (loss_i2t + loss_t2i) / 2.0
