"""SigLIP pairwise sigmoid loss (Zhai et al., 2023).

Every (image, text) pair in the batch is an independent binary decision,
positive only on the diagonal -- no batch-wide softmax. Loss values are not
comparable in magnitude to InfoNCE; compare runs on retrieval metrics.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigLIPLoss(nn.Module):
    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias: torch.Tensor,
        neg_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logit_scale: scalar, already exponentiated.
            logit_bias: scalar, additive.
            neg_mask: optional (B, B) bool, True for off-diagonal pairs to
                drop from the loss instead of treating them as negatives.
        """
        batch_size = image_embeds.shape[0]
        logits = image_embeds @ text_embeds.t() * logit_scale + logit_bias
        labels = 2 * torch.eye(batch_size, device=logits.device, dtype=logits.dtype) - 1
        loss = -F.logsigmoid(labels * logits)
        if neg_mask is not None:
            loss = loss.masked_fill(neg_mask, 0.0)
        return loss.sum() / batch_size
