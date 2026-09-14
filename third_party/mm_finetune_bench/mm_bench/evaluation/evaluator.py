"""Zero-shot (Phase 1) / fine-tuned (Phase 2) retrieval evaluation over the
5% benchmark holdout.

``Evaluator.run`` is deliberately checkpoint-agnostic: pass
``checkpoint_path=None`` for the raw pretrained baseline (Phase 1), or a
path produced by ``training.checkpoint.save_checkpoint`` for Phase 2 -- both
write to the same JSON schema so ``evaluation.compare`` can diff them
mechanically.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from mm_bench.data.dataset import ImageCaptionDataset
from mm_bench.data.types import Sample
from mm_bench.evaluation.retrieval_metrics import compute_retrieval_metrics, flatten_metrics
from mm_bench.models.base import DualEncoderWrapper
from mm_bench.training.checkpoint import load_checkpoint
from mm_bench.utils.device import move_batch
from mm_bench.utils.logging_utils import get_console_logger

logger = get_console_logger(__name__)


class Evaluator:
    def __init__(
        self,
        model: DualEncoderWrapper,
        benchmark_samples: List[Sample],
        images_root: str,
        batch_size: int = 256,
        num_workers: int = 8,
        device: Optional[torch.device] = None,
        precision: str = "fp16",
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
        self.model.to(device=self.device, dtype=self.dtype)

        dataset = ImageCaptionDataset(
            benchmark_samples,
            images_root=images_root,
            image_transform=model.get_image_transform(),
            text_tokenizer=model.get_text_tokenizer(),
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=model.get_collate_fn(),
            drop_last=False,
            pin_memory=self.device.type == "cuda",
        )
        self.n_samples = len(benchmark_samples)

    @torch.no_grad()
    def _embed_all(self):
        self.model.eval()
        image_embeds_chunks, text_embeds_chunks = [], []
        total_batches = len(self.dataloader)
        t0 = time.time()
        for i, batch in enumerate(self.dataloader):
            image_embeds_chunks.append(
                self.model.encode_image(move_batch(batch["pixel_values"], self.device, self.dtype)).float().cpu()
            )
            text_embeds_chunks.append(
                self.model.encode_text(move_batch(batch["text_tokens"], self.device, self.dtype)).float().cpu()
            )
            if i % 20 == 0:
                done = sum(c.shape[0] for c in image_embeds_chunks)
                rate = done / max(time.time() - t0, 1e-6)
                logger.info(f"  embed {i}/{total_batches} batches | {done} pairs | {rate:.1f} pairs/s")

        return torch.cat(image_embeds_chunks, dim=0), torch.cat(text_embeds_chunks, dim=0)

    def run(
        self,
        model_name: str,
        phase: str,
        output_path: str,
        checkpoint_path: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        embeddings_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            phase: ``"baseline"`` (Phase 1, zero-shot) or ``"finetuned"``
                (Phase 2). Purely descriptive metadata written into the
                output JSON -- whether weights are actually fine-tuned is
                determined by whether ``checkpoint_path`` is given.
            embeddings_path: if set, also save the float32 image/text
                embeddings (split order) plus the image paths as ``.npz``,
                so fusion experiments can reuse them without re-encoding.
        """
        if checkpoint_path is not None:
            load_checkpoint(checkpoint_path, self.model, map_location=str(self.device), restore_rng=False)
            logger.info(f"Loaded checkpoint for evaluation: {checkpoint_path}")

        t0 = time.time()
        image_embeds, text_embeds = self._embed_all()
        elapsed = time.time() - t0

        if embeddings_path is not None:
            import numpy as np

            Path(embeddings_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                embeddings_path,
                image=image_embeds.numpy(),
                text=text_embeds.numpy(),
                paths=np.array([s.image_path for s in self.dataloader.dataset.samples]),
            )
            logger.info(f"Wrote embeddings -> {embeddings_path}")

        metrics = compute_retrieval_metrics(image_embeds, text_embeds)
        result = {
            "model_name": model_name,
            "phase": phase,
            "checkpoint_path": checkpoint_path,
            "n_samples": self.n_samples,
            "eval_seconds": elapsed,
            "metrics": metrics,
            "metrics_flat": flatten_metrics(metrics),
        }
        if extra_metadata:
            result["metadata"] = extra_metadata

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Wrote {phase} metrics for {model_name} -> {output_path}")
        return result
