"""Main fine-tuning loop.

Owns: dataloader construction, mixed-precision forward/backward, gradient
accumulation, the differential-LR optimizer + scheduler, periodic and
best-on-validation checkpointing, exact ``--resume_from_checkpoint``
restoration, and optional multi-GPU data parallelism.

Multi-GPU (launched with torchrun): each rank encodes its slice of the
global batch; embeddings are all-gathered *with gradient*, so every rank
computes the contrastive loss over the whole global batch (global_batch - 1
negatives per pair instead of local_batch - 1). Trainable-parameter
gradients are then all-reduced by hand. Manual all-reduce rather than
DistributedDataParallel because the wrappers are driven through
``encode_image`` / ``encode_text`` rather than ``forward()``, which DDP's
reducer never sees.

Precision layout: frozen base weights are cast to the autocast dtype (bf16
by default); every trainable parameter stays fp32 as the optimizer's master
copy.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch.utils.data import DataLoader, Sampler

from mm_bench.data.dataset import ImageCaptionDataset
from mm_bench.data.types import Sample
from mm_bench.losses import build_loss
from mm_bench.models.base import DualEncoderWrapper
from mm_bench.training.checkpoint import load_checkpoint, rotate_checkpoints, save_checkpoint
from mm_bench.training.optim import build_optimizer, build_scheduler, clamp_logit_scale
from mm_bench.utils.device import move_batch
from mm_bench.utils.logging_utils import JsonlLogger, get_console_logger
from mm_bench.utils.seed import capture_rng_state, restore_rng_state, seed_everything

logger = get_console_logger(__name__)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _base_module(net: torch.nn.Module) -> torch.nn.Module:
    get_base = getattr(net, "get_base_model", None)
    return get_base() if callable(get_base) else net


class PlannedBatchSampler(Sampler[List[int]]):
    """Yields this rank's slice of each global batch.

    Global batches come from a precomputed plan (hard-negative mining) or, if
    none is given, from a random permutation. Either way the order depends
    only on ``(seed, epoch)``, so every rank derives the same sequence and the
    ranks' slices reassemble the intended global batch after all-gather.
    ``start`` skips batches already trained on, making mid-epoch resume exact
    without decoding the skipped images.
    """

    def __init__(self, num_samples: int, global_batch: int, rank: int, world_size: int, seed: int,
                 plan: Optional[np.ndarray] = None):
        self.num_samples = num_samples
        self.global_batch = global_batch
        self.local_batch = global_batch // world_size
        self.rank = rank
        self.seed = seed
        self.plan = plan
        self.num_batches = len(plan) if plan is not None else num_samples // global_batch
        self.epoch = 0
        self.start = 0

    def set_epoch(self, epoch: int, start: int = 0) -> None:
        self.epoch = epoch
        self.start = start

    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        if self.plan is not None:
            order = torch.randperm(self.num_batches, generator=g).tolist()
            rows = (self.plan[b] for b in order[self.start :])
        else:
            perm = torch.randperm(self.num_samples, generator=g)[: self.num_batches * self.global_batch]
            table = perm.view(self.num_batches, self.global_batch).numpy()
            rows = (table[b] for b in range(self.start, self.num_batches))
        lo = self.rank * self.local_batch
        for row in rows:
            yield [int(i) for i in row[lo : lo + self.local_batch]]

    def __len__(self) -> int:
        return self.num_batches - self.start


@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0
    best_metric: Optional[float] = None


class Trainer:
    def __init__(
        self,
        model: DualEncoderWrapper,
        train_samples: List[Sample],
        images_root: str,
        train_cfg: Dict[str, Any],
        output_dir: str,
        eval_fn: Optional[Callable[[DualEncoderWrapper], Dict[str, float]]] = None,
        max_steps: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
        batch_plan: Optional[np.ndarray] = None,
        dup_ids: Optional[np.ndarray] = None,
    ):
        """
        Args:
            eval_fn: ``model -> {metric: value}`` run on rank 0 every
                ``eval_every_steps`` against a *validation* split (never the
                5% benchmark holdout); ``best.pt`` tracks its ``mean_recall``.
            max_steps: stop after this many optimizer steps (smoke tests).
            batch_plan: ``(num_batches, global_batch)`` dataset indices, e.g.
                from scripts/mine_hard_batches.py; None = random batches.
            dup_ids: per-sample near-duplicate cluster ids aligned with
                ``train_samples``; same-cluster pairs are masked out of the
                loss instead of being treated as negatives.
        """
        self.model = model
        self.train_cfg = train_cfg
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.eval_fn = eval_fn
        self.max_steps = max_steps
        self.seed = train_cfg.get("seed", 42)
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        seed_everything(self.seed)

        self.device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        precision = train_cfg.get("precision", "bf16")
        self.autocast_dtype = DTYPES[precision]
        self.use_scaler = precision == "fp16"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)
        self._place_model()

        dataset = ImageCaptionDataset(
            train_samples,
            images_root=images_root,
            image_transform=model.get_image_transform(),
            text_tokenizer=model.get_text_tokenizer(),
        )
        self.local_batch = train_cfg["per_device_batch_size"]
        self.global_batch = self.local_batch * world_size
        if batch_plan is not None and batch_plan.shape[1] != self.global_batch:
            raise ValueError(f"batch plan rows hold {batch_plan.shape[1]} samples, but "
                             f"per_device_batch_size x world_size = {self.global_batch}")
        self.grad_accum_steps = train_cfg.get("grad_accum_steps", 1)
        self.sampler = PlannedBatchSampler(len(dataset), self.global_batch, rank, world_size, self.seed, batch_plan)
        num_workers = train_cfg.get("num_workers", 8)
        self.dataloader = DataLoader(
            dataset,
            batch_sampler=self.sampler,
            num_workers=num_workers,
            prefetch_factor=train_cfg.get("prefetch_factor", 4) if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            collate_fn=model.get_collate_fn(),
            pin_memory=self.device.type == "cuda",
        )
        self.batches_per_epoch = self.sampler.num_batches
        self.dup_of = ({s.image_path: int(d) for s, d in zip(train_samples, dup_ids)}
                       if dup_ids is not None else None)

        loss_cfg = dict(model.config.get("loss", train_cfg.get("loss", {"type": "infonce"})))
        self.loss_type = loss_cfg.get("type", "infonce")
        cfg_max = train_cfg.get("loss", {}).get("logit_scale_max", math.log(100.0))
        raw_scale = getattr(model.net, "logit_scale", None)
        # Never clamp below the pretrained temperature: SigLIP checkpoints ship
        # a learned log-scale above ln(100), and clamping on step 1 would
        # change the pretrained model before any learning happens.
        self.logit_scale_max = max(cfg_max, float(raw_scale.detach().float().max())) if raw_scale is not None else cfg_max
        self.criterion = build_loss(self.loss_type, **({"logit_scale_max": self.logit_scale_max} if self.loss_type == "infonce" else {}))

        self.optimizer = build_optimizer(model, train_cfg)
        self.trainable = [p for g in self.optimizer.param_groups for p in g["params"]]
        if world_size > 1:
            for p in self.trainable:
                dist.broadcast(p.data, src=0)
        steps_per_epoch = max(1, self.batches_per_epoch // self.grad_accum_steps)
        self.num_training_steps = steps_per_epoch * train_cfg["epochs"]
        if max_steps:
            self.num_training_steps = min(self.num_training_steps, max_steps)
        self.scheduler = build_scheduler(self.optimizer, train_cfg, self.num_training_steps)

        if train_cfg.get("grad_checkpointing", True):
            self._enable_gradient_checkpointing()

        self.state = TrainerState()
        self.jsonl_logger = JsonlLogger(self.output_dir / "train_log.jsonl") if self.is_main else None
        self._masked_pairs = 0

    def _info(self, msg: str) -> None:
        if self.is_main:
            logger.info(msg)

    def _place_model(self) -> None:
        self.model.to(self.device)
        if self.autocast_dtype == torch.float32:
            return
        for p in self.model.parameters():
            p.data = p.data.to(torch.float32 if p.requires_grad else self.autocast_dtype)
        # peft casts LoRA inputs to the fp32 adapter dtype: under autocast that is a redundant fp32 activation copy (4.4 GiB OOM on PE-Core's mlp.fc2).
        for module in self.model.modules():
            if hasattr(module, "cast_input_dtype_enabled"):
                module.cast_input_dtype_enabled = False

    def _enable_gradient_checkpointing(self) -> None:
        base = _base_module(self.model.net)
        if hasattr(base, "gradient_checkpointing_enable"):
            # Non-reentrant: with LoRA only in the upper blocks the input to a
            # checkpointed segment doesn't require grad, and reentrant
            # checkpointing would then silently drop the adapter gradients.
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        elif hasattr(base, "set_grad_checkpointing"):
            base.set_grad_checkpointing(True)
        else:
            logger.warning("Model exposes no gradient-checkpointing hook; skipping.")

    # -- resume -----------------------------------------------------------
    def _rank_rng_path(self, tag: str) -> Path:
        return self.checkpoint_dir / f"rng_{tag}_rank{self.rank}.pt"

    def resume_from_checkpoint(self, path: str) -> None:
        info = load_checkpoint(path, self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                               scaler=self.scaler if self.use_scaler else None, map_location=str(self.device))
        # The checkpoint carries rank 0's RNG only; each rank's own state
        # (saved alongside) makes dropout masks resume exactly on every rank.
        rank_rng = Path(path).parent / f"rng_{Path(path).stem}_rank{self.rank}.pt"
        if rank_rng.exists():
            restore_rng_state(torch.load(rank_rng, weights_only=False))
        self.state = TrainerState(epoch=info["epoch"], global_step=info["global_step"], best_metric=info["best_metric"])
        self._info(f"Resumed from {path} at epoch={self.state.epoch} step={self.state.global_step}"
                   + (" (per-rank RNG restored)" if rank_rng.exists() else ""))

    def _resume_position(self) -> tuple[int, int]:
        """(epoch, global batches already consumed within it) for global_step."""
        consumed = self.state.global_step * self.grad_accum_steps
        epoch = consumed // self.batches_per_epoch
        return epoch, consumed - epoch * self.batches_per_epoch

    # -- core loop -----------------------------------------------------------
    def _gather(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(dist_nn.all_gather(x)) if self.world_size > 1 else x

    def _forward_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        image_embeds = self._gather(self.model.encode_image(move_batch(batch["pixel_values"], self.device)))
        text_embeds = self._gather(self.model.encode_text(move_batch(batch["text_tokens"], self.device)))

        neg_mask = None
        if self.dup_of is not None:
            dup = torch.tensor([self.dup_of[p] for p in batch["image_paths"]], device=self.device)
            if self.world_size > 1:
                parts = [torch.empty_like(dup) for _ in range(self.world_size)]
                dist.all_gather(parts, dup)
                dup = torch.cat(parts)
            neg_mask = dup.unsqueeze(0) == dup.unsqueeze(1)
            neg_mask.fill_diagonal_(False)
            self._masked_pairs = int(neg_mask.sum())

        logit_scale = self.model.get_logit_scale()
        if self.loss_type == "siglip":
            return self.criterion(image_embeds, text_embeds, logit_scale, self.model.get_logit_bias(), neg_mask=neg_mask)
        return self.criterion(image_embeds, text_embeds, logit_scale, neg_mask=neg_mask)

    def _allreduce_grads(self) -> None:
        # all_gather's backward already sums each embedding's gradient over
        # ranks, so every rank holds world_size x its share of the global-loss
        # gradient; SUM then divide yields exactly the single-process gradient
        # of the loss over the full global batch.
        grads = [p.grad for p in self.trainable if p.grad is not None]
        flat = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat)
        flat.div_(self.world_size)
        for g, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
            g.copy_(synced)

    def _done(self) -> bool:
        return bool(self.max_steps) and self.state.global_step >= self.max_steps

    def fit(self) -> TrainerState:
        max_grad_norm = self.train_cfg.get("max_grad_norm", 1.0)
        save_every = self.train_cfg.get("save_every_steps", 500)
        eval_every = self.train_cfg.get("eval_every_steps", 500)
        log_every = self.train_cfg.get("log_every_steps", 20)
        keep_last_n = self.train_cfg.get("keep_last_n_checkpoints", 3)

        self.model.train()
        start_epoch, skip = self._resume_position()
        step_at_start = self.state.global_step
        if self.eval_fn is not None and step_at_start == 0 and self.train_cfg.get("eval_at_start", False):
            self._run_eval()
        t0 = time.time()

        for epoch in range(start_epoch, self.train_cfg["epochs"]):
            self.state.epoch = epoch
            self.sampler.set_epoch(epoch, start=skip if epoch == start_epoch else 0)
            micro_step = 0
            for batch in self.dataloader:
                with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype,
                                    enabled=self.autocast_dtype != torch.float32):
                    loss = self._forward_loss(batch) / self.grad_accum_steps

                if self.use_scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                micro_step += 1
                if micro_step % self.grad_accum_steps != 0:
                    continue

                if self.use_scaler:
                    self.scaler.unscale_(self.optimizer)
                if self.state.global_step == step_at_start:
                    # An adapter the forward pass bypasses trains nothing and
                    # raises no error -- only a missing grad reveals it.
                    dead = [n for n, p in self.model.named_parameters() if p.requires_grad and p.grad is None]
                    self._info(f"first step: {len(self.trainable)} trainable tensors, {len(dead)} without grad"
                               + (f" e.g. {dead[:3]}" if dead else ""))
                if self.world_size > 1:
                    self._allreduce_grads()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable, max_grad_norm)
                if self.use_scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                clamp_logit_scale(self.model, self.logit_scale_max)

                self.state.global_step += 1
                loss_val = loss.item() * self.grad_accum_steps
                if self.is_main:
                    self.jsonl_logger.log(
                        {"loss": loss_val, "grad_norm": float(grad_norm), "logit_scale": float(self.model.get_logit_scale()),
                         "lr": self.scheduler.get_last_lr()[0], "masked_pairs": self._masked_pairs,
                         "epoch": epoch, "elapsed_sec": time.time() - t0},
                        step=self.state.global_step,
                    )

                if self.state.global_step % log_every == 0 or self._done():
                    done = self.state.global_step - step_at_start
                    elapsed = time.time() - t0
                    rate = done * self.global_batch * self.grad_accum_steps / elapsed
                    eta = (self.num_training_steps - self.state.global_step) * elapsed / done
                    mem = torch.cuda.max_memory_allocated() / 1e9 if self.device.type == "cuda" else 0.0
                    self._info(
                        f"step {self.state.global_step}/{self.num_training_steps} loss={loss_val:.4f} "
                        f"gnorm={float(grad_norm):.3f} scale={float(self.model.get_logit_scale()):.1f} "
                        f"masked={self._masked_pairs} lr={self.scheduler.get_last_lr()[0]:.2e} "
                        f"{rate:.1f} img/s eta={eta / 3600:.2f}h mem={mem:.1f}GB"
                    )

                if self.state.global_step % save_every == 0:
                    tag = f"step_{self.state.global_step}"
                    torch.save(dataclasses.asdict(capture_rng_state()), self._rank_rng_path(tag))
                    rotate_checkpoints(str(self.checkpoint_dir), keep_last_n, pattern=f"rng_step_*_rank{self.rank}.pt")
                    if self.is_main:
                        self._save(tag=tag)
                        rotate_checkpoints(str(self.checkpoint_dir), keep_last_n)

                if self.eval_fn is not None and self.state.global_step % eval_every == 0:
                    self._run_eval()

                if self._done():
                    break
            if self._done():
                break

        if self.eval_fn is not None and self.state.global_step % eval_every != 0:
            self._run_eval()
        if self.is_main:
            self._save(tag="last")
        if self.world_size > 1:
            dist.barrier()
        return self.state

    def _save(self, tag: str) -> None:
        path = self.checkpoint_dir / f"{tag}.pt"
        save_checkpoint(str(path), self.model, self.optimizer, self.scheduler,
                        self.scaler if self.use_scaler else None, epoch=self.state.epoch,
                        global_step=self.state.global_step, best_metric=self.state.best_metric)
        logger.info(f"Saved checkpoint: {path}")

    def _run_eval(self) -> None:
        if self.is_main:
            self.model.eval()
            t = time.time()
            with torch.no_grad():
                metrics = self.eval_fn(self.model)
            self.model.train()
            self.jsonl_logger.log({"eval": metrics}, step=self.state.global_step)
            primary = metrics.get("mean_recall")
            improved = primary is not None and (self.state.best_metric is None or primary > self.state.best_metric)
            logger.info(f"eval @ step {self.state.global_step}: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                        + f" ({time.time() - t:.0f}s)" + (" <- new best" if improved else ""))
            if improved:
                self.state.best_metric = primary
                self._save(tag="best")
        if self.world_size > 1:
            dist.barrier()
