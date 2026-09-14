# Hyperparameter Impact: PE Core BigG / MetaCLIP 2 / SigLIP2 / Jina Omni v5

This document is the deliverable-1 companion to `configs/models/*.yaml`: it
explains *why* each config is shaped the way it is, so a future change to
`r`, a learning rate, or the loss type is made with the trade-off in mind
rather than by trial and error alone.

---

## 1. Cross-cutting concepts (apply to all four models)

### 1.1 Adaptation strategy: LoRA vs QLoRA vs partial unfreeze vs full FT

| Strategy | When it's the right call | Cost | Risk |
|---|---|---|---|
| **Full fine-tune** | Never, for these four models on a single-node budget (0.4B-2B+ params each; Adam states alone would need 3-4x model size in optimizer memory). Not offered as the default for any config here. | Highest (weights + Adam m/v + activations) | Catastrophic forgetting of the pretrained image-text alignment is easy with a naively-chosen LR. |
| **Partial unfreeze** (last N blocks) | Small domain shift from pretraining data, want the model to keep almost all of its prior behavior, willing to accept a coarser-grained adaptation than LoRA's per-block low-rank update. | Medium (optimizer states only for unfrozen params) | Under-parameterized relative to LoRA at equal unfrozen-block count -- last-N-blocks-only can't touch anything upstream. |
| **LoRA** (default for all 4 configs) | Standard choice here: injects a small, high-LR-tolerant set of adapter parameters into targeted linear layers, leaves 99%+ of weights untouched (hence untouched = no forgetting risk on those weights), and keeps optimizer memory to ~the adapter size. | Low | Rank too low underfits; rank too high approaches full-FT cost/behavior. |
| **QLoRA** (opt-in via `quantization: 4bit`) | Same adaptation quality as LoRA, but the frozen base weights are loaded in 4-bit NF4 -- use when VRAM (not compute) is the binding constraint, e.g. running PE-Core-BigG or a Jina Omni checkpoint on <40GB cards. Only wired up for the `transformers`-backed wrappers (SigLIP2, Jina Omni) in this codebase; `open_clip` has no first-party bitsandbytes k-bit loading path, so PE-Core-BigG/MetaCLIP2 configs must stay `quantization: null`. | Lowest VRAM | Slower per-step (dequant overhead), small accuracy tax vs bf16 LoRA. |

**Why LoRA is the default across the board, not per-model:** all four
architectures are *pretrained dual encoders with an already-excellent
image-text alignment* -- the fine-tuning goal is narrow domain adaptation
(AIC keyframe/caption style) rather than teaching a new capability from
scratch. That is exactly LoRA's sweet spot: it can shift the geometry of the
existing embedding space without the risk of the catastrophic forgetting a
high global LR full fine-tune would risk.

### 1.2 LoRA rank (r) and scaling factor (alpha)

- **r** controls the adapter's capacity: the update to each targeted weight
  matrix `W` is `ΔW = (alpha/r) · B·A` with `B ∈ R^{d×r}`, `A ∈ R^{r×k}`.
  Larger `r` = more capacity to represent an update, but also more
  parameters, more overfitting risk on a domain-specific caption dataset,
  and a slower/heavier merge step.
- **alpha** rescales the adapter's contribution independent of `r`; the
  `alpha/r` ratio is what actually matters, not `alpha` alone. All four
  configs use `alpha = 2r`, a widely-used default that keeps the adapter's
  effective LR roughly rank-independent as `r` is swept.
- **Per-model choice in this repo:**
  - PE-Core-BigG, MetaCLIP2: `r=16, alpha=32` -- these are the two largest
    vision towers (ViT-bigG/ViT-H); a moderate rank balances capacity
    against the parameter count of `attn.out_proj` + MLP layers, which are
    already wide (>1024-dim) in these architectures, so even `r=16`
    represents a meaningful low-rank subspace.
  - SigLIP2 (`ViT-gopt-16-384`, giant-opt tower): `r=8, alpha=16` -- SigLIP's
    sigmoid loss already tolerates smaller effective batches/negatives
    better than softmax InfoNCE (see 1.4), so we intentionally under-provision
    adapter capacity here to keep the giant-opt tower's fine-tune cheap;
    empirically SigLIP models have also been observed to need less
    adaptation capacity to shift retrieval behavior than CLIP-style models.
  - Jina Omni v5 (Qwen-family decoder backbone): `r=16, alpha=32`, but
    **shallower** (`target_module_depth_fraction: 0.3` vs `0.5` for the
    others) -- decoder-style backbones are more prone to representation
    collapse when adapted too deep into early layers that encode general
    language structure; restrict LoRA to the last 30% of blocks.

### 1.3 Differential learning rates

Every config's `differential_lr_groups` + `configs/train_default.yaml`'s
`learning_rate` map partitions trainable parameters into five buckets:

| Group | Typical LR here | Rationale |
|---|---|---|
| `vision_encoder` / `text_encoder` (only non-zero when `adaptation: partial_unfreeze`, or for any `modules_to_save` params living inside a tower) | `5e-6` | These are pretrained weights; a large LR destroys the pretrained alignment before the adapter has a chance to specialize it. |
| `projection` (`modules_to_save`: pooling heads, projection layers) | `5e-5` | 10x the tower LR -- these are small, cheap-to-retrain layers that sit at the exact bottleneck where image/text spaces meet; they benefit from moving faster than the backbone but still shouldn't be as volatile as a freshly-initialized adapter. |
| `lora` (all LoRA A/B matrices, any tower) | `5e-4` | 100x the tower LR -- LoRA's `B` matrix is zero-initialized by construction (so the adapter starts as a no-op); a high LR is what lets it actually learn something meaningful within a small number of epochs on a domain-specific dataset. |
| `logit_scale` / `logit_bias` | `5e-4` | Single scalar(s); needs to move fast to recalibrate the contrastive temperature to the new data's difficulty (in-domain captions are typically easier to discriminate than the model's original pretraining distribution, so the optimal temperature shifts). |

**Practical effect if these ratios are collapsed to one LR:** the most
common failure mode observed in CLIP-style fine-tuning is a single LR that's
tuned for the LoRA/projection parameters (i.e. `1e-4`-ish) also being applied
to any unfrozen backbone weights, which reliably wrecks zero-shot transfer
outside the fine-tuning domain within a few hundred steps -- this is why
`vision_encoder`/`text_encoder` sit two full orders of magnitude below
`lora` even though the backbone is technically frozen 99% of the time this
matters (`modules_to_save` params and, for `partial_unfreeze`, real backbone
layers, are the exception that make this bucket meaningful).

### 1.4 Loss formulation: InfoNCE vs SigLIP sigmoid

| | InfoNCE (softmax, symmetric) | SigLIP (pairwise sigmoid) |
|---|---|---|
| Used by | PE-Core-BigG, MetaCLIP2, Jina Omni (configs) | SigLIP2 (config) |
| Normalization | Row-wise softmax over the *entire batch* -- every negative in the batch competes for probability mass. | None -- independent binary decision per pair, `-log(sigmoid(z))`/`-log(sigmoid(-z))`. |
| Batch-size sensitivity | Needs a **large** batch (typically >=2k in the original CLIP work, though LoRA fine-tuning on a narrower domain tolerates far smaller) because the softmax's discriminative signal is diluted with too few negatives. | Scales to very large batches *more cheaply* per Zhai et al. (no batch-wide softmax to materialize), and degrades more gracefully at small batch size since every pair is its own independent loss term. |
| Learnable params | `logit_scale` only (temperature). | `logit_scale` **and** `logit_bias` -- the bias corrects for the extreme class imbalance of "1 positive vs (B-1) negatives per row" in a pointwise loss; without it the sigmoid saturates toward "everything is a negative" early in training. |
| Why this repo's default per model | PE-Core/MetaCLIP2/Jina Omni were *pretrained* with an InfoNCE-family loss -- keeping the fine-tuning objective consistent with the pretraining objective avoids fighting the loss landscape the pretrained logit_scale was calibrated for. | SigLIP2 was *pretrained* with the sigmoid loss; same consistency argument in the other direction. |

`losses/infonce.py` and `losses/siglip_loss.py` are both available to every
model via `loss.type` in `configs/train_default.yaml` if a controlled A/B
of loss formulation *holding architecture fixed* is ever wanted -- but the
per-model config default should track whichever loss that checkpoint was
actually pretrained with.

### 1.5 Batch size, gradient accumulation, and effective negatives

**The critical distinction:** for a contrastive loss, `grad_accum_steps`
increases the *optimizer's effective step size* (it averages gradients
over more micro-batches before stepping) but does **not** increase the
number of in-batch negatives any single micro-batch's loss computation
sees -- `InfoNCELoss`/`SigLIPLoss` are each computed independently per
micro-batch, inside the accumulation loop, *before* accumulation happens.

Concretely, with `per_device_batch_size: 128` and `grad_accum_steps: 4`:
- Effective *optimizer* batch size = 128 x 4 = 512 (affects LR schedule /
  gradient noise).
- Effective *contrastive negative pool* per loss computation = 127 (one
  micro-batch's worth), **not** 511.

If more negatives are needed than a single micro-batch can hold in VRAM,
the correct lever is **not** `grad_accum_steps` -- it is either (a)
increasing `per_device_batch_size` directly (bounded by VRAM), or (b)
adding a cross-rank gradient-preserving all-gather of embeddings before the
loss (explicitly called out as unimplemented in `losses/infonce.py`'s
docstring -- multi-GPU runs that need this should add
`torch.distributed.nn.all_gather` at the `Trainer._forward_loss` call site).
This is why `configs/train_default.yaml` documents both numbers and their
distinct roles rather than presenting a single "batch size" knob.

### 1.6 Mixed precision and FlashAttention-2

- **bf16 is the default precision** (`configs/train_default.yaml:
  precision: bf16`) over fp16: bf16 has the same exponent range as fp32 (no
  `GradScaler` needed, one less moving part to get wrong across a resume),
  at the cost of fewer mantissa bits -- acceptable for these models since
  LoRA's adapter parameters and the frozen backbone's forward pass are both
  precision-tolerant relative to full fine-tuning.
- **fp16 remains supported** (`Trainer` conditionally instantiates a
  `GradScaler`, whose state is checkpointed/restored) for any hardware
  where bf16 tensor cores aren't available.
- **FlashAttention-2** (`flash_attention: true`) is requested via
  `attn_implementation="flash_attention_2"` for the two `transformers`
  -backed wrappers (SigLIP2, Jina Omni); open_clip's PE-Core/MetaCLIP2
  attention backend is whatever that specific checkpoint's open_clip model
  definition ships (typically PyTorch SDPA, which already dispatches to a
  fused flash-attention kernel on supported hardware without an explicit
  flag). The practical impact is largest for SigLIP2's giant-opt tower and
  for Jina Omni's decoder backbone, both of which process long token
  sequences per image (SigLIP2: high patch count at 384px; Jina Omni:
  long-context text side up to 512 tokens) where attention's quadratic term
  dominates step time.
- **Gradient checkpointing** (`grad_checkpointing: true`) trades ~30% more
  compute time for a large activation-memory reduction, and is what makes
  the larger `per_device_batch_size` values in `train_default.yaml`
  feasible for PE-Core-BigG/MetaCLIP2's largest towers at all on a single
  card.

---

## 2. Per-model summary tables

### 2.1 PE Core BigG

| Aspect | Value | Notes |
|---|---|---|
| Backend | `open_clip_torch`, `hf-hub:facebook/PE-Core-bigG14-448` | |
| Adaptation | LoRA, `r=16, alpha=32`, dropout 0.05 | Depth fraction 0.5 (last half of blocks, both towers) |
| Target modules | `attn.out_proj`, `mlp.c_fc`, `mlp.c_proj` | Post-attention projection + both MLP layers; deliberately excludes `attn.in_proj_weight` (fused QKV) since PEFT's `Linear`-wrapping assumptions are cleaner against separate `out_proj` |
| `modules_to_save` | `visual.proj`, `text_projection` | Full-precision, fully trainable projection heads |
| Loss | InfoNCE, learnable `logit_scale` | Matches pretraining objective |
| Differential LR | vision/text `5e-6`, projection `5e-5`, LoRA `5e-4`, logit_scale `5e-4` | |
| Precision | bf16 + grad checkpointing | Largest tower in the suite (bigG); checkpointing close to mandatory at reasonable batch size |
| Quantization | None (QLoRA not available for open_clip backend in this codebase) | Use `partial_unfreeze` with a small `unfreeze_last_n_*` instead if VRAM-bound |

### 2.2 MetaCLIP 2

| Aspect | Value | Notes |
|---|---|---|
| Backend | `open_clip_torch`, `hf-hub:facebook/MetaCLIP-2-worldwide-huge` | |
| Adaptation | LoRA, `r=16, alpha=32`, dropout 0.05 | Identical structure to PE-Core-BigG (shares `OpenCLIPDualEncoderWrapper`) |
| Target modules | `attn.out_proj`, `mlp.c_fc`, `mlp.c_proj` | |
| `modules_to_save` | `visual.proj`, `text_projection` | |
| Loss | InfoNCE | |
| Differential LR | Same scheme as PE-Core-BigG | |
| Precision | bf16 + grad checkpointing | ViT-H-sized tower -- comfortably fits with checkpointing at the configured batch size |

### 2.3 ViT-gopt-16-SigLIP2-384

| Aspect | Value | Notes |
|---|---|---|
| Backend | `transformers.Siglip2Model`, `google/siglip2-giant-opt-patch16-384` | |
| Adaptation | LoRA, `r=8, alpha=16` (intentionally lower than the other 3 -- see 1.2) | Depth fraction 0.5 |
| Target modules | `attention.{q,k,v,out}_proj` | All four attention projections (vs. only `out_proj` for the open_clip models) since HF's `SiglipAttention` names them separately rather than as a fused `in_proj_weight` |
| `modules_to_save` | `vision_model.head`, `text_model.head` | Attention-pooling heads -- structurally different from a plain linear projection, kept fully trainable |
| Loss | **SigLIP sigmoid**, learnable `logit_scale` **and** `logit_bias` | The one model in the suite using the pairwise loss; matches pretraining |
| Differential LR | Same 4-tier scheme, `logit_scale`/`logit_bias` share the `logit_scale` bucket | |
| Precision | bf16 + FlashAttention-2 + grad checkpointing | `attn_implementation="flash_attention_2"` requested explicitly through `transformers` |
| Quantization | QLoRA available (`quantization: 4bit`) | Giant-opt tower is the largest single vision tower in the suite; QLoRA is the recommended fallback under VRAM pressure |

### 2.4 jinaai/jina-embeddings-v5-omni-small-retrieval

| Aspect | Value | Notes |
|---|---|---|
| Backend | `transformers.AutoModel` (`trust_remote_code=True`) | Custom modeling code, Qwen-family decoder backbone -- see the integration caveat in `mm_bench/models/jina_omni.py`'s module docstring |
| Adaptation | LoRA, `r=16, alpha=32`, **shallow** depth fraction 0.3 | Stacks a *new* task LoRA on top of Jina's own internal task-LoRA mechanism, rather than fighting it |
| Target modules | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | Standard Qwen/Llama-family decoder module names (attention + gated-MLP) |
| `modules_to_save` | `multi_modal_projector` | Vision-to-LM-space bridge, if present in the checkpoint |
| Loss | InfoNCE | Retrieval-task convention |
| Differential LR | `vision_tower.*` / `language_model.*` regex groups; text tower fully frozen outside LoRA (`unfreeze_last_n_text_blocks: 0`) since it's a shared decoder backbone with broad linguistic knowledge worth protecting | |
| Precision | bf16 + FlashAttention-2 + grad checkpointing | Long text context (up to 512 tokens) makes FlashAttention-2 the highest-leverage optimization for this model specifically |
| Quantization | QLoRA available (`quantization: 4bit`) | |

---

## 3. Two-phase benchmarking: what to watch for in the delta

`results_comparison.csv`/`.md` reports `delta = v2_finetuned - v1_baseline`
per `(model, direction, metric)`. Interpretation notes specific to this
setup:

- **A large positive delta on `recall@1`/`recall@5` with a *negative* delta
  on `mean_rank`'s absolute scale across *both* directions** is the clean
  success signal -- fine-tuning improved in-domain retrieval symmetrically.
- **An asymmetric delta** (e.g. `image_to_text` improves a lot,
  `text_to_image` barely moves or regresses) usually indicates the caption
  distribution is narrower/more repetitive than the image distribution
  (a known risk with auto-generated captions -- see the captioning prompt
  used to build this dataset), which can make text embeddings collapse
  toward a smaller effective cluster count. If seen, prefer inspecting
  caption diversity per subset before changing any hyperparameter here.
- **SigLIP2's delta is not directly comparable in raw loss terms** to the
  three InfoNCE models (see 1.4) -- always compare on Recall@K/MRR/MeanRank,
  never on the raw training loss curve, when judging SigLIP2 against the
  others.
- Because the benchmark split is stratified 5% *per subset* (`L21`..`L30`),
  a per-subset breakdown of the same metrics (not currently emitted by
  `evaluation.compare` beyond the aggregate) is the natural next diagnostic
  if the aggregate delta looks surprising -- `Sample.subset` is preserved
  through the whole pipeline specifically so this slice is always available
  to add without re-deriving the split.
