# DINOv2-Registers Patch-Budget Ablation Study

## 1. Goal

Quantify how robust a **frozen DINOv2 ViT-S/14 with registers** encoder is when it receives only a fraction of the image's patch tokens. We progressively reduce the percentage of visible patches (100% → 5%) and measure downstream performance on:

- **Semantic (dense) task:** monocular depth estimation on NYU Depth v2 (labeled subset).
- **Classification task:** Oxford-IIIT Pets (linear probe on the CLS token).

Final deliverable: a results CSV + a single plot (x = % patches kept, y = performance, one curve per task) showing the minimum patch budget at which the model still performs meaningfully.

**Why the registers variant:** register tokens absorb the global/artifact information that vanilla DINOv2 stores in redundant patch tokens, so patch tokens in the `_reg` model are cleaner spatial features. This makes the ablation a fairer test of how much *spatial* evidence the model needs.

---

## 2. Non-negotiable engineering rules

1. **Modular, no redundancy.** One responsibility per module. Shared logic (feature extraction, masking, metrics) lives in exactly one place. No copy-pasted training loops between tasks.
2. **Config-driven.** A single `config.yaml` controls everything (budgets, seeds, dataset paths, head hyperparams). No magic numbers inside code.
3. **Readable for reviewers.** Type hints, short docstrings, clear naming. This repo will be shared with the team — structure > cleverness.
4. **Reproducible.** All randomness (masking, head init, dataloader shuffling) seeded from config. Every result row in the CSV records its seed.
5. **Frozen backbone always.** The encoder is never trained or fine-tuned. Only the tiny task heads train.
6. Fail loudly: assert tensor shapes at module boundaries (token counts, feature dims).

---

## 3. Model & patch-dropping mechanism

### 3.1 Backbone
```python
import torch
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
model.eval().requires_grad_(False)
```
- ViT-S/14 with 4 register tokens. Embed dim = 384, patch size = 14.
- Input resolution: **224×224** for classification (16×16 = 256 patches), **420×420** (or nearest multiple of 14 to NYU aspect) for depth so the dense head has enough spatial resolution. Keep resolution per-task in config.

### 3.2 Patch dropping (the core intervention)
Implement a wrapper `MaskedDinoV2(nn.Module)` that reproduces the DINOv2 forward but drops patch tokens **after** patch embedding + positional-encoding interpolation and **before** the transformer blocks:

1. Compute patch tokens + interpolated pos-embed as usual.
2. Randomly sample `k = round(keep_ratio * N_patches)` patch indices (uniform, no replacement) using a generator seeded with `mask_seed`.
3. Keep: CLS token + all 4 register tokens + the k sampled patch tokens. Discard the rest entirely (do **not** replace with mask tokens — ViTs handle variable-length sequences natively).
4. Return: CLS feature, register features, patch features, and the kept-index tensor (needed to scatter dense predictions back onto the image grid).

Prefer implementing this by subclassing / monkey-patching `prepare_tokens_with_masks` + forward rather than re-implementing attention. Verify with an assertion that `keep_ratio=1.0` reproduces the stock model's outputs to within float tolerance (this is a required unit test).

---

## 4. Datasets

### 4.1 NYU Depth v2 — labeled subset (depth)
- Download `nyu_depth_v2_labeled.mat` (~2.8 GB) and `splits.mat` from the official NYU site (`http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat` and the splits file linked on the same page). Script: `scripts/download_nyu.py` → saves to `data/nyu/`, verifies file size/hash, skips if present.
- 1449 aligned RGB (480×640) + depth pairs; use the official split: **795 train / 654 test**.
- Load with `h5py`. Depth in meters; clip to [0.1, 10.0] and mask invalid pixels.
- Standard eval crop (Eigen crop) for metrics.
- (The `.mat` also contains semantic labels — leave a stub `tasks/segmentation.py` with a TODO so seg can be added later without new downloads.)

### 4.2 Oxford-IIIT Pets (classification)
- HuggingFace: `load_dataset("timm/oxford-iiit-pet")` (fallback id if unavailable: `pcuenq/oxford-pets`). 37 classes, ~3.7k train / ~3.7k test.
- Script: `scripts/download_pets.py` caches to `data/pets/`.

---

## 5. Tasks & heads (frozen features only)

### 5.1 Depth estimation
- **Head:** a small linear/DPT-lite head — default: `Linear(384 → 1)` applied per patch token, predictions scattered back to their grid positions using kept indices, then bilinearly upsampled to image resolution. Keep the head class swappable via config.
- **Loss:** scale-invariant log loss (SILog) on **visible-patch pixels only**.
- **Metrics:** RMSE, AbsRel, δ<1.25 — computed in two modes:
  - `visible_only` (default, the honest measurement): metrics only over pixels belonging to kept patches.
  - `full_image` (optional flag): head also predicts dropped-patch positions via learned mask tokens inserted at the head input — measures true "fill-in" ability. Implement but keep off by default.
- **Training:** AdamW, lr 1e-3, cosine schedule, ~20 epochs. Head is tiny; this is minutes per run.

### 5.2 Classification
- **Head:** linear probe on CLS token (`Linear(384 → 37)`).
- **Loss:** cross-entropy. **Metric:** top-1 accuracy.
- **Training:** AdamW, lr 1e-3, ~15 epochs, light augmentation (flip only — keep probing clean).
- Note: CLS is always kept, so degradation here isolates how much the global token depends on patch evidence.

### 5.3 Protocol
Default protocol: **train the head separately at each patch budget** (head sees masked features during training and eval — measures "best achievable at this budget"). Also support `--reuse-full-head`: train once at 100%, evaluate under all budgets (measures representation robustness without adaptation). Both write to the same CSV with a `protocol` column.

**Speed tip:** since the backbone is frozen, precompute and cache features per `(dataset, budget, mask_seed)` to `data/cache/` before head training. Masks are fixed per image per (budget, seed) — this is intentional and fine.

---

## 6. Ablation sweep

- Patch budgets: `[1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]`.
- Mask seeds per budget: `[0, 1, 2]` → report mean ± std.
- Total runs: 7 budgets × 3 seeds × 2 tasks = 42 head trainings (each small).
- Runner: `python -m ablation.run --config config.yaml` executes the full grid, resumes from CSV if interrupted (skip rows already present).

### Results CSV schema
`results/results.csv`:
```
task, protocol, keep_ratio, mask_seed, metric_name, value, n_epochs, timestamp
```

---

## 7. Plot

`python -m ablation.plot --csv results/results.csv` →
- Single matplotlib figure, x-axis = % patches kept (descending left→right optional flag), twin y-axes:
  - Left y: depth δ<1.25 (higher = better) — solid line.
  - Right y: Pets top-1 accuracy — dashed line.
- Shaded band = ±1 std over mask seeds. Horizontal dotted reference lines at each task's 100% performance.
- Save `results/patch_budget_ablation.png` (300 dpi) + a secondary figure with RMSE/AbsRel.

---

## 8. Repository structure

```
dino_patch_ablation/
├── config.yaml
├── README.md                  # how to run everything in 5 commands
├── scripts/
│   ├── download_nyu.py
│   └── download_pets.py
├── src/
│   ├── models/
│   │   └── masked_dinov2.py   # MaskedDinoV2 wrapper + kept-index logic
│   ├── data/
│   │   ├── nyu.py             # Dataset class, transforms, eigen crop
│   │   └── pets.py
│   ├── tasks/
│   │   ├── base.py            # shared head-training loop (one loop for all tasks)
│   │   ├── depth.py           # head, SILog, depth metrics
│   │   ├── classification.py  # linear probe, top-1
│   │   └── segmentation.py    # stub / TODO
│   ├── features.py            # feature extraction + caching
│   └── utils.py               # seeding, csv logging, asserts
├── ablation/
│   ├── run.py                 # sweep runner (resumable)
│   └── plot.py
├── tests/
│   ├── test_masking.py        # keep_ratio=1.0 == stock model; token counts
│   └── test_scatter.py        # dense predictions land on correct grid cells
├── data/                      # gitignored
└── results/
```

---

## 9. config.yaml (starting values)

```yaml
seed: 42
device: cuda
backbone: dinov2_vits14_reg

budgets: [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]
mask_seeds: [0, 1, 2]
protocol: per_budget        # per_budget | reuse_full_head

depth:
  resolution: [420, 560]    # multiples of 14, ~NYU aspect
  epochs: 20
  lr: 1.0e-3
  batch_size: 16
  eval_mode: visible_only   # visible_only | full_image
  depth_clip: [0.1, 10.0]

classification:
  dataset: timm/oxford-iiit-pet
  resolution: 224
  epochs: 15
  lr: 1.0e-3
  batch_size: 256

cache_features: true
paths:
  data: data/
  cache: data/cache/
  results: results/
```

---

## 10. Order of work (do these in sequence, verify each)

1. Scaffold repo + config + utils (seeding, CSV logger).
2. `MaskedDinoV2` + `tests/test_masking.py` — **do not proceed until the keep_ratio=1.0 equivalence test passes.**
3. Download scripts + dataset classes; visualize 3 samples per dataset as a sanity check (save to `results/sanity/`).
4. Feature extraction + caching.
5. Classification task end-to-end at 100% budget → sanity target: linear probe on Pets should reach roughly ≥90% top-1; if far below, debug before sweeping.
6. Depth task end-to-end at 100% budget → sanity target: δ<1.25 in a reasonable linear-probe range (~0.75–0.85); if wildly off, check depth units/crop.
7. Full sweep runner + resume logic.
8. Plotting.
9. README with exact commands.

## 11. Open questions to flag (don't block on these)

- Whether random masking should be block-wise (contiguous regions) as an additional condition — leave a `mask_strategy: random` config hook (`random | block`) with only `random` implemented.
- Whether registers themselves should be ablated (dropped) as a follow-up experiment — out of scope for now.
