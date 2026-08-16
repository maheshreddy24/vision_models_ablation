# DINOv2 Patch-Budget Ablation

How many patches does a frozen **DINOv2 ViT-S/14** actually need? We drop
random patch tokens before the transformer blocks (CLS + any register tokens
always kept) and sweep the kept fraction 100% → 5%, measuring:

- **Depth** (NYU Depth v2 labeled subset, 795/654 split): linear log-depth
  head on patch tokens. Metrics: RMSE, AbsRel, δ<1.25 — on **visible-patch
  pixels only** by default (`eval_mode: full_image` switches to inpainting mode).
- **Classification** (Oxford-IIIT Pets, HuggingFace): linear probe on CLS. Top-1.

`config.yaml`'s `backbone` selects the variant — `dinov2_vits14` (original,
no registers, the current default) or `dinov2_vits14_reg` (with 4 register
tokens). `MaskedDinoV2` detects `num_register_tokens` from the loaded model,
so both work with no code changes; only `results/` and the plot title differ.

## Setup

```bash
pip install -r requirements.txt
```

## Run everything (one command)

```bash
python main.py
```

That runs: dataset download (~2.8 GB NYU + Pets) → full sweep
(7 budgets × 3 mask seeds × 2 tasks, resumable) → plots.

Useful variants:

```bash
python main.py --stages sweep,plot              # data already downloaded
python main.py --tasks classification           # quick task first (cached CLS features)
python main.py --stages plot                    # re-plot from existing results.csv
python main.py --workers 6                      # run 6 sweep entries at once
pytest tests/ -v                                # REQUIRED before sweeping:
                                                # verifies keep_ratio=1.0 == stock model
```

Everything else (budgets, seeds, epochs, resolutions, `protocol:
per_budget|reuse_full_head`) lives in `config.yaml`.

### Running experiments in parallel

`--workers N` runs N sweep entries — (task, keep_ratio, mask_seed) combos —
concurrently in separate processes, each with its own copy of the frozen
backbone kept warm across every job it's assigned. Since the backbone never
trains, this is safe; the only cost is one backbone's worth of RAM per
worker plus per-process dataloader/head memory. Pick `N` based on cores and
RAM you're willing to spend — there's no cross-run dependency to worry about
under `protocol: per_budget` (the default). Under `reuse_full_head`, the
shared head is trained once up front (sequentially) and its weights are
handed to every worker before the parallel eval-only sweep starts.

## Outputs

- `results/results.csv` — one row per (task, protocol, keep_ratio, mask_seed, metric)
- `results/patch_budget_ablation.png` — main figure: % patches kept vs
  depth δ<1.25 (left axis) and Pets top-1 (right axis), ±1 std bands
- `results/depth_errors.png` — RMSE / AbsRel vs patch budget

## Notes / deviations

- Depth metrics are computed at model resolution (420×560) rather than the
  strict Eigen-crop protocol at 480×640 — fine for a *relative* ablation,
  not comparable to published NYU numbers.
- Masks are deterministic per (mask_seed, sample_id), independent of batch
  composition — verified by tests.
- `src/tasks/` is designed so segmentation can be added later; the NYU `.mat`
  already contains the labels.
- Sanity targets at 100% budget: Pets top-1 roughly ≥ 0.90, depth δ<1.25
  roughly 0.75–0.85. If far off, debug before trusting the sweep.
