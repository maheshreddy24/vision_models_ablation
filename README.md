# DINOv2-Registers Patch-Budget Ablation

How many patches does a frozen **DINOv2 ViT-S/14 (with registers)** actually
need? We drop random patch tokens before the transformer blocks (CLS +
registers always kept) and sweep the kept fraction 100% → 5%, measuring:

- **Depth** (NYU Depth v2 labeled subset, 795/654 split): linear log-depth
  head on patch tokens. Metrics: RMSE, AbsRel, δ<1.25 — on **visible-patch
  pixels only** by default (`eval_mode: full_image` switches to inpainting mode).
- **Classification** (Oxford-IIIT Pets, HuggingFace): linear probe on CLS. Top-1.

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
pytest tests/ -v                                # REQUIRED before sweeping:
                                                # verifies keep_ratio=1.0 == stock model
```

Everything else (budgets, seeds, epochs, resolutions, `protocol:
per_budget|reuse_full_head`) lives in `config.yaml`.

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
