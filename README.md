# JOLT — Robust Off-Road Terrain Segmentation under Legged Motion

A **parameterized motion-degradation robustness benchmark** for off-road terrain
segmentation — motion blur, rolling shutter, and viewpoint jitter induced by legged
ego-motion — plus a gated robustness method. We deliberately do **not** chase clean
mIoU; we measure **retention** under corruption, where coarse-label noise largely
cancels.

> **Why robustness, not accuracy?** Off-road labels (RUGD) are coarse, so clean mIoU
> penalizes models that are more spatially faithful than the ground truth, and
> foundation models (DINOv2/SAM2) are near-ceiling on clean accuracy. Robustness is a
> *relative* quantity (model vs. itself, clean→corrupted), so it is fair even with
> noisy GT. See [`terrain-perception-RESEARCH.md`](terrain-perception-RESEARCH.md)
> for the full plan and [`docs/literature-justification.tex`](docs/literature-justification.tex)
> for positioning vs. prior work.

## Status

| Component | State |
|---|---|
| **Gate (§4)** — go/no-go diagnostic | ✅ implemented, `jolt/scripts/gate.py` |
| Corruptions, metrics, RUGD loaders, DINOv2 model, smoke test | ✅ implemented + tested |
| `benchmark.py` (full sweep), `crossdataset.py` (§7), `baselines.py`, SAM2 wrapper body | ⏳ not yet implemented |

**The gate is the deciding experiment — run it before building anything downstream.**
If legged-motion corruption clearly degrades SOTA terrain segmentation, build the
benchmark + method; if not, pivot to a cross-dataset generalization study.

## Repository layout

```
jolt/
  corruptions/   motion_blur · rolling_shutter · jitter · geometry · severity · compose
  data/          rugd.py  (RUGDFrames = full-frame benchmark unit; RUGDRaw = patches for training)
  metrics/       miou (streaming confusion matrix) · retention (R, mCR) · boundary_f
  models/        dinov2_head (subject) · sam2_wrapper (ceiling ablation, stub)
  scripts/       gate.py  ·  smoke_test.py
  README.md      package-level docs + methodological caveats
configs/         corruption_severities.yaml
docs/            literature-justification.tex  ·  running-on-gpu.md
data/            datasets live here (git-ignored; see data/README.md)
legacy/          original exploratory DINOv2/RUGD fine-tuning notebooks
```

## Quickstart

```bash
pip install -e .          # or: pip install -r requirements.txt

# 1. Offline plumbing check (no GPU / no weights needed):
python -m jolt.scripts.smoke_test \
    --images-dir data/rugd_sample/RUGD_sample-data/images \
    --annotations-dir data/rugd_sample/RUGD_sample-data/annotations

# 2. The gate (needs DINOv2 weights; GPU recommended):
python -m jolt.scripts.gate \
    --images-dir data/RUGD/images --annotations-dir data/RUGD/annotations \
    --split val --model dinov2 --checkpoint best_model.pth --seeds 3 --limit 300
```

Full GPU run commands (RTX 4060 / 8 GB settings, SAM2 ablation, monitoring):
**[`docs/running-on-gpu.md`](docs/running-on-gpu.md)**.

## The metric

- `m0` — clean mIoU (reference only).
- `R(c,s) = m(c,s) / m0` ∈ [0,1] — **retention**, the headline robustness metric.
- `mCR` — mean retention over corruptions × severities.
- Boundary-F retention reported alongside (boundaries degrade first under blur).

Corruptions are synthesized on the **full-resolution frame** (global ego-motion is a
full-sensor effect), then resized to a ÷14 inference size for the model. The gate
reports mean ± std over multiple corruption seeds; **frames** (not crops) are the
i.i.d. aggregation unit. See [`jolt/README.md`](jolt/README.md) for the methodological
caveats (retention cancels label noise only approximately; report photometric vs.
geometric retention separately; severity values are not yet IMU-calibrated).

## License
MIT — see [`LICENSE`](LICENSE).
