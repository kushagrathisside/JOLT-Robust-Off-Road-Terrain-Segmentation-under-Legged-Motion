# JOLT — Robust Outdoor Segmentation under Legged Motion

A **parameterized motion-degradation robustness benchmark** for outdoor scene
segmentation — motion blur, rolling shutter, and viewpoint jitter induced by legged
ego-motion — plus a gated robustness method. We deliberately do **not** chase clean
mIoU; we measure **retention** under corruption, where coarse-label noise largely
cancels. The current code evaluates on RUGD, an off-road dataset; RELLIS-3D and GOOSE
are planned, not implemented.

> **Why robustness, not accuracy?** Off-road labels (RUGD) are coarse, so clean mIoU
> penalizes models that are more spatially faithful than the ground truth, and
> foundation models (DINOv2/SAM2) are near-ceiling on clean accuracy. Robustness is a
> *relative* quantity (model vs. itself, clean→corrupted), so it is fair even with
> noisy GT. See [`docs/RESEARCH.md`](docs/RESEARCH.md) for the full plan and
> [`docs/literature-justification.tex`](docs/literature-justification.tex) for
> positioning vs. prior work.

## Status

| Component | State | Next step |
|---|---|---|
| Gate (`jolt/scripts/gate.py`, RESEARCH §4) | implemented; not yet run on RUGD | Run on RUGD val ([`running-on-gpu.md`](docs/running-on-gpu.md) §3) |
| Smoke test (`jolt/scripts/smoke_test.py`) | implemented; needs torch and the RUGD sample | Document where the RUGD sample comes from |
| Corruptions: motion blur, rolling shutter, jitter, legged composite | implemented | Calibrate severities against IMU or gait data (RESEARCH §5.1) |
| Metrics: mIoU, retention, mCR, boundary-F | implemented | Add unit tests (none exist yet) |
| RUGD loaders (`RUGDFrames`, `RUGDRaw`) | implemented | Check `RUGD_SPLITS` against the protocol |
| DINOv2 frozen head (`jolt/models/dinov2_head.py`) | implemented; loader refuses an untrained head | Supply `best_model.pth` ([`running-on-gpu.md`](docs/running-on-gpu.md) §1) |
| Head training | no script in the repo; the notebook is on `legacy-notebooks` | Decide whether to document the notebook or add a script |
| SAM2 wrapper (`sam2_wrapper.py`) | stub; raises `NotImplementedError` | Implement the mask-to-label assignment before use |
| Baselines (SegFormer, DeepLabv3) | not started | Choose and implement; cite after checking |
| Benchmark sweep (`scripts/benchmark.py`) | not started | After the gate passes |
| Cross-dataset (`scripts/crossdataset.py`, RELLIS-3D and GOOSE loaders) | not started | After the gate passes |
| Fourth degradation | open | Define and justify; the legged composite is a combination, not a fourth |
| Gait-frequency sweep | optional; not started | Cite the source of each frequency value |
| Literature | positioning in `docs/literature-justification.tex`; some sources unchecked | Verify each source before citing |
| Report | not started | Write once results exist |

**The gate is the deciding experiment — run it before building anything downstream.**
If legged-motion corruption clearly degrades SOTA outdoor segmentation, build the
benchmark and method; if not, pivot to a cross-dataset generalization study.

## Repository layout

Section references (§) in code and docs point to [`docs/RESEARCH.md`](docs/RESEARCH.md).

```
.gitignore                        ignored paths: data, runs, checkpoints, caches
LICENSE                           MIT licence
README.md                         this file: status, layout, quickstart
pyproject.toml                    package metadata and dependencies (pip install -e .)
requirements.txt                  dependency floors and the pip-freeze note
configs/
  corruption_severities.yaml      severity ladders (mirrors corruptions/severity.py)
data/
  README.md                       where datasets go; RUGD layout; split notes
docs/
  RESEARCH.md                     research plan: gate, benchmark spec, method options, references
  literature-justification.tex    literature positioning and threats to novelty (LaTeX)
  running-on-gpu.md               GPU setup, head checkpoint source, gate commands
jolt/
  README.md                       package docs and methodological caveats
  __init__.py                     package marker
  corruptions/
    __init__.py                   package marker
    compose.py                    corruption registry, legged composite, seeded RNG
    geometry.py                   backward warp maps; composing warps; image and mask resampling
    jitter.py                     viewpoint jitter (small random rotational homography)
    motion_blur.py                directional motion blur (photometric; mask unchanged)
    rolling_shutter.py            row-wise readout shear (geometric)
    severity.py                   severity ladders 1–5 for each corruption
  data/
    __init__.py                   package marker
    rugd.py                       RUGD loaders (RUGDFrames for the benchmark, RUGDRaw for head training), colour map, splits
  metrics/
    __init__.py                   package marker
    boundary_f.py                 boundary F-score, streamed over a split
    miou.py                       streaming confusion-matrix mIoU
    retention.py                  retention R and mCR
  models/
    __init__.py                   package marker
    dinov2_head.py                frozen DINOv2 with a conv head (the subject) and checkpoint loader
    sam2_wrapper.py               SAM2 ceiling model; stub that raises NotImplementedError
  scripts/
    __init__.py                   package marker
    gate.py                       the gate: clean vs. corrupted retention and advisory verdict
    smoke_test.py                 offline checks on RUGD sample frames (needs torch)
```

## Quickstart

```bash
pip install -e .          # or: pip install -r requirements.txt

# 1. Offline plumbing check (no GPU, no weights; needs torch and the RUGD sample):
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
