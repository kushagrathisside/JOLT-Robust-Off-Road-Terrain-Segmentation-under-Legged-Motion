# JOLT — Robust Off-Road Terrain Segmentation under Legged Motion

> **Repo-ready research doc.** Copy into the forked ViT repo root as `RESEARCH.md`.
> *Defaults chosen (change freely):* project name `JOLT` (rename as you like); scope =
> **benchmark + method** (the method materializes only if the diagnostic gate passes).
> Self-contained — no external-project references.

A **parameterized motion-degradation robustness benchmark** for off-road terrain
segmentation (motion blur, rolling shutter, viewpoint jitter induced by legged ego-motion),
plus — gated on a diagnostic — a **robustness method**. We deliberately do **not** chase
clean-mIoU SOTA.

---

## 1. Motivation — why robustness, not accuracy

Two problems make in-domain mIoU the wrong target for off-road terrain segmentation:

1. **Noisy ground truth.** Off-road labels (e.g. RUGD) are coarse/blocky; a strong model's
   prediction is often *more* spatially faithful than the label, so **mIoU penalizes a
   more-correct model.**
2. **Foundation-model saturation.** SAM2 / DINOv2 / open-vocab segmenters are near-ceiling;
   there is little novelty headroom and no reliable signal in chasing clean mIoU.

**Reframe:** measure **robustness to legged-motion corruption**, not absolute accuracy.
Robustness is a **relative** quantity (model vs itself, clean→corrupted), so **label noise
cancels** — it is fair even with coarse GT. This is the methodological lineage of corruption
benchmarks (ImageNet-C), applied to a domain (legged off-road perception) where the
corruption is real and unowned.

## 2. Whitespace / contribution

Legged ego-motion (blur, rolling shutter, viewpoint jitter) is *acknowledged as a nuisance*
in off-road robotics but **no benchmark or method owns it**. Contributions:

- **C1 (benchmark):** a parameterized, gait-matched motion-degradation suite + a robustness
  metric, applied to off-road terrain segmentation across models and datasets.
- **C2 (method, gated):** a robustness method (see §6) — *only if* the gate (§4) shows SOTA
  actually degrades.

> If the gate fails (SOTA already robust), the contribution narrows to a **cross-dataset
> generalization** study (§7) + an honest negative result; do not force C2.

---

## 3. Status
Forked ViT/DINOv2 + RUGD baseline (exploratory, mIoU ≈ 0.46). Reframing toward C1/C2.
**Nothing downstream is built until the gate (§4) passes.**

## 4. ⛔ THE GATE — run this first

The whole project rests on one empirical claim: **legged-motion corruption degrades SOTA
terrain segmentation.** Test decisively (≈ a weekend):

1. Take RUGD (+ RELLIS) val sets (clean).
2. Apply the parameterized corruptions (§5.1) at severities 1–5.
3. Evaluate **DINOv2-frozen+head** (your pipeline) and **SAM2** (the ceiling) on clean vs
   corrupted; report **retention** (§5.2) per corruption × severity.
4. **Decision:**
   - **Large, monotone drop** (retention falls clearly with severity) → real problem →
     build C1 + C2. ✅
   - **Small/flat drop** → SOTA already robust → pivot to cross-dataset-only (§7). ✗

> Build nothing else until this passes. It is the cheapest decisive experiment here.

---

## 5. Benchmark specification

### 5.1 Corruptions (gait-matched, parameterized)
Severity ladder 1–5 (ImageNet-C style). Starting values — **calibrate to real legged
IMU/gait data if available** (a refinement, not a blocker):

| Corruption | Model | Severity 1 → 5 (parameter) |
|---|---|---|
| **Motion blur** | directional linear kernel, random orientation per image | kernel length L = {3, 7, 11, 15, 21} px |
| **Rolling shutter** | row-wise time-shift warp (row sampled at t + row·readout/H) + horizontal motion | max bottom-row displacement = {2, 4, 8, 16, 32} px |
| **Viewpoint jitter** | random homography from small pitch/roll/yaw | rotation σ = {0.5, 1, 2, 4, 8}° |
| *(combined)* | apply all three together at matched severity | the "legged" composite |

Gait link: severity ≈ faster gait / longer exposure / rougher terrain. Document the
mapping; if you have IMU traces, fit kernel length to measured angular velocity × exposure.

### 5.2 Metrics (robust to noisy GT)
- **Clean mIoU** `m0` (reference only).
- **Corrupt mIoU** `m(c,s)` per corruption `c`, severity `s`.
- **Retention** `R(c,s) = m(c,s) / m0` ∈ [0,1] — *the* robustness metric (GT noise cancels).
- **mCR** (mean corruption retention) = mean over `c,s` of `R` — single headline number.
- **Boundary-F1 retention** — boundaries degrade first under blur; report alongside mIoU.
- Report the per-corruption breakdown **and** mCR; severities plotted as retention curves.

### 5.3 Baselines (what to benchmark)
- **DINOv2 (frozen ViT-S/B) + light decoder** — your pipeline (the subject).
- **SAM2** (zero-shot / promptable) — the ceiling.
- **SegFormer / DeepLabv3** (supervised CNN/transformer) — conventional reference.
- *(optional)* an **open-vocab** segmenter (CLIP-based) — generalization reference.

### 5.4 Datasets & splits
- **Primary:** RUGD (val) — synthesize corruptions on clean images. *(No jerky data needed —
  synthesizing the corruption IS the contribution.)*
- **Cross-dataset (§7):** RELLIS-3D (standard companion), GOOSE / GOOSE-legged (closest to
  legged ground-level).

### 5.5 Protocol
- Fixed seeds for corruption RNG; report mean over 3 seeds.
- Same input resolution across models; document inference resolution.
- Release the corruption transforms + eval scripts so results are reproducible.

---

## 6. Method (only if the gate passes)
Pick after inspecting failure modes:
- **Proprioception/IMU-conditioned deblurring or stabilization** (most distinctive — legged
  robots have rich ego-motion signals that predict the blur; underused).
- **Temporal fusion / test-time augmentation** across adjacent frames.
- **Corruption-aware fine-tuning / distillation** (train the student under the parameterized
  corruptions; keep a frozen foundation backbone + small robustifying head → edge-friendly).

## 7. Cross-dataset generalization (secondary axis / fallback)
RUGD ↔ RELLIS ↔ GOOSE zero-shot transfer (the standard fairness test). This is the fallback
contribution if the gate fails, and a complementary axis if it passes.

## 8. Non-goals / honest scope
- **Not** a clean-mIoU SOTA claim.
- **Edge/efficiency** is at most a deployment note — efficient terrain segmentation is
  already well covered; do **not** claim it as novel, and do not justify it via jerk (the two
  are orthogonal).
- Corruption parameters are *plausible* starting values, not validated against a specific
  robot until calibrated to IMU/gait data.

---

## 9. Repo structure (proposed)

```
jolt/
  corruptions/      motion_blur.py · rolling_shutter.py · jitter.py · severity.py · compose.py
  models/           dinov2_head.py · sam2_wrapper.py · baselines.py
  data/             rugd.py · rellis.py · goose.py (loaders + splits)
  metrics/          retention.py (R, mCR) · boundary_f.py · miou.py
  scripts/
    gate.py         # §4 diagnostic: DINOv2 & SAM2, clean vs corrupted, retention
    benchmark.py    # full model × corruption × severity sweep → tables/curves
    crossdataset.py # §7 zero-shot transfer
  configs/          corruption_severities.yaml · models.yaml · datasets.yaml
  README.md / RESEARCH.md
```

### Quickstart
```bash
pip install -r requirements.txt            # torch, timm, mujoco-free; + SAM2, DINOv2 weights
python scripts/gate.py --models dinov2,sam2 --datasets rugd,rellis   # RUN FIRST (§4)
# if the gate passes:
python scripts/benchmark.py --config configs/corruption_severities.yaml
python scripts/crossdataset.py --train rugd --test rellis,goose
```

---

## 10. Related work / positioning
- **Corruption-robustness lineage:** ImageNet-C (Hendrycks & Dietterich, 2019) — parameterized
  corruption + relative robustness metric (the methodological precedent).
- **Off-road perception noting motion blur/rolling shutter as a nuisance (unowned):** TAIL
  terrain-SLAM dataset; Verti-Bench off-road mobility.
- **Off-road segmentation baselines/efficiency:** GANav (efficient terrain segmentation);
  Nano-U (distill-to-edge — why edge is *not* our novelty).
- **Datasets:** RUGD; RELLIS-3D; GOOSE / GOOSE-Ex (legged subset).
- **Foundation models benchmarked:** DINOv2; SAM2.

### References (fill DOIs/URLs)
- Hendrycks & Dietterich, *Benchmarking Neural Network Robustness to Common Corruptions* (ICLR 2019).
- RUGD: Wigness et al. (IROS 2019).
- RELLIS-3D: https://arxiv.org/abs/2011.12954
- GOOSE: https://goose-dataset.de/
- GANav: https://arxiv.org/pdf/2103.04233
- TAIL: https://arxiv.org/abs/2403.16875 · Verti-Bench: https://arxiv.org/abs/2502.11426
- DINOv2: https://arxiv.org/abs/2304.07193 · SAM2 (Meta).
