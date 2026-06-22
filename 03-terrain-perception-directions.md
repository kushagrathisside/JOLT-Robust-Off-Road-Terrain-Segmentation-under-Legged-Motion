# Terrain Perception — Current Directions

> The Perception layer of the thesis. Status: weakest thread today (DINOv2/RUGD, mIoU ≈ 0.46, exploratory). This doc reframes it into a defensible, gradeable chapter and gives the exact next experiment.

## The two problems that killed the original framing

1. **Noisy ground truth.** In `rugdresult.png`, the prediction (448px) is **more spatially faithful** than the coarse blocky RUGD GT — green grass-into-gravel intrusions the GT flattens. So **mIoU penalizes a more-correct model**. Classic RUGD label noise.
2. **SOTA saturation.** SAM2 / DINOv2/v3 / open-vocab are near-ceiling. Beating them on mIoU is unwinnable, and there's no novelty headroom.

➡ **Conclusion: stop grading perception by mIoU. Change what you measure. Do not refactor-to-SOTA to "win mIoU."**

## Key scoping decision (locked)

**Perception is real-image and kept SEPARATE from Regolith sim.**
- Reason: real RGB (RUGD/RELLIS/GOOSE) ≠ rendered MuJoCo terrain (domain gap + taxonomy mismatch). Rendering a rich sim env for vision is also a hard job → skipped.
- Consequence: perception is graded on **its own terms** (robustness + cross-dataset generalization), **not** on locomotion success. (The "grade-on-locomotion" option is off the table because there's no sim-vision link.)
- The fusion idea (privileged Regolith semantics → critic, visual → actor) stays **future work**.

## The reframe: robustness, not accuracy

**Headline = a parameterized motion-degradation (jerk) robustness benchmark for off-road terrain segmentation.**

Why this works:
- **Whitespace**: legged ego-motion (motion blur, rolling shutter, viewpoint jitter) is *acknowledged as a nuisance* in the literature but **nobody owns a benchmark/method for it** (TAIL, Verti-Bench mention it in passing).
- **On-brand**: same DNA as Regolith (parameterized terrain) and FORGE (parameterized tasks) → you become the "parameterized **corruption**" person.
- **Solves the grading problem**: robustness is **Δ-degradation (model vs itself under corruption)**. The noisy GT **cancels out** in the before/after comparison — fair *even with RUGD's coarse labels*.
- **Doesn't fight SAM/DINO on clean mIoU** — it carves the robustness niche they don't occupy.

### Axes of the chapter
- **Primary**: parameterized jerk/blur/rolling-shutter robustness benchmark + a robustness method.
- **Secondary**: cross-dataset zero-shot (RUGD ↔ RELLIS ↔ GOOSE-legged) — the fairness/generalization axis.
- **Deployment note (not a contribution)**: edge-runnable (Nano-U already owns distill-to-edge; do not claim it as novel; do not justify edge via jerk — they're orthogonal).

## ⛔ THE GATE — run this first, before building anything

The whole chapter rests on one empirical claim: **legged jerk actually degrades SOTA terrain segmentation.** Foundation models trained at scale may already be robust. Test it decisively (a weekend):

1. Take RUGD (and RELLIS) val sets.
2. Apply **gait-matched synthetic corruption**:
   - **Motion blur**: directional/linear blur kernels whose length is tied to realistic legged gait frequency × exposure (sweep severity). Trot/gallop ≈ a few Hz body oscillation → pick kernel lengths matching plausible exposure smear.
   - **Rolling shutter**: row-wise time-shift warp (sweep readout time).
   - **Viewpoint jitter**: random small homographies / pitch-roll perturbations per frame (sweep amplitude).
3. Run **SAM2 / DINOv2(frozen+head)** on clean vs corrupted; report **Δ-mIoU** (and Δ-boundary-F) vs severity.
4. **Decision**:
   - **Large, monotone drop** → real problem. Build the benchmark + method. ✅
   - **Small/flat drop** → SOTA already robust → **kill angle (i)**, pivot (cross-dataset-only chapter, or fold perception into future work). ✗

> Build nothing downstream until the gate passes. This is the cheapest decisive experiment in the whole thesis.

## If the gate passes — method options (pick after seeing failure modes)

- **Proprioception/IMU-conditioned deblurring or stabilization** (legged robots have rich proprioception that predicts the ego-motion causing the blur — under-used signal).
- **Test-time augmentation / temporal fusion** across adjacent frames.
- **Blur-/corruption-aware fine-tuning or distillation** (train the student under the parameterized corruptions).
- **Frozen foundation features + a small robustifying head** (keeps it edge-friendly).

## Datasets (real-image, terrain — not pure driving)

| Dataset | Notes | Legged relevance |
|---|---|---|
| **RUGD** | Husky (wheeled), 24 classes, ground-level. What you have. | medium (smooth motion) |
| **[GOOSE / GOOSE-Ex](https://goose-dataset.de/)** | German off-road, RGB+LiDAR, **released a legged-robot + excavator subset** | **highest** |
| **[RELLIS-3D](https://ar5iv.labs.arxiv.org/html/2011.12954)** | off-road rural, RGB+LiDAR, UGV; standard RUGD companion | high |
| **Freiburg Forest** | forested trails, RGB+depth+NIR (small) | medium |
| **YCOR** (Yamaha-CMU Off-Road) | ground-level off-road | medium |
| **[GO: Great Outdoors](https://arxiv.org/abs/2501.19274)** (2025) | newer multimodal outdoor | medium |
| **TAS500 / WildScenes** | unstructured natural semantic (WildScenes 3D-leaning) | medium |
| *ORFD / TartanDrive / GOOSE-auto* | **driving-oriented** — use only for transfer | low |

- Reality: **most off-road semantic datasets are vehicle-collected**; truly legged ground-level labeled terrain data is rare. **GOOSE-legged** is the closest, **RELLIS** is the standard cross-dataset companion.
- For the **jerk benchmark you don't even need jerky data** — you *synthesize* the corruption on existing clean datasets (that's the point and the contribution).

## Don't refactor RUGD to chase SOTA

- With DINO/SAM saturated + noisy GT, in-domain mIoU is a dead end for both novelty and grading.
- Either (a) **jerk-robustness benchmark** (recommended, gated), or (b) if the gate fails, a **cross-dataset-only** evaluation chapter + a short "DINOv2 for terrain" Medium article.
- Edge stays a deployment note.

## Concrete next steps

1. ▶ **Run the gate** (gait-matched corruption × SAM2/DINOv2, Δ-mIoU). Decide go/no-go.
2. If go: lock the **parameterized corruption spec** (kernel-length↔gait-frequency mapping, rolling-shutter readout, jitter amplitude) → that spec *is* the benchmark.
3. Add **GOOSE-legged** + **RELLIS** for cross-dataset.
4. Pick a method after inspecting failure modes (proprioception-conditioned stabilization is the most distinctive).
5. Keep it **separate from Regolith**; note the asymmetric-actor-critic fusion as future work.
6. Optional: **"Fine-tuning DINOv2 for terrain"** Medium article from the existing RUGD work (salvages the exploratory notebook with honest framing).

## Open question for next session

For the gate: which foundation model to stress-test first — **SAM2** (promptable, strong zero-shot masks) or **DINOv2 frozen + linear/light head** (closer to what you already built)? Recommend testing **both**: DINOv2-head represents *your* pipeline; SAM2 represents the *ceiling*. If even SAM2 degrades under jerk, the contribution is strongest.
