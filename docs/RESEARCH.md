# JOLT — Research Plan

A **parameterized motion-degradation robustness benchmark** for outdoor scene
segmentation (motion blur, rolling shutter, and viewpoint jitter induced by legged
ego-motion), plus — gated on a diagnostic — a **robustness method**. We deliberately
do **not** chase clean-mIoU state of the art.

This is a plan, not a results report. Claims marked *(unverified)* have not been
checked against a source or a run. Section numbers (§4, §5.1, …) are the ones cited in
the code, so keep them stable when editing.

---

## 1. Motivation — why robustness, not accuracy

Two problems make in-domain mIoU a poor target for outdoor segmentation on RUGD:

1. **Noisy ground truth.** RUGD labels are coarse. In an exploratory run, the model's
   prediction was more spatially faithful than the label in places (grass intruding
   into gravel, which the label flattens), so mIoU penalizes a more-correct prediction.
   This is one observation, and the figure it came from is not in the repository.
   *(unverified as a general claim)*
2. **Foundation-model saturation.** The working hypothesis is that SAM2, DINOv2 and
   open-vocabulary segmenters are near ceiling on clean accuracy, leaving little
   headroom. *(hypothesis; not measured here)*

**Reframe:** measure robustness to legged-motion corruption, not absolute accuracy.
Robustness is a relative quantity (a model against itself, clean vs. corrupted), so
label noise largely cancels. This follows the corruption-benchmark lineage of
ImageNet-C [4].

## 2. Positioning and contributions

- **C1 (benchmark):** a parameterized, gait-matched motion-degradation suite and a
  retention metric, applied to outdoor segmentation across models and datasets.
- **C2 (method, gated):** a robustness method (§6), built only if the gate (§4) shows
  that state-of-the-art segmentation degrades.

**Scope of the novelty claim.** Corruption-robustness benchmarks for segmentation
exist (Kamann & Rother [11]), and a published method already synthesizes motion blur
from segmentation labels for road scenes (CCMBA [12]). What survives the check in §2.1
is narrower: a gait-matched, retention-based benchmark for outdoor scenes under legged
corruption. See `docs/literature-justification.tex`, section "Threats to novelty", for
the detailed positioning.

An earlier draft said that no work addresses legged ego-motion. That was wrong: legged
gait blur has been studied for feature detection (2019) and for occupancy prediction
(2025–26). Its examples were also wrong. TAIL is a SLAM dataset and Verti-Bench is a
simulation mobility benchmark, and neither was confirmed to treat blur or rolling
shutter as a nuisance.

If the gate fails, the contribution narrows to cross-dataset generalization (§7) and an
honest negative result. Do not force C2.

### 2.1 Literature check (2026-10-16)

A small review, done with web search. It used abstracts and search snippets, not full
texts, so it is not a systematic review. Numbers in brackets refer to §11.

- **Segmentation under corruption.** Kamann & Rother [11] corrupted about 400,000 images
  from Cityscapes, PASCAL VOC and ADE20K. Networks largely tolerate optical (PSF) blur,
  and blur hurts small, distant objects most. The search did **not** confirm that the
  benchmark includes *motion* blur. The literature doc says it does, so check the PDF.
- **Motion blur for segmentation (on road).** CCMBA [12] (CVPR 2023) builds
  class-centric motion blur from segmentation labels. It covers camera ego-motion blur
  and dynamic-object blur, is evaluated on VOC and Cityscapes, and has released code.
  This is the closest prior art to C1's motion-blur part.
- **Foundation models under blur.** For SAM, the two studies disagree. One finds blur,
  including motion blur, is a weak spot for small objects [13]. The other finds SAM
  tolerates most blur types, with zoom blur as the exception [14]. Both test the original
  SAM, not SAM2. For DINOv2, feature-level studies find that mild blur changes features
  little, and larger blur changes them much more [15]. The ECCV 2024 BRAVO challenge was
  won with DINOv2 and a simple decoder on out-of-distribution urban scenes [16]. No
  segmentation-under-blur study of DINOv2 was found.
- **Off-road segmentation.** No 2025 off-road blur or corruption benchmark was found.
  COARSE [17] (IROS 2025) trains on coarse real labels plus densely labelled
  out-of-domain data. It reports gains of 9.7% on RUGD and 8.4% on RELLIS-3D over
  coarse-only training. That supports the coarse-label premise.
- **Legged-robot blur and jitter.** Gültekin & Saranlı [18] (2019) benchmark deblurring
  for legged-robot vision, using a feature-detection metric. PanoMMOcc [19] (2026) and
  OneOcc [20] (2025) study gait-induced vertical jitter and blur in panoramic occupancy
  prediction, and PanoMMOcc adds a jitter-compensation module. No segmentation-robustness
  benchmark under legged corruption was found.
- **IMU-aided deblurring.** Gyroscope-aided deblurring exists [21, 22]. One 2024 method
  derives blur paths from IMU data to make training pairs.
- **Rolling shutter.** No segmentation study was found. A 2026 UF paper handles monocular
  depth under strong rolling-shutter distortion from ego-motion and reports a test on a
  legged robot [23]. It is about depth, not segmentation.
- **Gait frequency (optional sweep).** For a compliant quadruped, the best forward trot
  was at 3.5 Hz locomotion frequency, with centre-of-mass height changing by ±5 mm and
  pitch by 0.06 ± 0.04 rad [24]. A horse-like quadruped study reports a speed-optimal trot
  near 4 Hz and walk near 3.4 Hz [25], per the search snippet. These are quadruped
  locomotion frequencies, not humanoid body-oscillation frequencies. No humanoid source
  was found.

**Implications for the plan.**
1. The motion-blur part of C1 is not novel and must cite CCMBA [12].
2. The novelty claim should be a gait-matched, retention-based, outdoor-scene benchmark.
   The whitespace is narrower than the earlier draft said.
3. The gate prior is unclear. For DINOv2, low severities may come out flat. For SAM2,
   the literature gives no prediction.
4. The IMU-conditioned method idea (§6) and jitter compensation for legged perception
   already exist. The novelty would have to come from coupling them to segmentation
   robustness on outdoor scenes.

If the gate fails, the contribution narrows to cross-dataset generalization (§7) and an
honest negative result. Do not force C2.

## 3. Status

- Exploratory DINOv2 (ViT-S/14, frozen backbone) with a small conv head on RUGD. The
  head was trained on **224×224 patches** in a Colab notebook kept on the
  `legacy-notebooks` branch. Training settings are in
  [`running-on-gpu.md`](running-on-gpu.md) §1.
- The exploratory mIoU of about 0.46 was computed with the notebook's metric (per-batch
  averages, void class included). It is **not comparable** to the gate's metric
  (dataset-level confusion matrix, void ignored).
- The head was trained at 224×224, but the gate infers on full frames at 448×560. The
  effect of this scale change has not been measured.
- Nothing downstream is built until the gate (§4) passes.

## 4. The gate — run first

The whole project rests on one empirical claim: legged-motion corruption degrades
state-of-the-art outdoor segmentation. Test it on the RUGD val split:

1. Take the clean RUGD val frames.
2. Apply the corruptions (§5.1) at severities 1–5, with 3 corruption seeds.
3. Evaluate DINOv2 (frozen + head, the subject) and SAM2 (the ceiling; its wrapper is
   not yet implemented) on clean vs. corrupted frames. Report retention (§5.2) per
   corruption and severity.
4. Decide:
   - **Large, monotone drop** in retention with severity → real problem → build C1 + C2.
   - **Small or flat drop** → state-of-the-art already robust → pivot to cross-dataset
     generalization (§7).

The verdict printed by `jolt/scripts/gate.py` uses cut-offs set in the code: PASS if
mean retention at the highest severity is ≤ 0.85 and monotone, BORDERLINE if ≤ 0.92,
otherwise FAIL. These cut-offs are heuristics. They are not derived here or validated.
Read the per-corruption curves and the seed spread before deciding.

Prior from the literature (§2.1): DINOv2 may stay flat at low severities, because mild
blur changes its features little. The verdict's "monotone" check should be read with that
in mind. SAM2 has no clear prediction, and the original-SAM studies disagree.

> Build nothing else until the gate passes. It is the cheapest decisive experiment in
> the plan.

## 5. Benchmark specification

### 5.1 Corruptions (gait-matched, parameterized)

Severity ladder 1–5 (ImageNet-C style). These are starting values, to be calibrated
against real legged IMU and gait data when available. The values live in
`jolt/corruptions/severity.py` and `configs/corruption_severities.yaml`.

| Corruption | Model | Severity 1 → 5 (parameter) |
|---|---|---|
| **Motion blur** | directional linear kernel, random orientation per image | kernel length L = {3, 7, 11, 15, 21} px |
| **Rolling shutter** | row-wise time-shift warp (row sampled at t + row·readout/H) + horizontal motion | max bottom-row displacement = {2, 4, 8, 16, 32} px |
| **Viewpoint jitter** | random homography from small pitch/roll/yaw | rotation σ = {0.5, 1, 2, 4, 8}° |
| *(legged composite)* | jitter → rolling shutter → motion blur at one matched severity | combines the three; not a fourth degradation |

Gait link: severity ≈ faster gait / longer exposure / rougher terrain. Document the
mapping. If IMU traces are available, fit the kernel length to measured angular
velocity × exposure time. The gait-frequency sources found so far are for quadrupeds
(§2.1 [24, 25]). The optional humanoid sweep needs a humanoid source.

### 5.2 Metrics (robust to noisy ground truth)

- **Clean mIoU** `m0` (reference only).
- **Corrupt mIoU** `m(c,s)` per corruption `c` and severity `s`.
- **Retention** `R(c,s) = m(c,s) / m0` ∈ [0,1]. This is the robustness metric, and label
  noise cancels in the ratio (approximately, since mIoU is nonlinear).
- **mCR** (mean corruption retention): the mean of `R` over `c, s`. The headline number.
- **Boundary-F retention**, reported alongside. Boundaries degrade first under blur.
- Report the per-corruption breakdown and mCR, and plot retention curves against
  severity.

Geometric corruptions warp the mask with nearest-neighbour resampling, which adds
resampling artifacts to the ground truth. Report photometric and geometric retention
separately. Motion-blur retention is the cleanest single number.

### 5.3 Baselines

- **DINOv2 (frozen ViT-S/14) + conv head**: the subject. Implemented.
- **SAM2**: the ceiling. The wrapper is a stub. The mask-to-label assignment it needs is
  not implemented, and that choice would affect the reported numbers.
- **SegFormer / DeepLabv3**: planned, not implemented. *(unverified; not checked)*
- **Open-vocabulary (CLIP-based) segmenter**: optional, planned. *(unverified; not
  checked)*

### 5.4 Datasets and splits

- **Primary: RUGD** [1], val split. The colour map in `jolt/data/rugd.py` encodes 25
  label ids (0–24, with 0 as void). The split lists `RUGD_SPLITS` are labelled as
  "commonly used" in the code and must be checked against the protocol before any
  result is reported. *(unverified)*
- **Cross-dataset (planned):** RELLIS-3D [2], a multimodal off-road dataset from
  Texas A&M's RELLIS Campus.
- **Cross-dataset (planned):** GOOSE / GOOSE-Ex [3]. GOOSE-Ex adds data from a legged
  platform (a Boston Dynamics Spot quadruped) and an excavator. Licence terms not
  checked. *(unverified)*

### 5.5 Protocol

- Fixed seeds for the corruption RNG. Report mean ± std over 3 seeds (the gate default).
- Same input resolution across models. Document the inference resolution.
- Release the corruption transforms and the evaluation scripts so results reproduce.

## 6. Method (only if the gate passes)

Pick a method after inspecting failure modes:

- **IMU- or proprioception-conditioned deblurring or stabilization.** Legged robots
  have rich ego-motion signals that could predict the blur. Gyroscope- and IMU-aided
  deblurring already exist for general cameras [21, 22], and jitter compensation for
  legged panoramic perception exists [19]. The novelty would be coupling them to
  segmentation robustness on outdoor scenes.
- **Temporal fusion or test-time augmentation** across adjacent frames.
- **Corruption-aware fine-tuning or distillation.** Train the student under the
  parameterized corruptions, and keep a frozen foundation backbone with a small
  robustifying head.

## 7. Cross-dataset generalization (secondary axis, and fallback)

Zero-shot transfer between RUGD, RELLIS-3D and GOOSE, which is the standard fairness
test. This is the fallback contribution if the gate fails, and a complementary axis if
it passes. Not implemented yet.

## 8. Non-goals and scope

- **Not** a clean-mIoU state-of-the-art claim.
- **Edge and efficiency** are at most a deployment note. Distilled terrain segmentation
  for tiny robots is already published (Nano-U [8], one 2026 preprint, not replicated),
  so do not claim edge efficiency as novel.
- Corruption parameters are plausible starting values, not validated against a robot
  until they are calibrated to IMU and gait data.

## 9. Repository layout

```
jolt/
  corruptions/   motion_blur · rolling_shutter · jitter · geometry · severity · compose
  data/          rugd.py  (RUGDFrames = full-frame benchmark unit; RUGDRaw = 224 patches, head training only)
  metrics/       miou (confusion matrix) · retention (R, mCR) · boundary_f
  models/        dinov2_head (subject) · sam2_wrapper (ceiling; stub)
  scripts/       gate.py · smoke_test.py
planned:         scripts/benchmark.py · scripts/crossdataset.py · models/baselines.py
                 data/rellis.py · data/goose.py
```

## 10. Quickstart

The commands are in the top-level [`README.md`](../README.md) and in
[`jolt/README.md`](../jolt/README.md). The GPU runs are in
[`running-on-gpu.md`](running-on-gpu.md).

## 11. References

"Checked" means the title, authors and venue or arXiv ID were confirmed by a web search
on the date of the commit that added this file. Full PDFs were not read. Items 11–27
were added on 2026-10-16 from search snippets and abstracts only.

1. N. Wigness, S. Eum, J. G. Rogers, D. Han, H. Kwon. *A RUGD Dataset for Autonomous
   Navigation and Visual Perception in Unstructured Outdoor Environments.* IROS 2019.
   DOI 10.1109/IROS40897.2019.8968283. *(checked)*
2. P. Jiang, P. Osteen, M. Wigness, S. Saripalli. *RELLIS-3D Dataset: Data, Benchmarks
   and Analysis.* arXiv:2011.12954. https://arxiv.org/abs/2011.12954 *(checked)*
3. Hagmanns et al. *Excavating in the Wild: The GOOSE-Ex Dataset for Semantic
   Segmentation.* arXiv:2409.18788 (2024). GOOSE project site: https://goose-dataset.de/
   *(checked; GOOSE site not opened)*
4. D. Hendrycks, T. Dietterich. *Benchmarking Neural Network Robustness to Common
   Corruptions and Perturbations.* ICLR 2019. arXiv:1903.12261 (ICLR version); the 2018
   preprint is arXiv:1807.01697. *(checked)*
5. M. Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision.*
   arXiv:2304.07193 (published in TMLR, 2024). *(checked)*
6. N. Ravi et al. (Meta FAIR). *SAM 2: Segment Anything in Images and Videos.*
   arXiv:2408.00714. *(checked)*
7. T. Guan et al. *GANav: Efficient Terrain Segmentation for Robot Navigation in
   Unstructured Outdoor Environments.* arXiv:2103.04233 (IEEE RA-L 2022, per the UMD
   project page). Segmentation for navigation, not a corruption benchmark. *(checked)*
8. F. Pizzolato, F. Pasti, N. Bellotto. *Nano-U: Efficient Terrain Segmentation for Tiny
   Robot Navigation.* arXiv:2605.10210 (May 2026). A single preprint; no replication
   found. *(checked; existence only)*
9. C. Yao et al. *TAIL: A Terrain-Aware Multi-Modal SLAM Dataset for Robot Locomotion in
   Deformable Granular Environments.* arXiv:2403.16875. A SLAM dataset with wheeled and
   legged data. Whether it treats motion blur as a nuisance was **not confirmed**.
   *(checked; existence only)*
10. T. Xu et al. *Verti-Bench: A General and Scalable Off-Road Mobility Benchmark for
    Vertically Challenging Terrain.* arXiv:2502.11426 (RSS 2025). A simulation mobility
    benchmark. Whether it treats blur or rolling shutter as a nuisance was **not
    confirmed**. *(checked; existence only)*

11. A. Kamann, C. Rother. *Benchmarking the Robustness of Semantic Segmentation Models
    with Respect to Common Corruptions.* International Journal of Computer Vision 129
    (2021), 462–483. DOI 10.1007/s11263-020-01383-2. *(snippet; motion-blur coverage not
    confirmed)*
12. Aakanksha, A. N. Rajagopalan. *Improving Robustness of Semantic Segmentation to
    Motion-Blur Using Class-Centric Augmentation.* CVPR 2023, pp. 10470–10479. Code:
    https://github.com/aka-discover/CCMBA_CVPR23 *(snippet)*
13. *On the Robustness of Segment Anything.* arXiv:2305.16220 (original SAM; blur named as
    a weakness). *(snippet)*
14. Qiao et al. *Robustness of SAM: Segment Anything Under Corruptions and Beyond.*
    arXiv:2306.07713 (original SAM). *(snippet)*
15. *Probing Perturbation Invariance in DINOv2: Mechanistic Gaps Between Real and
    Generated Image Representations.* ICML 2026. https://icml.cc/virtual/2026/79536
    *(snippet; feature-level, not segmentation)*
16. *First Place Solution to the ECCV 2024 BRAVO Challenge: Evaluating Robustness of Vision
    Foundation Models for Semantic Segmentation.* arXiv:2409.17208. *(snippet)*
17. A. Noca et al. *COARSE: Collaborative Pseudo-Labeling with Coarse Real Labels for
    Off-Road Semantic Segmentation.* arXiv:2503.03947; IROS 2025. *(snippet)*
18. Gültekin, Saranlı. *Feature detection performance based benchmarking of motion
    deblurring methods: applications to vision for legged robots.* Image and Vision
    Computing 82 (2019), 26–38. *(snippet)*
19. *Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots* (PanoMMOcc,
    VoxelHound). arXiv:2603.13108 (2026). *(snippet)*
20. *OneOcc: Semantic Occupancy Prediction for Legged Robots with a Single Panoramic
    Camera.* arXiv:2511.03571 (2025). *(snippet)*
21. Mustaniemi et al. *Gyroscope-Aided Motion Deblurring with Deep Networks.* WACV 2019.
    arXiv:1810.00986. *(snippet)*
22. Luan et al. *Gyroscope-Assisted Motion Deblurring Network.* arXiv:2402.06854 (2024).
    *(snippet)*
23. Rolling-shutter anchor-point method for monocular depth under ego-motion (2026).
    https://focus.ece.ufl.edu/research/2026/rolling-shutter-anchor-point/paper.pdf
    Authors not recorded. *(snippet)*
24. *Oncilla robot: a versatile open-source quadruped research robot with compliant
    pantograph legs.* arXiv:1803.06259. Authors not recorded. *(snippet)*
25. Moro et al. (2013). Horse-like motion primitives applied to a compliant quadruped.
    https://dlg.is.mpg.de/publications/moro_horse-like_2013. Journal and figures not
    verified. *(snippet only)*
26. L. Kong et al. *Robo3D: Towards Robust and Reliable 3D Perception against Corruptions.*
    ICCV 2023. arXiv:2303.17597. LiDAR, not camera. *(snippet)*
27. X. Yan et al. *Benchmarking the Robustness of LiDAR Semantic Segmentation Models.*
    arXiv:2301.00970 (SemanticKITTI-C). LiDAR, not camera. *(snippet)*

Not checked: SegFormer, DeepLabv3, and the other sources in
`docs/literature-justification.tex`. Check them before citing.
