# JOLT — legged-motion robustness benchmark for outdoor scene segmentation

A parameterized motion-degradation robustness benchmark (motion blur, rolling
shutter, viewpoint jitter) for outdoor scene segmentation, plus a gated
robustness method. We measure **retention** under corruption, not clean mIoU.
The current code evaluates on RUGD, an off-road dataset.
See [`../terrain-perception-RESEARCH.md`](../terrain-perception-RESEARCH.md) for
the full plan and [`../docs/literature-justification.tex`](../docs/literature-justification.tex)
for positioning.

## Status
- ✅ **Gate (§4)** — `jolt/scripts/gate.py` (the go/no-go diagnostic). **Run this first.**
- ✅ Corruptions, metrics, RUGD loaders, DINOv2 model, smoke test.
- ⏳ **Not yet implemented:** `benchmark.py` (full sweep), `crossdataset.py` (§7),
  `baselines.py` (SegFormer/DeepLab), the SAM2 wrapper body. Don't assume these exist.

## Quickstart
```bash
pip install -e .                       # or: pip install -r requirements.txt
# offline plumbing check (no GPU/weights):
python -m jolt.scripts.smoke_test \
    --images-dir data/rugd_sample/RUGD_sample-data/images \
    --annotations-dir data/rugd_sample/RUGD_sample-data/annotations
# the gate (needs DINOv2 weights; GPU recommended):
python -m jolt.scripts.gate \
    --images-dir RUGD/images --annotations-dir RUGD/annotations \
    --split val --model dinov2 --checkpoint best_model.pth --seeds 3 --limit 300
```

## Models
- **DINOv2 frozen + head (`--model dinov2`)** is the **primary subject** — it is the
  pipeline under study and the one closest to the corruption-robustness literature.
- **SAM2 (`--model sam2`)** is an **optional ceiling ABLATION**, not the headline.
  It is heavier and requires a class-agnostic-mask → semantic-label assignment
  (e.g. oracle majority-vote GT), which inflates its *absolute* clean mIoU; only its
  *retention* is comparable. The wrapper is currently a stub — implement the
  assignment before using it.

## Important methodological notes

### Corruptions are synthesized on the FULL frame, then resized for the model
Motion blur, rolling shutter, and viewpoint jitter are **global, full-sensor
ego-motion effects**. They are applied to the native-resolution RUGD frame
(`RUGDFrames`, 550×688), *then* the corrupted frame is resized to a multiple-of-14
inference size and the logits are upsampled back for scoring. Do **not** corrupt the
224×224 patches (`RUGDRaw`) — that mis-scales the sensor geometry (kernel length,
readout shear, and intrinsics all become meaningless at the crop scale). `RUGDRaw`
is for **training the head only**.

### Retention cancels label noise only approximately
`R = m(c,s)/m0` is designed so RUGD's coarse ground truth largely cancels in the
clean→corrupt ratio. Two honest caveats to state in any write-up:
1. **mIoU is nonlinear**, so the cancellation is approximate, not exact.
2. **Geometric corruptions warp the GT mask** (nearest-neighbour) to stay aligned,
   which injects resampling artifacts and void borders into the GT itself. So
   *geometric* retention conflates perceptual degradation with GT-resampling noise,
   while *photometric* (motion-blur) retention is a clean same-GT comparison.
   **Report photometric and geometric retention separately**, and treat motion-blur
   retention as the cleanest single number.

### Severity values are plausible, not yet calibrated
The severity ladders (`corruptions/severity.py`) are ImageNet-C-style starting
values at the correct full-frame scale, but not yet tied to real legged IMU/gait
traces. Calibrating kernel length to (angular velocity × exposure) is the
refinement that turns this from "generic corruption on off-road data" into a
genuinely *gait-matched* benchmark — the main novelty lever (see literature doc).

### Reproducibility
Corruption RNG is seeded deterministically per (seed, frame, corruption, severity)
via a process-independent hash (not Python's salted `hash()`), so runs reproduce
across machines. The gate reports mean ± std over `--seeds` corruption seeds.
Frames (not patches) are the i.i.d. aggregation unit.
