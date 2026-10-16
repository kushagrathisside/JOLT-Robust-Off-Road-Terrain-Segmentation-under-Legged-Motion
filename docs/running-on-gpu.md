# Running JOLT on GPU (RTX 4060 / 8 GB)

Concrete commands for running the gate on a local NVIDIA GPU. Tuned for an
**8 GB** card (RTX 4060): the only knobs that matter are `--batch-size`,
`--infer-size`, and the SAM2 variant. No session limits locally, so long runs
(SAM2) can just finish.

## 0. Setup

```bash
git clone https://github.com/kushagrathisside/JOLT-Robust-Off-Road-Terrain-Segmentation-under-Legged-Motion.git
cd JOLT-Robust-Off-Road-Terrain-Segmentation-under-Legged-Motion
python -m venv .venv && source .venv/bin/activate
pip install -e .

# verify the GPU is visible (expect: True and your card name)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi
```
If `cuda` is `False`, install the matching NVIDIA driver (≥ ~545 for a 4060) and a
CUDA-enabled torch build. Then freeze your exact env for reproducibility:
```bash
pip freeze > requirements.lock
```

## 1. Data + weights

```bash
# RUGD full frames + color annotations -> data/RUGD/ (see data/README.md)
#   data/RUGD/images/*.png   data/RUGD/annotations/*.png
# DINOv2 head checkpoint -> best_model.pth (see "Head checkpoint" below)
```

### Head checkpoint (`best_model.pth`)

The repository has **no head-training script**, and the checkpoint is not in git (`*.pth`
is ignored). The only head-training code is the legacy Colab notebook
`legacy/DinoV2_FineTuning_RUGD.ipynb`, which lives on the **`legacy-notebooks`** branch.
Nothing here has been run to produce a checkpoint.

**Inputs.** The notebook reads `rugd_224_patches.zip` from Google Drive. The
`RUGD_dataset_augmentation.ipynb` notebook (also on `legacy-notebooks`) makes that zip
by cutting the RUGD frames into 224×224 patches.

**Training settings** (training loop, notebook cell "4. Training Loop"; cell index 3):

- Backbone: `dinov2_vits14` from torch hub, frozen.
- Head: Conv 3×3 384→256, BatchNorm, ReLU, Conv 1×1 256→25 (`NUM_CLASSES = 25`).
- Optimizer: AdamW, lr 3e-4, on the head parameters only.
- Loss: cross-entropy. Batch size 64, 15 epochs.
- Saving: `torch.save(model.state_dict(), "best_model.pth")` when the validation metric
  improves.

To reproduce, run the notebook's cells 0–3 in Colab with the patch zip on Drive. Then
copy `best_model.pth` to the repository root, or pass its path with `--checkpoint`.

**Caveats (not yet measured):**

- The head is trained on **224×224 patches**, but the gate infers on **full 448×560
  frames**. The effect of that scale change is unknown.
- The notebook's validation mIoU is averaged per batch and includes the void class. The
  gate's metric is dataset-level and ignores void. The two numbers are not comparable.
- The saved `state_dict` has `backbone.*` and `head.*` keys. `jolt/models/dinov2_head.py`
  refuses a checkpoint that does not populate the head.
- The notebook's other saves (`best_model_finetuned.pth`, `best_model_base_finetuned.pth`)
  come from unfrozen or ViT-Base runs. They are not the documented head.

## 2. Sanity check (CPU is fine; needs torch and the RUGD sample)

```bash
python -m jolt.scripts.smoke_test \
    --images-dir data/rugd_sample/RUGD_sample-data/images \
    --annotations-dir data/rugd_sample/RUGD_sample-data/annotations
```

## 3. The gate — DINOv2 (primary subject)

Quick go/no-go on a subset (minutes):
```bash
python -m jolt.scripts.gate \
    --images-dir data/RUGD/images --annotations-dir data/RUGD/annotations \
    --split val --model dinov2 --backbone dinov2_vits14 --checkpoint best_model.pth \
    --seeds 3 --limit 300 --batch-size 8 --infer-size 448x560 \
    --device cuda --out runs/gate_dinov2_s
```

Full val run (still minutes on a 4060 — DINOv2 is cheap):
```bash
python -m jolt.scripts.gate \
    --images-dir data/RUGD/images --annotations-dir data/RUGD/annotations \
    --split val --model dinov2 --checkpoint best_model.pth \
    --seeds 3 --batch-size 8 --infer-size 448x560 --device cuda \
    --out runs/gate_dinov2_full
```

**8 GB notes**
- `--infer-size 448x560` (~RUGD aspect, both ÷14) fits comfortably. Higher fidelity:
  `--infer-size 518x644`; if you hit OOM, drop `--batch-size` to 4 or shrink infer-size.
- DINOv2-Base: add `--backbone dinov2_vitb14` (fits 8 GB at batch 4–8).

## 4. The gate — SAM2 (ceiling ABLATION, optional + heavy)

SAM2 is compute-bound (~seconds/image) and its wrapper is currently a stub
(implement the mask→semantic-label assignment first). On 8 GB use a **small/base-plus**
variant and a **hard subsample**; it can run uninterrupted:
```bash
python -m jolt.scripts.gate \
    --images-dir data/RUGD/images --annotations-dir data/RUGD/annotations \
    --split val --model sam2 --seeds 1 --limit 150 \
    --device cuda --out runs/gate_sam2
```

## 5. Read the result

```
runs/<name>/gate_summary.md      # retention table + advisory verdict
runs/<name>/gate_table.csv       # per corruption × severity: miou, retention (mean/std)
runs/<name>/gate_results.json    # full results incl. clean per-class IoU
runs/<name>/retention_curves.png # retention vs. severity
```

## Tips
- Watch usage in another shell: `watch -n1 nvidia-smi`.
- Long SAM2 runs heat a laptop — keep it plugged in and ventilated.
- OOM → lower `--batch-size`, then `--infer-size`. Throughput-bound (not OOM) →
  raise `--batch-size`.
- Reproducibility: same `--seed-base` + `--seeds` reproduces corruptions exactly
  (deterministic RNG), independent of machine.
