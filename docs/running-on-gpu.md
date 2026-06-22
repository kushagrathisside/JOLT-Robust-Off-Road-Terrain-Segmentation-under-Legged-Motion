# Running JOLT on GPU (RTX 4060 / 8 GB)

Concrete commands for running the gate on a local NVIDIA GPU. Tuned for an
**8 GB** card (RTX 4060): the only knobs that matter are `--batch-size`,
`--infer-size`, and the SAM2 variant. No session limits locally, so long runs
(SAM2) can just finish.

## 0. Setup

```bash
git clone <repo> && cd Image-Segmentation-Using-ViT-Finetuning
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
# DINOv2 head checkpoint from your training run -> best_model.pth
```

## 2. Sanity check (fast, CPU is fine)

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
