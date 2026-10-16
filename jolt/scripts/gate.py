"""RESEARCH §4 — THE GATE. Run this first; build nothing downstream until it passes.

Empirical question: does legged-motion corruption actually degrade SOTA terrain
segmentation? We evaluate a model on clean vs. each corruption x severity over the
ORIGINAL full-resolution RUGD frames and report retention R = m(c,s)/m0 (and
boundary-F retention), with mean +/- std over multiple corruption seeds.

Corruptions are synthesized on the native frame (global ego-motion is a full-sensor
effect), then the frame is resized to a multiple-of-14 inference size for the model
and the logits are upsampled back to full resolution for scoring. Frames (not crops)
are the i.i.d. aggregation unit.

    # primary subject (DINOv2 frozen + head):
    python -m jolt.scripts.gate \
        --images-dir RUGD/images --annotations-dir RUGD/annotations \
        --split val --model dinov2 --checkpoint best_model.pth \
        --seeds 3 --limit 300 --out runs/gate

    # ceiling ABLATION (optional, heavier): --model sam2

Decision (advisory, printed at the end):
    large, monotone drop in retention -> real problem -> build the benchmark and method  (PASS)
    small / flat drop                 -> SOTA already robust -> pivot (§7) (FAIL)
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

from jolt.corruptions import compose
from jolt.corruptions.severity import SEVERITIES
from jolt.data.rugd import (RUGDFrames, RUGD_SPLITS, imagenet_normalize)
from jolt.metrics.boundary_f import BoundaryF
from jolt.metrics.miou import ConfusionMatrix
from jolt.metrics.retention import retention


# --------------------------------------------------------------------------- #
# Predictors. A predictor maps a list of full-res uint8 frames -> list of
# full-res int label maps (same H,W as each input frame).
# --------------------------------------------------------------------------- #
def build_dinov2_predictor(args):
    from jolt.models.dinov2_head import load_dinov2_segmenter
    model = load_dinov2_segmenter(args.num_classes, args.backbone,
                                  args.checkpoint, args.device)
    ih, iw = args.infer_size

    @torch.no_grad()
    def predict(frames):
        tensors = [imagenet_normalize(cv2.resize(f, (iw, ih), interpolation=cv2.INTER_LINEAR))
                   for f in frames]
        logits = model(torch.stack(tensors).to(args.device))  # (B,K,ih,iw)
        preds = []
        for i, f in enumerate(frames):
            h, w = f.shape[:2]
            up = torch.nn.functional.interpolate(
                logits[i:i + 1], size=(h, w), mode="bilinear", align_corners=False)
            preds.append(up.argmax(1)[0].cpu().numpy())
        return preds
    return predict


def build_predictor(args):
    if args.model == "dinov2":
        return build_dinov2_predictor(args)
    if args.model == "sam2":
        from jolt.models.sam2_wrapper import build_sam2_predictor
        return build_sam2_predictor(args)
    raise ValueError(f"unknown model {args.model!r}")


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _eval_pass(predict, dataset, indices, corruption, severity, seed, args):
    """One (corruption, severity, seed) pass; corruption=None => clean.

    Returns (ConfusionMatrix, BoundaryF).
    """
    cm = ConfusionMatrix(args.num_classes, ignore_index=args.ignore_index)
    bf = BoundaryF(tol=args.boundary_tol, ignore_index=args.ignore_index)
    buf_imgs, buf_masks = [], []

    def flush():
        if not buf_imgs:
            return
        for p, m in zip(predict(buf_imgs), buf_masks):
            cm.update(p, m)
            bf.update(p, m)
        buf_imgs.clear()
        buf_masks.clear()

    for idx in indices:
        img, mask, _ = dataset[idx]
        if corruption is not None:
            rng = compose.make_rng(seed, idx, corruption, severity)
            img, mask = compose.apply(corruption, img, mask, severity, rng)
        buf_imgs.append(img)
        buf_masks.append(mask)
        if len(buf_imgs) >= args.batch_size:
            flush()
    flush()
    return cm, bf


def run_gate(args):
    sequences = RUGD_SPLITS.get(args.split) if args.use_splits else None
    dataset = RUGDFrames(args.images_dir, args.annotations_dir, sequences=sequences)
    n = len(dataset)
    indices = list(range(n))
    if args.limit and args.limit < n:
        rng = np.random.default_rng(args.seed_base)
        indices = sorted(rng.choice(n, size=args.limit, replace=False).tolist())
    print(f"[gate] {len(indices)}/{n} frames | model={args.model} | infer={args.infer_size} "
          f"| corruptions={args.corruptions} | severities={args.severities} | seeds={args.seeds}")

    predict = build_predictor(args)

    # Clean reference (seed-independent).
    cm0, bf0 = _eval_pass(predict, dataset, indices, None, None, 0, args)
    m0, bf0_score = cm0.miou(), bf0.score()
    print(f"[gate] clean mIoU m0={m0:.4f}  boundary-F={bf0_score:.4f}")

    results = {"clean": {"miou": m0, "boundary_f": bf0_score,
                         "per_class_iou": _per_class(cm0)},
               "config": {"model": args.model, "n_frames": len(indices),
                          "seeds": args.seeds, "infer_size": list(args.infer_size),
                          "split": args.split},
               "corruptions": {}}

    seeds = list(range(args.seed_base, args.seed_base + args.seeds))
    for c in args.corruptions:
        results["corruptions"][c] = {}
        for s in args.severities:
            mious, bfs = [], []
            for seed in seeds:
                cm, bf = _eval_pass(predict, dataset, indices, c, s, seed, args)
                mious.append(cm.miou())
                bfs.append(bf.score())
            mious, bfs = np.array(mious), np.array(bfs)
            r = mious / m0 if m0 > 0 else np.zeros_like(mious)
            rbf = bfs / bf0_score if bf0_score > 0 else np.zeros_like(bfs)
            results["corruptions"][c][s] = {
                "miou_mean": float(mious.mean()), "miou_std": float(mious.std()),
                "retention_mean": float(r.mean()), "retention_std": float(r.std()),
                "boundary_f_retention_mean": float(rbf.mean()),
            }
            print(f"[gate] {c:>15} s{s}: mIoU={mious.mean():.4f}+/-{mious.std():.4f}  "
                  f"R={r.mean():.3f}+/-{r.std():.3f}  R_BF={rbf.mean():.3f}")

    _summarize(results, args)
    return results


def _per_class(cm):
    iou = cm.per_class_iou()
    return {i: (None if np.isnan(v) else float(v)) for i, v in enumerate(iou)}


# --------------------------------------------------------------------------- #
# Reporting + verdict
# --------------------------------------------------------------------------- #
def _retention_grid(results):
    return {c: {int(s): v["retention_mean"] for s, v in by_s.items()}
            for c, by_s in results["corruptions"].items()}


def _verdict(results):
    """Advisory go/no-go (RESEARCH §4), informed by the across-seed spread."""
    grid = _retention_grid(results)
    severities = sorted({s for r in grid.values() for s in r})
    if not severities:
        return "NO DATA", []
    smax = max(severities)
    mean_at_max = float(np.mean([r.get(smax, 1.0) for r in grid.values()]))
    worst_at_max = float(min(r.get(smax, 1.0) for r in grid.values()))
    monotone = all(
        all(r.get(severities[i], 1.0) >= r.get(severities[i + 1], 1.0) - 0.02
            for i in range(len(severities) - 1))
        for r in grid.values()
    )
    notes = [f"mean retention @ s{smax} = {mean_at_max:.3f}; "
             f"worst-corruption retention @ s{smax} = {worst_at_max:.3f}; "
             f"monotone-degrading = {monotone}",
             "Thresholds below are heuristics, not significance tests — read the "
             "per-corruption curves and the across-seed std before committing."]
    if mean_at_max <= 0.85 and monotone:
        return "PASS — large, monotone drop -> build the benchmark and method", notes
    if mean_at_max <= 0.92:
        return "BORDERLINE — moderate drop; inspect curves/seeds before committing", notes
    return "FAIL — small/flat drop -> SOTA already robust -> pivot to cross-dataset (§7)", notes


def _summarize(results, args):
    os.makedirs(args.out, exist_ok=True)
    verdict, notes = _verdict(results)
    results["verdict"] = verdict
    results["verdict_notes"] = notes

    with open(os.path.join(args.out, "gate_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    lines = ["corruption,severity,miou_mean,miou_std,retention_mean,retention_std,boundary_f_retention_mean"]
    for c, by_s in results["corruptions"].items():
        for s, v in sorted(by_s.items()):
            lines.append(f"{c},{s},{v['miou_mean']:.4f},{v['miou_std']:.4f},"
                         f"{v['retention_mean']:.4f},{v['retention_std']:.4f},"
                         f"{v['boundary_f_retention_mean']:.4f}")
    with open(os.path.join(args.out, "gate_table.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")

    grid = _retention_grid(results)
    severities = sorted({s for r in grid.values() for s in r})
    md = [f"# JOLT Gate — {args.model}", "",
          f"- clean mIoU `m0` = **{results['clean']['miou']:.4f}**",
          f"- clean boundary-F = {results['clean']['boundary_f']:.4f}",
          f"- frames = {results['config']['n_frames']}, seeds = {results['config']['seeds']}, "
          f"infer-size = {results['config']['infer_size']}", "",
          "## Retention R(c,s) — mean over seeds", "",
          "| corruption | " + " | ".join(f"s{s}" for s in severities) + " |",
          "|" + "---|" * (len(severities) + 1)]
    for c in grid:
        md.append(f"| {c} | " + " | ".join(f"{grid[c].get(s, float('nan')):.3f}"
                                            for s in severities) + " |")
    md += ["", f"## Verdict\n\n**{verdict}**", "", *(f"- {n}" for n in notes)]
    with open(os.path.join(args.out, "gate_summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    _maybe_plot(grid, severities, args)
    print(f"\n[gate] VERDICT: {verdict}")
    print(f"[gate] wrote results to {args.out}/ (json, csv, md)")


def _maybe_plot(grid, severities, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.figure(figsize=(6, 4))
    for c, r in grid.items():
        plt.plot(severities, [r.get(s, np.nan) for s in severities], marker="o", label=c)
    plt.axhline(1.0, color="grey", ls="--", lw=0.8)
    plt.ylim(0, 1.05)
    plt.xlabel("severity"); plt.ylabel("retention R = m(c,s)/m0")
    plt.title(f"JOLT gate — {args.model}"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "retention_curves.png"), dpi=120)
    plt.close()


def _parse_size(text):
    h, w = text.lower().split("x")
    return int(h), int(w)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="RESEARCH §4 gate diagnostic (full-frame RUGD)")
    p.add_argument("--images-dir", required=True, help="RUGD frames dir (flat or nested by sequence)")
    p.add_argument("--annotations-dir", required=True, help="RUGD color-annotation dir")
    p.add_argument("--split", default="val", choices=list(RUGD_SPLITS))
    p.add_argument("--use-splits", action="store_true",
                   help="filter frames by RUGD_SPLITS[split] sequences (nested layout)")
    p.add_argument("--model", default="dinov2", choices=["dinov2", "sam2"],
                   help="dinov2 = the subject (primary); sam2 = ceiling ABLATION")
    p.add_argument("--backbone", default="dinov2_vits14")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--num-classes", type=int, default=25)
    p.add_argument("--ignore-index", type=int, default=0, help="RUGD void class")
    p.add_argument("--infer-size", type=_parse_size, default=(448, 560),
                   help="HxW multiple-of-14 inference size (default 448x560 ~ RUGD aspect)")
    p.add_argument("--corruptions", nargs="+", default=list(compose.ALL))
    p.add_argument("--severities", nargs="+", type=int, default=list(SEVERITIES))
    p.add_argument("--seeds", type=int, default=3, help="number of corruption seeds (mean+/-std)")
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="subsample N frames (deterministic)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--boundary-tol", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="runs/gate")
    return p.parse_args(argv)


if __name__ == "__main__":
    run_gate(parse_args())
