"""Offline smoke test for the JOLT gate plumbing (no DINOv2/GPU needed).

Validates the full-frame corruptions, mask alignment, RNG determinism, color->id
conversion, and the metrics on real RUGD sample frames. Run:
    python -m jolt.scripts.smoke_test \
        --images-dir data/rugd_sample/RUGD_sample-data/images \
        --annotations-dir data/rugd_sample/RUGD_sample-data/annotations
"""
import argparse

import numpy as np

from jolt.corruptions import compose
from jolt.data.rugd import RUGDFrames, color_mask_to_classid, RUGD_COLOR_MAP
from jolt.metrics.boundary_f import BoundaryF
from jolt.metrics.miou import ConfusionMatrix


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True)
    p.add_argument("--annotations-dir", required=True)
    p.add_argument("--n", type=int, default=10)
    args = p.parse_args(argv)

    ds = RUGDFrames(args.images_dir, args.annotations_dir)
    print(f"[smoke] loaded {len(ds)} full frames")
    img0, mask0, name0 = ds[0]
    assert img0.dtype == np.uint8 and img0.ndim == 3, "raw frame must be uint8 HxWx3"
    print(f"[smoke] {name0}: frame{img0.shape} mask{mask0.shape} classes={np.unique(mask0)[:8]}")
    assert mask0.max() < 25 and mask0.min() >= 0, "class ids out of range"

    # 0. color->id conversion maps a known color correctly.
    for (r, g, b), cid in list(RUGD_COLOR_MAP.items())[:5]:
        patch = np.full((2, 2, 3), (r, g, b), np.uint8)
        assert (color_mask_to_classid(patch) == cid).all(), f"color {(r,g,b)} != {cid}"
    print("[smoke] color->classid conversion ok")

    # 1. Each corruption preserves shape/dtype; geometry warps mask, blur doesn't.
    for c in compose.ALL:
        ci, cm = compose.apply(c, img0.copy(), mask0.copy(), 5, compose.make_rng(0, 0, c, 5))
        assert ci.shape == img0.shape and ci.dtype == np.uint8 and cm.shape == mask0.shape
        changed = "mask CHANGED" if not np.array_equal(cm, mask0) else "mask same"
        print(f"[smoke] {c:>15} s5 ok ({changed})")
    assert np.array_equal(
        compose.apply("motion_blur", img0.copy(), mask0.copy(), 5,
                      compose.make_rng(0, 0, "motion_blur", 5))[1], mask0), \
        "motion_blur (photometric) must leave the mask unchanged"

    # 2. RNG determinism: same (seed,idx,corruption,severity) -> identical output.
    a_i, a_m = compose.apply("legged", img0.copy(), mask0.copy(), 4, compose.make_rng(7, 3, "legged", 4))
    b_i, b_m = compose.apply("legged", img0.copy(), mask0.copy(), 4, compose.make_rng(7, 3, "legged", 4))
    assert np.array_equal(a_i, b_i) and np.array_equal(a_m, b_m), "make_rng is not deterministic!"
    print("[smoke] RNG determinism ok (reproducible across calls)")

    # 3. Metrics: perfect predictor -> mIoU ~1; retention < 1 under a geometric warp.
    cmat, bf = ConfusionMatrix(25, 0), BoundaryF()
    for i in range(min(args.n, len(ds))):
        _, m, _ = ds[i]
        cmat.update(m.copy(), m); bf.update(m.copy(), m)
    m_clean = cmat.miou()
    print(f"[smoke] perfect-predictor clean mIoU={m_clean:.4f} (expect ~1.0), BF={bf.score():.4f}")
    assert m_clean > 0.99, "confusion-matrix mIoU broken"

    cm_corrupt = ConfusionMatrix(25, 0)
    for i in range(min(args.n, len(ds))):
        img, m, _ = ds[i]
        _, mw = compose.apply("jitter", img, m.copy(), 5, compose.make_rng(0, i, "jitter", 5))
        cm_corrupt.update(m, mw)  # clean-frame pred vs warped GT
    r = cm_corrupt.miou() / m_clean
    print(f"[smoke] clean-pred vs jitter-warped-GT retention R={r:.3f} (expect <1.0)")
    assert r < 1.0, "geometric warp should reduce agreement"

    print("[smoke] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
