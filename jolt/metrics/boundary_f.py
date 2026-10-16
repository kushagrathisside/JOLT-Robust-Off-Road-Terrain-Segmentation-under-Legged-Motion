"""Boundary-F1 (RESEARCH §5.2).

Boundaries degrade first under motion blur, so we report a boundary score
alongside mIoU. This is the class-agnostic contour F-measure (Csurka et al.):
a predicted boundary pixel counts as a hit if it lies within `tol` px of a
ground-truth boundary pixel, and vice-versa. Counts are streamed across the
split so the final F is computed once. Like mIoU, its *retention* is what we
care about, so coarse GT boundaries cancel in the clean->corrupt ratio.
"""
import cv2
import numpy as np


def _boundary(labels: np.ndarray) -> np.ndarray:
    """Binary boundary map: pixels whose label differs from a 3x3 neighbour."""
    lab = labels.astype(np.int32)
    # Morphological gradient on labels detects any local label change.
    mx = cv2.dilate(lab.astype(np.float32), np.ones((3, 3), np.uint8))
    mn = cv2.erode(lab.astype(np.float32), np.ones((3, 3), np.uint8))
    return (mx != mn).astype(np.uint8)


class BoundaryF:
    def __init__(self, tol: int = 3, ignore_index: int = 0):
        self.tol = tol
        self.ignore_index = ignore_index
        self.match_pred = 0  # pred-boundary px matched by gt (precision numerator)
        self.total_pred = 0
        self.match_gt = 0    # gt-boundary px matched by pred (recall numerator)
        self.total_gt = 0
        k = 2 * tol + 1
        self._kernel = np.ones((k, k), np.uint8)

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        valid = (gt != self.ignore_index)
        pb = _boundary(pred) * valid
        gb = _boundary(gt) * valid
        pb_dil = cv2.dilate(pb, self._kernel)
        gb_dil = cv2.dilate(gb, self._kernel)
        self.match_pred += int(((pb == 1) & (gb_dil == 1)).sum())
        self.total_pred += int(pb.sum())
        self.match_gt += int(((gb == 1) & (pb_dil == 1)).sum())
        self.total_gt += int(gb.sum())

    def score(self) -> float:
        prec = self.match_pred / self.total_pred if self.total_pred else 0.0
        rec = self.match_gt / self.total_gt if self.total_gt else 0.0
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)
