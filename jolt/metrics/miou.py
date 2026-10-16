"""Streaming confusion-matrix mIoU (RESEARCH §5.2).

We accumulate a global KxK confusion matrix over the whole split and compute IoU
once at the end. This is the statistically correct dataset-level mIoU; it is NOT
the same as averaging per-image/per-batch mIoU (which is biased by small unions
and is what the legacy exploratory notebook did). Pixels labelled `ignore_index`
(RUGD void = 0, plus geometric out-of-frame fill) are dropped.
"""
import numpy as np


class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int = 0):
        self.n = num_classes
        self.ignore_index = ignore_index
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        """pred, gt: integer label arrays of identical shape."""
        pred = pred.reshape(-1).astype(np.int64)
        gt = gt.reshape(-1).astype(np.int64)
        valid = (gt != self.ignore_index) & (gt >= 0) & (gt < self.n) & \
                (pred >= 0) & (pred < self.n)
        gt, pred = gt[valid], pred[valid]
        idx = self.n * gt + pred
        self.mat += np.bincount(idx, minlength=self.n * self.n).reshape(self.n, self.n)

    def per_class_iou(self) -> np.ndarray:
        diag = np.diag(self.mat).astype(np.float64)
        union = self.mat.sum(0) + self.mat.sum(1) - diag
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union > 0, diag / union, np.nan)
        return iou

    def miou(self) -> float:
        """Mean IoU over classes that actually appear in the split (union > 0)."""
        iou = self.per_class_iou()
        return float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else 0.0
