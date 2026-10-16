"""Viewpoint jitter (RESEARCH §5.1).

Small per-frame camera rotations (pitch/roll/yaw) from legged body sway, applied
as the induced homography under a pinhole model. Intrinsics are derived from the
FULL frame (focal ~ frame width), so this must run on the native-resolution frame.
Rotation magnitudes are drawn from a zero-mean normal whose std grows with severity.
"""
import cv2
import numpy as np

from .geometry import identity_map, resample_image, resample_mask
from .severity import JITTER_ROT_SIGMA_DEG


def _rotation_matrix(pitch: float, roll: float, yaw: float) -> np.ndarray:
    """3x3 rotation from intrinsic pitch(x), yaw(y), roll(z) angles in radians."""
    cx, sx = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cz, sz = np.cos(roll), np.sin(roll)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float64)
    return rz @ ry @ rx


def jitter_map(h: int, w: int, severity: int, rng: np.random.Generator):
    """Backward sampling map for a small random rotational homography."""
    sigma = np.deg2rad(JITTER_ROT_SIGMA_DEG[severity])
    pitch, roll, yaw = rng.normal(0.0, sigma, size=3)
    f = float(w)  # focal ~ frame width (~53deg HFOV); principal point at center
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], np.float64)
    H = K @ _rotation_matrix(pitch, roll, yaw) @ np.linalg.inv(K)
    # Backward map: output pixel (x,y) samples source at H^-1 (x,y,1).
    Hinv = np.linalg.inv(H)
    map_x, map_y = identity_map(h, w)
    ones = np.ones_like(map_x)
    src = np.stack([map_x, map_y, ones], axis=-1) @ Hinv.T  # (H,W,3)
    z = src[..., 2]
    return (src[..., 0] / z).astype(np.float32), (src[..., 1] / z).astype(np.float32)


def viewpoint_jitter(image, mask, severity, rng):
    """Apply a small random rotational homography (geometric: mask warped to match)."""
    h, w = image.shape[:2]
    m = jitter_map(h, w, severity, rng)
    return resample_image(image, m), resample_mask(mask, m)
