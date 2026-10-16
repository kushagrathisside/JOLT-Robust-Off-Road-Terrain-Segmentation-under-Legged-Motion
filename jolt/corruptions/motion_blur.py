"""Directional linear motion blur (RESEARCH §5.1).

Models the exposure smear from body oscillation under a gait: a straight-line
kernel of length L at a random orientation per image. L grows with severity
(faster gait / longer exposure / rougher terrain).
"""
import cv2
import numpy as np

from .severity import MOTION_BLUR_KERNEL_LEN


def _linear_kernel(length: int, angle_deg: float) -> np.ndarray:
    """A normalized line-shaped blur kernel of the given length and orientation."""
    length = max(1, int(length))
    if length == 1:
        return np.ones((1, 1), np.float32)
    kernel = np.zeros((length, length), np.float32)
    # Draw a 1px line through the kernel center at the requested angle.
    c = (length - 1) / 2.0
    theta = np.deg2rad(angle_deg)
    dx, dy = np.cos(theta), np.sin(theta)
    half = (length - 1) / 2.0
    x0, y0 = c - dx * half, c - dy * half
    x1, y1 = c + dx * half, c + dy * half
    cv2.line(kernel, (int(round(x0)), int(round(y0))),
             (int(round(x1)), int(round(y1))), color=1.0, thickness=1)
    s = kernel.sum()
    return kernel / s if s > 0 else kernel


def motion_blur_image(image, severity, rng):
    """Directional motion blur on the image only (photometric). uint8 HxWx3 in/out."""
    length = MOTION_BLUR_KERNEL_LEN[severity]
    angle = float(rng.uniform(0.0, 180.0))  # random orientation per image
    kernel = _linear_kernel(length, angle)
    out = cv2.filter2D(image, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT)
    return out.astype(np.uint8)


def motion_blur(image, mask, severity, rng):
    """Apply directional motion blur (photometric: mask is unchanged).

    `image` uint8 HxWx3 RGB, `mask` HxW int. Returns (image, mask).
    """
    return motion_blur_image(image, severity, rng), mask
