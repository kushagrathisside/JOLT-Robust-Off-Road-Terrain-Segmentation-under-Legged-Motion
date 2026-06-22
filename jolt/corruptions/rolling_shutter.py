"""Rolling-shutter warp (JOLT §5.1).

A CMOS sensor reads rows sequentially: row r is exposed at time t + r*readout/H.
Under horizontal ego-motion this shears the image row-by-row. We model it as a
per-row horizontal displacement that grows linearly from the top row (0 px) to
the bottom row (`max_shift` px) of the FULL frame — so it must be applied to the
native-resolution frame, not a crop, for the readout geometry to be physical.
"""
import numpy as np

from .geometry import identity_map, resample_image, resample_mask
from .severity import ROLLING_SHUTTER_MAX_SHIFT


def rolling_shutter_map(h: int, w: int, severity: int, rng: np.random.Generator):
    """Backward sampling map for the row-wise readout shear."""
    max_shift = ROLLING_SHUTTER_MAX_SHIFT[severity]
    direction = 1.0 if rng.random() < 0.5 else -1.0  # left/right motion per image
    rows = np.arange(h, dtype=np.float32)
    row_shift = direction * max_shift * (rows / max(1, h - 1))  # (H,)
    map_x, map_y = identity_map(h, w)
    map_x = map_x + row_shift[:, None]
    return map_x, map_y


def rolling_shutter(image, mask, severity, rng):
    """Apply a row-wise time-shift warp (geometric: mask is warped to match)."""
    h, w = image.shape[:2]
    m = rolling_shutter_map(h, w, severity, rng)
    return resample_image(image, m), resample_mask(mask, m)
