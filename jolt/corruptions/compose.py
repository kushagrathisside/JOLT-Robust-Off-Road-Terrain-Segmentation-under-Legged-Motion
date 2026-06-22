"""Corruption registry + the 'legged' composite (JOLT §5.1).

Every corruption shares the signature ``f(image, mask, severity, rng) -> (image, mask)``
so the gate/benchmark can iterate over them uniformly. Geometric corruptions warp
the mask to stay aligned; photometric ones pass it through. The composite combines
the two geometric warps into a single coordinate map so the mask is resampled only
once (avoids compounded nearest-neighbour error).
"""
import hashlib

import numpy as np

from .motion_blur import motion_blur, motion_blur_image
from .rolling_shutter import rolling_shutter, rolling_shutter_map
from .jitter import viewpoint_jitter, jitter_map
from .geometry import compose_maps, resample_image, resample_mask

# name -> callable. Order matters for the composite (blur last so it smears the
# already-warped image, like a real exposure over a moving/rolling capture).
CORRUPTIONS = {
    "motion_blur": motion_blur,
    "rolling_shutter": rolling_shutter,
    "jitter": viewpoint_jitter,
}

# The single-corruption set vs. the matched-severity composite.
SINGLE = tuple(CORRUPTIONS.keys())
COMPOSITE = "legged"
ALL = SINGLE + (COMPOSITE,)


def legged_composite(image, mask, severity, rng):
    """Apply jitter -> rolling shutter -> motion blur at one matched severity.

    The two geometric warps are composed into a single backward map, so the image
    and mask are each resampled exactly once before the (photometric) blur.
    """
    h, w = image.shape[:2]
    j = jitter_map(h, w, severity, rng)            # applied first
    r = rolling_shutter_map(h, w, severity, rng)   # applied second
    combined = compose_maps(outer=r, inner=j)
    image = resample_image(image, combined)
    mask = resample_mask(mask, combined)
    image = motion_blur_image(image, severity, rng)  # photometric, mask untouched
    return image, mask


def apply(name, image, mask, severity, rng):
    """Dispatch by corruption name (single or 'legged' composite)."""
    if name == COMPOSITE:
        return legged_composite(image, mask, severity, rng)
    if name not in CORRUPTIONS:
        raise KeyError(f"unknown corruption {name!r}; choices: {ALL}")
    return CORRUPTIONS[name](image, mask, severity, rng)


def _stable_int(text: str) -> int:
    """Process-independent 32-bit hash (Python's hash() is salted per run)."""
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def make_rng(seed, image_index, corruption_name, severity):
    """Deterministic per-(seed, image, corruption, severity) RNG, reproducible
    across processes (JOLT §5.5)."""
    seq = np.random.SeedSequence([int(seed), int(image_index),
                                  _stable_int(corruption_name), int(severity)])
    return np.random.default_rng(seq)
