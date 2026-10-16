"""Coordinate-map utilities for the geometric corruptions (RESEARCH §5.1).

A geometric corruption is represented as a backward sampling map
``(map_x, map_y)``: the corrupted output at pixel ``(x, y)`` reads the source
image at ``(map_x[y,x], map_y[y,x])``. Representing warps this way lets us

  * resample the RGB image (bilinear) and the label mask (nearest) consistently, and
  * **compose** several geometric warps and resample the mask exactly ONCE,
    avoiding the compounded nearest-neighbour error of warping it per step.

Out-of-frame samples are filled with the void id in the mask so the metric
ignores invented labels.
"""
import cv2
import numpy as np

VOID_ID = 0


def identity_map(h: int, w: int):
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    return map_x, map_y


def compose_maps(outer, inner):
    """Compose two backward maps so the result == apply `inner` then `outer`.

    If image1[p] = image0[inner(p)] and image2[p] = image1[outer(p)], then
    image2[p] = image0[inner(outer(p))]. We therefore sample `inner` at the
    coordinates given by `outer`.
    """
    ox, oy = outer
    ix, iy = inner
    cx = cv2.remap(ix, ox, oy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    cy = cv2.remap(iy, ox, oy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cx, cy


def resample_image(image, map_xy):
    map_x, map_y = map_xy
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT).astype(np.uint8)


def resample_mask(mask, map_xy):
    map_x, map_y = map_xy
    out = cv2.remap(mask.astype(np.int32), map_x, map_y, interpolation=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=VOID_ID)
    return out.astype(mask.dtype)
