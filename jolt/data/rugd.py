"""RUGD loaders (RESEARCH §5.4).

Two loaders, both yielding RAW uint8 RGB + integer class mask so corruptions are
applied to raw pixels *before* ImageNet normalization (a corruption models a
sensor/physics effect and must precede the model's input transform):

  * ``RUGDFrames`` — the ORIGINAL full-resolution frames (550x688) with the
    color-coded annotations converted to class ids. This is the correct unit for
    the corruption benchmark: motion blur / rolling shutter / viewpoint jitter are
    global, full-sensor ego-motion effects, so they must be synthesized on the
    native frame, and the gate must aggregate per-frame (frames are the i.i.d.
    unit, not random crops of them).
  * ``RUGDRaw`` — the pre-generated 224x224 patches. Kept for *training* the head
    NOT for the corruption benchmark (corrupting a crop mis-scales the
    sensor geometry).

Use ``imagenet_normalize`` to turn a (possibly corrupted) uint8 image into the
network input tensor.
"""
import os

import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

# Official RUGD colormap (RGB -> class id); see RUGD_annotation-colormap.txt.
RUGD_COLOR_MAP = {
    (0, 0, 0): 0, (108, 64, 20): 1, (255, 229, 204): 2, (0, 102, 0): 3,
    (0, 255, 0): 4, (0, 153, 153): 5, (0, 128, 255): 6, (0, 0, 255): 7,
    (255, 255, 0): 8, (255, 0, 127): 9, (64, 64, 64): 10, (255, 128, 0): 11,
    (255, 0, 0): 12, (153, 76, 0): 13, (102, 102, 0): 14, (102, 0, 0): 15,
    (0, 255, 128): 16, (204, 153, 255): 17, (102, 0, 204): 18, (255, 153, 204): 19,
    (0, 102, 102): 20, (153, 204, 255): 21, (102, 255, 255): 22, (101, 101, 11): 23,
    (114, 85, 47): 24,
}
NUM_CLASSES = 25
VOID_ID = 0

# Commonly-used RUGD sequence split (verify against your protocol before reporting).
RUGD_SPLITS = {
    "train": ("park-2", "trail", "trail-3", "trail-4", "trail-6", "trail-9",
              "trail-10", "trail-11", "trail-12", "trail-13", "trail-14",
              "trail-15", "creek"),
    "val": ("park-8", "trail-5", "trail-7"),
    "test": ("park-1", "trail-1", "trail-2"),
}


def _build_color_lut():
    """RGB tuple -> class id, packed for vectorized lookup."""
    return {(r << 16) | (g << 8) | b: cid for (r, g, b), cid in RUGD_COLOR_MAP.items()}


_COLOR_LUT = _build_color_lut()


def color_mask_to_classid(rgb_mask: np.ndarray) -> np.ndarray:
    """Convert an HxWx3 RGB annotation to an HxW class-id mask (unknown -> void)."""
    packed = (rgb_mask[..., 0].astype(np.uint32) << 16) | \
             (rgb_mask[..., 1].astype(np.uint32) << 8) | rgb_mask[..., 2].astype(np.uint32)
    out = np.full(packed.shape, VOID_ID, dtype=np.int64)
    for key, cid in _COLOR_LUT.items():
        out[packed == key] = cid
    return out


def imagenet_normalize(image_uint8: np.ndarray) -> torch.Tensor:
    """uint8 HxWx3 RGB -> normalized CxHxW float tensor (DINOv2/ImageNet stats)."""
    x = image_uint8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)).contiguous()


def _imread_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class RUGDFrames:
    """Original full-res RUGD frames + color annotations -> class-id masks.

    Yields (image_uint8 HxWx3 RGB, mask HxW int64, name). The benchmark unit.

    Layout options:
      * flat:   images_dir/*.png  + annotations_dir/*.png   (the sample-data zip)
      * nested: images_dir/<seq>/<frame>.png  (the full download); pass `sequences`
                to keep only those (use RUGD_SPLITS[split]).
    """

    def __init__(self, images_dir: str, annotations_dir: str, sequences=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.pairs = []  # (image_path, annotation_path, name)
        for dirpath, _, names in os.walk(images_dir):
            for fn in sorted(names):
                if not fn.endswith((".png", ".jpg")):
                    continue
                stem = os.path.splitext(fn)[0]
                seq = stem.rsplit("_", 1)[0]  # e.g. 'creek_00001' -> 'creek'
                if sequences is not None and seq not in sequences:
                    continue
                img_path = os.path.join(dirpath, fn)
                rel = os.path.relpath(img_path, images_dir)
                ann_path = os.path.join(annotations_dir, rel)
                if not os.path.exists(ann_path):  # flat fallback
                    ann_path = os.path.join(annotations_dir, fn)
                if os.path.exists(ann_path):
                    self.pairs.append((img_path, ann_path, stem))
        if not self.pairs:
            raise FileNotFoundError(
                f"no (image, annotation) pairs under {images_dir} / {annotations_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, ann_path, name = self.pairs[idx]
        img = _imread_rgb(img_path)
        mask = color_mask_to_classid(_imread_rgb(ann_path))
        return img, mask, name


class RUGDRaw:
    """Pre-generated 224x224 patches (for TRAINING the head, not the benchmark).

    Yields (image_uint8 HxWx3 RGB, mask HxW int64, name). Masks are already
    single-channel class ids (produced by RUGD_dataset_augmentation.ipynb, on the
    legacy-notebooks branch).
    """

    def __init__(self, root: str, split: str = "val"):
        self.images_dir = os.path.join(root, split, "images")
        self.masks_dir = os.path.join(root, split, "masks")
        self.files = sorted(f for f in os.listdir(self.images_dir)
                            if f.endswith((".jpg", ".png")))
        if not self.files:
            raise FileNotFoundError(f"no images under {self.images_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = _imread_rgb(os.path.join(self.images_dir, name))
        mask_path = os.path.join(self.masks_dir, os.path.splitext(name)[0] + ".png")
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            mask = np.zeros(img.shape[:2], np.uint8)
        return img, mask.astype(np.int64), name
