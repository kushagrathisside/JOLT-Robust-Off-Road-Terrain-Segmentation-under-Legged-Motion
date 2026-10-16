"""DINOv2 frozen-backbone segmenter (RESEARCH §5.3 — the pipeline under study, the subject).

Mirrors the FrozenBackboneSegmenter in the legacy notebook
DinoV2_FineTuning_RUGD.ipynb (on the legacy-notebooks branch), so an existing
checkpoint loads directly: frozen DINOv2-S/14 patch tokens -> a light conv head ->
bilinear upsample to input size.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = {"dinov2_vits14": 384, "dinov2_vitb14": 768}


class FrozenBackboneSegmenter(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "dinov2_vits14"):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone)
        for p in self.backbone.parameters():
            p.requires_grad = False
        # Patch size (14 for all DINOv2 ViTs) — read it rather than hard-code.
        self.patch_size = getattr(self.backbone, "patch_size", 14)
        dim = EMBED_DIM[backbone]
        self.head = nn.Sequential(
            nn.Conv2d(dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        if h % self.patch_size or w % self.patch_size:
            raise ValueError(
                f"input {h}x{w} not divisible by patch_size {self.patch_size}; "
                "resize to a multiple of 14 before the model (see gate --infer-size)")
        nH, nW = h // self.patch_size, w // self.patch_size
        with torch.no_grad():
            tokens = self.backbone.forward_features(x)["x_norm_patchtokens"]
        b, n, c = tokens.shape
        assert n == nH * nW, f"token count {n} != grid {nH}x{nW}"
        # Reshape supports NON-SQUARE grids (was sqrt(n), which assumed square).
        feat = tokens.permute(0, 2, 1).reshape(b, c, nH, nW)
        logits = self.head(feat)
        return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)


def load_dinov2_segmenter(num_classes: int, backbone: str = "dinov2_vits14",
                          checkpoint: str | None = None, device: str = "cpu"):
    model = FrozenBackboneSegmenter(num_classes, backbone)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        state = state.get("state_dict", state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        # Fail loud: a head that didn't load means we'd benchmark random weights.
        head_missing = [k for k in missing if k.startswith("head.")]
        if head_missing:
            raise RuntimeError(
                f"checkpoint did not populate the head: missing {head_missing}. "
                f"(unexpected keys: {list(unexpected)[:5]}...). Refusing to run on "
                "an untrained head; check the checkpoint / key names.")
        if missing or unexpected:
            print(f"[dinov2] loaded with missing={list(missing)[:5]} "
                  f"unexpected={list(unexpected)[:5]}")
    elif checkpoint is None:
        print("[dinov2] WARNING: no checkpoint — head is randomly initialized "
              "(fine only for plumbing tests, NOT for real retention numbers).")
    return model.to(device).eval()
