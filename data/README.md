# data/

This folder holds datasets locally. **Its contents are git-ignored** (see
`../.gitignore`) — only this README is tracked, so the folder always exists but no
data is committed.

## RUGD (primary, §5.4)
Original frames + color annotations: <http://rugd.vision/>
```
data/RUGD/
  images/        # full-resolution frames (550x688), e.g. creek_00001.png
  annotations/   # color-coded label PNGs, same names
```
Either a flat layout (as in the sample-data zip) or the official nested-by-sequence
layout works; pass `--use-splits` to filter by the canonical train/val/test
sequences (see `jolt/data/rugd.py: RUGD_SPLITS`).

The gate/benchmark synthesize corruptions on these **full frames** — do not
pre-crop. The 224×224 patch zip (`rugd_224_patches.zip`) is for **training the
head only** (`RUGDRaw`), not for the corruption benchmark.

## Cross-dataset (§7, future)
- RELLIS-3D: <https://arxiv.org/abs/2011.12954>
- GOOSE / GOOSE-Ex (legged subset): <https://goose-dataset.de/>
