"""SAM2 baseline (RESEARCH §5.3 — the ceiling).

SAM2 is the strongest zero-shot mask model; if even SAM2 degrades under
legged-motion corruption, the gate's case is strongest. SAM2 is not pip-installable
in every environment, so this is a thin, optional wrapper that fails loudly with
setup instructions rather than silently faking a result.

To enable: `pip install git+https://github.com/facebookresearch/sam2.git`, download
the checkpoint, then pass --model sam2. For semantic eval you need a label
assignment for SAM2's class-agnostic masks (e.g. majority-vote GT label per mask
for an upper-bound 'oracle-assignment' ceiling, or a CLIP/text head). Document the
assignment choice in the paper — it materially affects the numbers.
"""


def build_sam2_predictor(args):
    raise NotImplementedError(
        "SAM2 wrapper is a stub. Install SAM2 and implement mask->semantic-label "
        "assignment here before using --model sam2. See module docstring."
    )
