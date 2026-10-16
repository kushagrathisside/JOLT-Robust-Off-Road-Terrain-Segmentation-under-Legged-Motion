"""Severity ladders for the gait-matched motion-degradation corruptions (RESEARCH §5.1).

Severities run 1..5 (ImageNet-C style). The numbers below are the *plausible
starting values* from the research doc; they are meant to be calibrated against
real legged IMU/gait traces later (a refinement, not a blocker). Keeping them in
one place makes that recalibration a single-file edit.
"""

# Motion blur: directional linear kernel length in pixels.
MOTION_BLUR_KERNEL_LEN = {1: 3, 2: 7, 3: 11, 4: 15, 5: 21}

# Rolling shutter: max horizontal displacement (px) of the bottom row relative
# to the top row (the row-wise readout smear).
ROLLING_SHUTTER_MAX_SHIFT = {1: 2, 2: 4, 3: 8, 4: 16, 5: 32}

# Viewpoint jitter: std-dev (degrees) of the sampled pitch/roll/yaw perturbation.
JITTER_ROT_SIGMA_DEG = {1: 0.5, 2: 1.0, 3: 2.0, 4: 4.0, 5: 8.0}

SEVERITIES = (1, 2, 3, 4, 5)
