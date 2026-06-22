"""Retention metric (JOLT §5.2) — *the* robustness number.

R(c,s) = m(c,s) / m0 in [0,1], where m0 is clean mIoU and m(c,s) is mIoU under
corruption c at severity s. Because R is a ratio of the same model against itself,
RUGD's coarse/noisy ground truth cancels — the whole point of the reframe.

mCR = mean of R over all (c,s) = single headline robustness score.
"""
from statistics import mean


def retention(m_corrupt: float, m_clean: float) -> float:
    if m_clean <= 0:
        return 0.0
    return m_corrupt / m_clean


def mcr(retentions) -> float:
    """Mean corruption retention over an iterable of R(c,s) values."""
    vals = list(retentions)
    return mean(vals) if vals else 0.0
