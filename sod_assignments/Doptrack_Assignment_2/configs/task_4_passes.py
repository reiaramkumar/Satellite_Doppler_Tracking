"""
configs/task4_passes.py
Q4 — Effect of pass selection on orbit quality.
Bad passes identified from visual inspection of per-pass residual plots:
  Pass 4 (idx 3) — sharp spikes >60 m/s, radio frequency interference
  Pass 5 (idx 4) — systematic curved trend, unmodelled drift
  Pass 6 (idx 5) — elevated residuals at start, receiver start-up transient
  Pass 7 (idx 6) — two large isolated spikes ~50 m/s
Best pass identified as Pass 8 (idx 7) — symmetric, zero-mean, no artifacts
"""
import copy
from shared_config import DEFAULT_ACCELERATIONS, DEFAULT_INDICES

ALL_PASSES  = DEFAULT_INDICES           # all 11 passes
BAD_PASSES  = [3, 4, 5, 6]             # 0-indexed: passes 4,5,6,7
BEST_PASS   = [7]                       # 0-indexed: pass 8


def _cfg(label, indices):
    return dict(
        label=label,
        arc_type='per_pass',
        indices=indices,
        accelerations=copy.deepcopy(DEFAULT_ACCELERATIONS),
        estimate_bias=True,
        bias_definition='per_pass',
    )


CONFIGS = [
    _cfg("All passes",        ALL_PASSES),
    _cfg("Bad passes removed", [i for i in ALL_PASSES if i not in BAD_PASSES]),
    _cfg("Best pass only",    BEST_PASS),
]