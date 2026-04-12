"""
configs/task5_validation.py
Q5 — Validation: compare observation residuals vs TLE distance.
Reuses the per_pass and per_day arc runs from task1 for cross-comparison.
"""
import copy
from shared_config import DEFAULT_ACCELERATIONS, DEFAULT_INDICES

CONFIGS = [
    dict(
        label="Validation: per_pass, all passes",
        arc_type='per_pass',
        indices=DEFAULT_INDICES,
        accelerations=copy.deepcopy(DEFAULT_ACCELERATIONS),
        estimate_bias=True,
        bias_definition='per_pass',
    ),
    dict(
        label="Validation: per_day, all passes",
        arc_type='per_day',
        indices=DEFAULT_INDICES,
        accelerations=copy.deepcopy(DEFAULT_ACCELERATIONS),
        estimate_bias=True,
        bias_definition='per_pass',
    ),
]