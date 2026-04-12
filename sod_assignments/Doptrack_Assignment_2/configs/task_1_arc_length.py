"""
configs/task1_arc_length.py
Q1 — Effect of arc length on final residual.
Three runs: per_pass / per_day / per_3_days, all biases on, full dyn model.
"""
import copy
from shared_config import DEFAULT_ACCELERATIONS, DEFAULT_INDICES

CONFIGS = [
    dict(
        label="Arc: per_pass",
        arc_type='per_pass',
        indices=DEFAULT_INDICES,
        accelerations=copy.deepcopy(DEFAULT_ACCELERATIONS),
        estimate_bias=True,
        bias_definition='per_pass',
    ),
    dict(
        label="Arc: per_day",
        arc_type='per_day',
        indices=DEFAULT_INDICES,
        accelerations=copy.deepcopy(DEFAULT_ACCELERATIONS),
        estimate_bias=True,
        bias_definition='per_pass',
    ),
    dict(
        label="Arc: per_3_days",
        arc_type='per_3_days',
        indices=DEFAULT_INDICES,
        accelerations=copy.deepcopy(DEFAULT_ACCELERATIONS),
        estimate_bias=True,
        bias_definition='per_pass',
    ),
]
