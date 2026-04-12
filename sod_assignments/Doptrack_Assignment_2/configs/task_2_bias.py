"""
configs/task2_bias.py
Q2 — Effect of deactivating bias estimation.
Baseline (biases ON) vs biases OFF, per_pass arcs.
"""
import copy
from shared_config import DEFAULT_ACCELERATIONS, DEFAULT_INDICES

CONFIGS = [
    dict(
        label="Bias ON (baseline)",
        arc_type='per_pass',
        indices=DEFAULT_INDICES,
        accelerations=copy.deepcopy(DEFAULT_ACCELERATIONS),
        estimate_bias=True,
        bias_definition='per_pass',
    ),
    dict(
        label="Bias OFF",
        arc_type='per_pass',
        indices=DEFAULT_INDICES,
        accelerations=copy.deepcopy(DEFAULT_ACCELERATIONS),
        estimate_bias=False,
        bias_definition='per_pass',
    ),
]