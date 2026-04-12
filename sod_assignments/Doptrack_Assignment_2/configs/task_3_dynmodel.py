"""
configs/task3_dynmodel.py
Q3 — Effect of dynamical model changes.
per_pass arcs, all biases on.

Note: propagation.py checks for 'spherical_harmonic_gravity' key explicitly,
so it must be present even when False. Point-mass case runs nb_iterations=1
because it diverges badly after the first iteration (initial residual ~8000 m/s)
due to the large J2 error making the linearisation invalid.
"""
import copy
from shared_config import DEFAULT_ACCELERATIONS, DEFAULT_INDICES


def _mod(changes: dict):
    acc = copy.deepcopy(DEFAULT_ACCELERATIONS)
    for body, params in changes.items():
        acc[body].update(params)
    return acc


CONFIGS = [
    dict(
        label="Full model (baseline)",
        arc_type='per_pass',
        indices=DEFAULT_INDICES,
        accelerations=_mod({}),
        estimate_bias=True,
        bias_definition='per_pass',
    ),
    dict(
        label="No Earth drag",
        arc_type='per_pass',
        indices=DEFAULT_INDICES,
        accelerations=_mod({'Earth': {'drag': False}}),
        estimate_bias=True,
        bias_definition='per_pass',
    ),
    dict(
        label="No solar radiation pressure",
        arc_type='per_pass',
        indices=DEFAULT_INDICES,
        accelerations=_mod({'Sun': {'solar_radiation_pressure': False}}),
        estimate_bias=True,
        bias_definition='per_pass',
    ),
    dict(
        label="Point mass Earth (no SH)",
        arc_type='per_pass',
        indices=DEFAULT_INDICES,
        accelerations=_mod({'Earth': {
            'point_mass_gravity': True,
            'spherical_harmonic_gravity': False,  # key must exist for propagation.py
            'drag': True,
        }}),
        estimate_bias=True,
        bias_definition='per_pass',
        nb_iterations=1,  # diverges after first iteration — record first-pass metrics only
    ),
]