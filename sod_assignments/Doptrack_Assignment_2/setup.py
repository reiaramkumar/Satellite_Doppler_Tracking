"""
setup.py — builds environment, propagator, estimator from a config dict.
"""

import sys
import os
import copy

_here = os.path.dirname(os.path.abspath(__file__))
for _p in [
    _here,
    os.path.abspath(os.path.join(_here, '..')),
    os.path.abspath(os.path.join(_here, '../..')),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tudatpy import constants
from tudatpy.estimation import estimation_analysis

from propagation_functions.environment import define_environment
from propagation_functions.propagation import (
    propagate_initial_state, get_initial_states,
    define_multi_arc_propagation_settings
)

# ALL estimation helpers live in estimation_functions.estimation
from estimation_functions.estimation import (
    define_observation_settings, define_parameters,
    simulate_observations_from_estimator,
    define_doptrack_station, define_arcs, get_link_ends_id,
    run_estimation, get_residuals_per_pass
)

# Only data loading lives in observations_data
from estimation_functions.observations_data import (
    load_and_format_observations, extract_recording_start_times_yml
)

from utility_functions.time import get_start_next_day
from utility_functions.tle import get_tle_initial_conditions, propagate_sgp4

from shared_config import (
    metadata, data, METADATA_FOLDER, DATA_FOLDER,
    MASS, REF_AREA, SRP_COEF, DRAG_COEF, SAT_NAME,
    DEFAULT_ACCELERATIONS
)


def build_estimator(cfg):
    indices         = cfg.get('indices', [0,1,2,3,4,6,7,8,9,10,11])
    arc_type        = cfg.get('arc_type', 'per_day')
    accelerations   = copy.deepcopy(cfg.get('accelerations', DEFAULT_ACCELERATIONS))
    bias_definition = cfg.get('bias_definition', 'per_pass')
    estimate_bias   = cfg.get('estimate_bias', True)
    prop_days       = cfg.get('propagation_days', 10.0)

    # Initial epoch + state
    initial_epoch, _, _ = get_tle_initial_conditions(
        METADATA_FOLDER + metadata[0], old_yml=False)
    propagation_time = prop_days * constants.JULIAN_DAY
    final_epoch      = get_start_next_day(initial_epoch) + propagation_time
    mid_epoch        = (initial_epoch + final_epoch) / 2.0
    initial_state    = propagate_sgp4(
        METADATA_FOLDER + metadata[0], initial_epoch, [mid_epoch], old_yml=False)[0, 1:]

    print(f"initial_epoch    = {initial_epoch}")
    print(f"final_epoch      = {final_epoch}")
    print(f"propagation_time = {propagation_time}")
    print(f"window days      = {(final_epoch - initial_epoch) / 86400:.2f}")

    # Observations
    recording_start_times = extract_recording_start_times_yml(
        METADATA_FOLDER, [metadata[i] for i in indices], old_yml=False)
    passes_start_times, passes_end_times, observation_times, observations_set = \
        load_and_format_observations(
            SAT_NAME, DATA_FOLDER,
            [data[i] for i in indices],
            recording_start_times, old_obs_format=False)

    # Arcs
    arc_start_times, arc_mid_times, arc_end_times = define_arcs(
        arc_type, passes_start_times, passes_end_times)

    # Single-arc environment for initial propagation
    bodies = define_environment(MASS, REF_AREA, DRAG_COEF, SRP_COEF,
                                SAT_NAME, multi_arc_ephemeris=False)
    propagate_initial_state(initial_state, initial_epoch, final_epoch,
                            bodies, accelerations, SAT_NAME)
    arc_wise_initial_states = get_initial_states(bodies, arc_mid_times, SAT_NAME)

    # Multi-arc environment
    bodies = define_environment(MASS, REF_AREA, DRAG_COEF, SRP_COEF,
                                SAT_NAME, multi_arc_ephemeris=True)
    multi_arc_propagator_settings = define_multi_arc_propagation_settings(
        arc_wise_initial_states, arc_start_times, arc_end_times,
        bodies, accelerations, SAT_NAME)

    define_doptrack_station(bodies)

    # Bias / observation settings
    Doppler_models = dict(
        constant_absolute_bias={
            'activated': estimate_bias,
            'time_interval': bias_definition
        },
        linear_absolute_bias={
            'activated': estimate_bias,
            'time_interval': bias_definition
        }
    )
    observation_settings = define_observation_settings(
        SAT_NAME, Doppler_models, passes_start_times, arc_start_times)

    # Parameters
    parameters_list = dict(
        initial_state={'estimate': True},
        constant_absolute_bias={'estimate': estimate_bias},
        linear_absolute_bias={'estimate': estimate_bias}
    )
    parameters_to_estimate = define_parameters(
        parameters_list, bodies, multi_arc_propagator_settings, SAT_NAME,
        arc_start_times, arc_mid_times,
        [(get_link_ends_id("DopTrackStation", SAT_NAME), passes_start_times)],
        Doppler_models
    )

    estimator = estimation_analysis.Estimator(
        bodies, parameters_to_estimate, observation_settings,
        multi_arc_propagator_settings)

    ideal_observations = simulate_observations_from_estimator(
        SAT_NAME, observation_times, estimator, bodies)

    print(f"First observation epoch: {observation_times[0]}")

    return dict(
        estimator=estimator,
        parameters_to_estimate=parameters_to_estimate,
        observations_set=observations_set,
        bodies=bodies,
        arc_start_times=arc_start_times,
        arc_mid_times=arc_mid_times,
        arc_end_times=arc_end_times,
        passes_start_times=passes_start_times,
        observation_times=observation_times,
        nb_arcs=len(arc_start_times),
        arc_wise_initial_states=arc_wise_initial_states,
        accelerations=accelerations,
    )