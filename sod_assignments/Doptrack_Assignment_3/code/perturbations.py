import numpy as np
from tudatpy.dynamics import environment


def apply_perturbation(config, scenario):
    parameters_to_estimate = scenario['parameters_to_estimate']
    truth_parameters = scenario['truth_parameters']
    arc_mid_times = scenario['arc_mid_times']
    nb_arcs = scenario['nb_arcs']

    perturbed_parameters = truth_parameters.copy()

    if config.use_next_tle_as_perturbation:
        next_delfi_tle = environment.Tle(   "1 32789U 08021G   20092.14603172 +.00001512 +00000-0 +10336-3 0  9992",
                                            "2 32789 097.4277 137.6209 0011263 214.0075 146.0432 15.07555919650162")
        next_delfi_ephemeris = environment.TleEphemeris('Earth', 'J2000', next_delfi_tle, False)
        perturbed_arc_wise_initial_states = []
        for time in arc_mid_times:
            perturbed_arc_wise_initial_states.append(next_delfi_ephemeris.cartesian_state(time))

        for i in range(nb_arcs):
            perturbed_parameters[i * 6:(i + 1) * 6] = perturbed_arc_wise_initial_states[i]


    if config.use_manual_perturbation:
        manual_state_perturbation = np.concatenate((config.manual_position_perturbation * np.ones(3),
                                                   config.manual_velocity_perturbation * np.ones(3)))

        for i in range(nb_arcs):
            perturbed_parameters[i * 6:(i + 1) * 6] += manual_state_perturbation

    parameters_to_estimate.parameter_vector = perturbed_parameters
    initial_parameters_perturbation = perturbed_parameters - truth_parameters

    return perturbed_parameters, initial_parameters_perturbation

