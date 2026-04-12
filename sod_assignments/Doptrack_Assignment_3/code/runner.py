import numpy as np

from tudatpy.estimation import estimation_analysis
from tudatpy.estimation.observations_setup import observations_wrapper
from tudatpy.estimation.observations import observations_processing
from tudatpy.estimation.observable_models_setup import model_settings
from tudatpy.dynamics import parameters

from scenario_builder import build_scenario
from perturbations import apply_perturbation
from analysis import compute_histories, compute_summary_metrics, compute_rsw_covariance


def run_scenario(config, verbose=True):
    scenario = build_scenario(config)
    parameters_to_estimate = scenario["parameters_to_estimate"]
    truth_parameters = scenario["truth_parameters"]
    estimator = scenario["estimator"]
    perturbed_parameters, initial_parameters_perturbation = apply_perturbation(config, scenario)
    simulated_observations = observations_wrapper.simulate_observations(
        scenario['observation_simulation_settings'],
        estimator.observation_simulators,
        scenario['bodies']
    )


    sorted_obs_collection = simulated_observations.sorted_observation_sets
    concatenated_times = np.array(simulated_observations.concatenated_times)
    concatenated_observations = np.array(simulated_observations.concatenated_observations)

    simulated_observations.set_constant_weight(
        config.noise_level ** -2,
        observations_processing.observation_parser(
            model_settings.one_way_instantaneous_doppler_type
        )
    )

    convergence_checker = estimation_analysis.estimation_convergence_checker(maximum_iterations=config.nb_iterations)
    estimation_input = estimation_analysis.EstimationInput(simulated_observations, convergence_checker=convergence_checker)

    estimation_input.define_estimation_settings(reintegrate_variational_equations=True)
    estimation_output = estimator.perform_estimation(estimation_input)

    updated_parameters = parameters_to_estimate.parameter_vector.copy()
    formal_errors = estimation_output.formal_errors
    true_errors = updated_parameters - truth_parameters
    correlations = estimation_output.correlations
    covariance = estimation_output.covariance
    parameters_history = estimation_output.parameter_history
    final_residuals = estimation_output.final_residuals
    residual_history = estimation_output.residual_history

    true_errors_history, true_to_formal_errors_history = compute_histories(
        parameters_history, truth_parameters, formal_errors
    )

    summary_metrics = compute_summary_metrics(
        final_residuals, true_errors, formal_errors, covariance
    )

    rsw_covariance, rsw_formal_errors, rsw_correlations = compute_rsw_covariance(
        covariance,
        scenario["arc_wise_initial_states"],
        parameters_to_estimate.parameter_set_size,
        scenario["nb_arcs"]
    )
    station_observation_data = []

    for obs_group in sorted_obs_collection.values():
        for i in range(len(obs_group)):
            station_observation_data.append({
                "station_index": i,
                "times": np.array(obs_group[i][0].observation_times),
                "values": np.array(obs_group[i][0].concatenated_observations),
            })

    if verbose:
        parameters.print_parameter_names(parameters_to_estimate)
        print("###############################################")
        print("PRINTING ESTIMATION OUTPUTS")
        print("estimated parameters", updated_parameters)
        print("initial parameters perturbation", initial_parameters_perturbation)
        print("true_errors", true_errors)
        print("formal errors", formal_errors)
        print("nb data points", len(concatenated_times))
        print("###############################################")


    return {
        "scenario_name": config.scenario_name,
        "config": config,
        "metadata": {
            "initial_epoch": scenario["initial_epoch"],
            "final_epoch": scenario["final_epoch"],
            "mid_epoch": scenario["mid_epoch"],
            "nb_arcs": scenario["nb_arcs"],
            "nb_parameters": parameters_to_estimate.parameter_set_size,
            "stations_names": scenario["stations_names"],
        },
        "raw": {
            "truth_parameters": truth_parameters,
            "perturbed_parameters": perturbed_parameters,
            "updated_parameters": updated_parameters,
            "initial_parameters_perturbation": initial_parameters_perturbation,
            "formal_errors": formal_errors,
            "true_errors": true_errors,
            "correlations": correlations,
            "covariance": covariance,
            "rsw_covariance": rsw_covariance,
            "rsw_formal_errors": rsw_formal_errors,
            "rsw_correlations": rsw_correlations,
            "parameters_history": parameters_history,
            "true_errors_history": true_errors_history,
            "true_to_formal_errors_history": true_to_formal_errors_history,
            "final_residuals": final_residuals,
            "residual_history": residual_history,
            "observation_times": concatenated_times,
            "observations": concatenated_observations,
            # "sorted_obs_collection": sorted_obs_colle,ction,
            "station_observation_data": station_observation_data,
            "arc_wise_initial_states": scenario["arc_wise_initial_states"],
        },
        "summary_metrics": summary_metrics,
    }


