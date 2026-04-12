from runner import run_scenario
from io_utils import save_results
from comparision_plots import *
from scenarios.noise_cases import get_noise_scenarios
from scenarios.station_cases import get_nearby_station_config, get_far_station_config
from scenarios.arc_cases import get_three_day_one_day_arcs
from scenarios.perturbation_cases import get_manual_perturbation_cases
from scenarios.gravity_cases import get_c20_c22_single_station, get_c20_c22_far_stations
from plots import (
                    plot_station_map,
                    plot_simulated_observations,
                    plot_true_to_formal_ratio,
                    plot_residuals,
                    plot_final_residual_histogram,
                    plot_correlation_matrix,)


def main():

    scenarios = []
    scenarios.extend(get_noise_scenarios())
    scenarios.append(get_three_day_one_day_arcs())
    scenarios.append(get_manual_perturbation_cases())
    scenarios.append(get_nearby_station_config())
    scenarios.append(get_far_station_config())
    scenarios.append(get_c20_c22_single_station())
    scenarios.append(get_c20_c22_far_stations())

    for config in scenarios:
        try:
            print(f"\n=== Running {config.scenario_name} ===")
            results = run_scenario(config, verbose=False)

            save_results(results, f"results/{config.scenario_name}.pkl")
            store_from_results(config.scenario_name, results)

            plot_station_map(config, save_path=f"figures/{config.scenario_name}/{config.scenario_name}_stations.png")
            plot_simulated_observations(results, save_path=f"figures/{config.scenario_name}/{config.scenario_name}_observations.png")
            plot_true_to_formal_ratio(results, save_path=f"figures/{config.scenario_name}/{config.scenario_name}_ratio.png")
            plot_residuals(results, save_path=f"figures/{config.scenario_name}/{config.scenario_name}_residuals.png")
            plot_final_residual_histogram(results, save_path=f"figures/{config.scenario_name}/{config.scenario_name}_hist.png")
            plot_correlation_matrix(results, rsw=False, save_path=f"figures/{config.scenario_name}/{config.scenario_name}_corr.png")
            plot_correlation_matrix(results, rsw=True, save_path=f"figures/{config.scenario_name}/{config.scenario_name}_corr_rsw.png")

            print(f"Saved: {config.scenario_name}")


        except Exception as e:
            print(f"FAILED: {config.scenario_name}")
            print(e)

if __name__ == "__main__":
    main()