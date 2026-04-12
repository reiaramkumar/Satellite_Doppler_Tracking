from runner import run_scenario
from io_utils import save_results
from scenarios.nominal import get_nominal_config
from plots import (
                    plot_station_map,
                    plot_simulated_observations,
                    plot_true_to_formal_ratio,
                    plot_residuals,
                    plot_final_residual_histogram,
                    plot_correlation_matrix,)



def main():

    # baseline config
    config = get_nominal_config()

    results = run_scenario(config, verbose=True)
    save_results(results, f"results/{config.scenario_name}.pkl")

    plot_station_map(config, save_path=f"figures/{config.scenario_name}_stations.png")
    plot_simulated_observations(results, save_path=f"figures/{config.scenario_name}_observations.png")
    plot_true_to_formal_ratio(results, save_path=f"figures/{config.scenario_name}_ratio.png")
    plot_residuals(results, save_path=f"figures/{config.scenario_name}_residuals.png")
    plot_final_residual_histogram(results, save_path=f"figures/{config.scenario_name}_hist.png")
    plot_correlation_matrix(results, rsw=False, save_path=f"figures/{config.scenario_name}_corr.png")
    plot_correlation_matrix(results, rsw=True, save_path=f"figures/{config.scenario_name}_corr_rsw.png")


if __name__ == "__main__":
    main()