import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def plot_station_map(config, save_path=None):
    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.scatter(4.3571, 52.0116, color='red', marker='*', s=200, label='DopTrack')

    if config.nb_fake_stations > 0:
        ax.scatter(config.stations_lon, config.stations_lat, color='blue', marker='*', s=200, label='Fake stations')

    ax.gridlines(draw_labels=True)
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()



def plot_simulated_observations(results, save_path=None):
    data = results["raw"]["station_observation_data"]
    initial_epoch = results["metadata"]["initial_epoch"]

    plt.figure()
    plt.title("Simulated Doppler observations")

    for i, station in enumerate(data):
        plt.scatter(
            (station["times"] - initial_epoch) / 3600.0,
            station["values"],
            s=8,
            label=f"Station {i}"
        )

    plt.grid()
    plt.xlabel("Time since initial epoch [hr]")
    plt.ylabel("Range-rate [m/s]")
    plt.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close()



def plot_true_to_formal_ratio(results, save_path=None):
    ratios = results["raw"]["true_to_formal_errors_history"][:, -1]
    nb_parameters = results["metadata"]["nb_parameters"]

    plt.figure()
    plt.title("True-to-formal errors ratio")
    plt.scatter(np.arange(nb_parameters), ratios, color="blue")
    plt.plot(
        np.arange(nb_parameters),
        np.ones(nb_parameters),
        linestyle="dashed",
        color="blue",
        label="true error = formal error"
    )
    plt.xlabel("Parameter index [-]")
    plt.ylabel("True-to-formal errors ratio [-]")
    plt.grid()
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_residuals(results, save_path=None):
    observation_times = results["raw"]["observation_times"]
    residual_history = results["raw"]["residual_history"]
    initial_epoch = results["metadata"]["initial_epoch"]
    noise_level = results["config"].noise_level

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))

    x = (np.array(observation_times) - initial_epoch) / 3600.0

    ax1.scatter(x, residual_history[:, 0], color='blue', label='residuals')
    ax1.plot(x, noise_level * np.ones(len(observation_times)), color='red', label='1sigma noise level')
    ax1.plot(x, -noise_level * np.ones(len(observation_times)), color='red')
    ax1.plot(x, 3 * noise_level * np.ones(len(observation_times)), color='red', linestyle='dotted', label='3sigma noise level')
    ax1.plot(x, -3 * noise_level * np.ones(len(observation_times)), color='red', linestyle='dotted')
    ax1.set_ylabel('Residuals [m/s]')
    ax1.set_xlabel('Time since initial epoch [hr]')
    ax1.set_title('First iteration')
    ax1.grid()
    ax1.legend()

    ax2.scatter(x, residual_history[:, -1], color='blue', label='residuals')
    ax2.plot(x, noise_level * np.ones(len(observation_times)), color='red', label='1sigma noise level')
    ax2.plot(x, -noise_level * np.ones(len(observation_times)), color='red')
    ax2.plot(x, 3 * noise_level * np.ones(len(observation_times)), color='red', linestyle='dotted', label='3sigma noise level')
    ax2.plot(x, -3 * noise_level * np.ones(len(observation_times)), color='red', linestyle='dotted')
    ax2.set_ylabel('Residuals [m/s]')
    ax2.set_xlabel('Time since initial epoch [hr]')
    ax2.set_title('Final iteration')
    ax2.grid()
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()



def plot_final_residual_histogram(results, save_path=None):
    final_residuals = results["raw"]["final_residuals"]
    observation_times = results["raw"]["observation_times"]
    noise_level = results["config"].noise_level

    plt.figure()
    plt.hist(final_residuals, 25, color='blue')
    plt.plot(-1.0 * noise_level * np.ones(2), [0, len(observation_times) / 10.0], color='red', linestyle='solid', label='1sigma noise level')
    plt.plot(1.0 * noise_level * np.ones(2), [0, len(observation_times) / 10.0], color='red', linestyle='solid')
    plt.plot(-3.0 * noise_level * np.ones(2), [0, len(observation_times) / 10.0], color='red', linestyle='dashed', label='3sigma noise level')
    plt.plot(3.0 * noise_level * np.ones(2), [0, len(observation_times) / 10.0], color='red', linestyle='dashed')
    plt.xlabel('Final iteration range-rate residual [m/s]')
    plt.ylabel('Occurrences [-]')
    plt.title('Final residuals histogram')
    plt.tight_layout()
    plt.grid()
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()



def plot_correlation_matrix(results, rsw=False, save_path=None):
    matrix = results["raw"]["rsw_correlations"] if rsw else results["raw"]["correlations"]
    title = "Correlation matrix (state RSW)" if rsw else "Correlation matrix"

    plt.figure()
    plt.imshow(np.abs(matrix), aspect='auto', interpolation='none')
    plt.colorbar(label='Absolute correlation [-]')
    plt.title(title)
    plt.xlabel('Parameter index [-]')
    plt.ylabel('Parameter index [-]')

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
