import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import os


def save_plot(fig, run_id, name, dpi = 400):
    dir = os.path.join('figures', run_id)
    os.makedirs(dir, exist_ok = True)
    fig.savefig(os.path.join(dir, f"{run_id}_{name}.png"), dpi=dpi, bbox_inches="tight")







def plots(case):

    # ... PLOT 1 - PROPAGATED ORBIT ...

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'Delfi-C3 trajectory around Earth')
    ax.plot(case['cartesian_states'][:, 1] / 1.0e3, case['cartesian_states'][:, 2] / 1.0e3, case['cartesian_states'][:, 3] / 1.0e3,
            label='Delfi-C3', linestyle='-.')
    ax.scatter(0.0, 0.0, 0.0, label="Earth", marker='o', color='blue')
    ax.legend()
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.set_zlabel('z [km]')
    save_plot(fig, case['run_id'], 'plot_1')
    plt.show()
    plt.close(fig)




    # ... PLOT 2 - ACCELERATIONS ON DELFI ...

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(f'Accelerations on Delfi')
    for i in range(np.shape(case['saved_accelerations'])[1] - 1):
        ax.plot((case['saved_accelerations'][:, 0] - case['start_recording_day']) / 86400, case['saved_accelerations'][:, i + 1],
                label=case['accelerations_ids'][i], linestyle='-')
    ax.legend()
    ax.set_xlabel('Time [Days since first recording day]')
    ax.set_ylabel('Acceleration [m/s]')
    plt.yscale('log')
    plt.grid()
    save_plot(fig, case['run_id'], 'plot_2')
    plt.show()
    plt.close(fig)




    # ... PLOT 3 - KEPLERIAN ELEMENTS ...
    fig = plt.figure(figsize=(12, 6))

    # semi-major axis
    ax = fig.add_subplot(231)
    ax.plot((case['keplerian_states'][:, 0] - case['start_recording_day']) / 3600, (case['keplerian_states'][:, 1]) / 1.0e3, linestyle='-.')
    ax.set_xlabel('Time [hours since start of TLE]')
    ax.set_ylabel('Semi-major axis [km]')
    ax.grid()

    # eccentricity
    ax = fig.add_subplot(232)
    ax.set_title(f'Propagated orbital elements of Delfi-C3')
    ax.plot((case['keplerian_states'][:, 0] - case['start_recording_day']) / 3600, case['keplerian_states'][:, 2], linestyle='-.')
    ax.set_xlabel('Time [hours since start of TLE]')
    ax.set_ylabel('Eccentricity [-]')
    ax.grid()

    # inclination
    ax = fig.add_subplot(233)
    ax.plot((case['keplerian_states'][:, 0] - case['start_recording_day']) / 3600, case['keplerian_states'][:, 3] / np.pi * 180, linestyle='-.')
    ax.set_xlabel('Time [hours since start of TLE]')
    ax.set_ylabel('Inclination [deg]')
    ax.grid()

    # argument of periapsis
    ax = fig.add_subplot(234)
    ax.plot((case['keplerian_states'][:, 0] - case['start_recording_day']) / 3600, (case['keplerian_states'][:, 4]) / np.pi * 180,
            linestyle='-.')
    ax.set_xlabel('Time [hours since start of TLE]')
    ax.set_ylabel('Argument of perigee [deg]')
    ax.grid()

    # right ascension of the ascending node
    ax = fig.add_subplot(235)
    ax.plot((case['keplerian_states'][:, 0] - case['start_recording_day']) / 3600, case['keplerian_states'][:, 5] / np.pi * 180, linestyle='-.')
    ax.set_xlabel('Time [hours since start of TLE]')
    ax.set_ylabel('RAAN [deg]')
    ax.grid()

    # true anomaly
    ax = fig.add_subplot(236)
    ax.plot((case['keplerian_states'][:, 0] - case['start_recording_day']) / 3600, case['keplerian_states'][:, 6] / np.pi * 180, linestyle='-.')
    ax.set_xlabel('Time [hours since start of TLE]')
    ax.set_ylabel('True anomaly [deg]')
    ax.grid()
    fig.tight_layout()
    save_plot(fig, case['run_id'], 'plot_3')
    plt.show()
    plt.close(fig)




    # ... PLOT 4 - DIFFERENCES BETWEEN PROPAGATED AND TLE ORBITS IN ORBITAL ELEMENTS ...

    fig = plt.figure(figsize=(12,6))

    # semi-major axis
    ax = fig.add_subplot(231)
    ax.plot((case['keplerian_states'][:, 0] - case['start_recording_day'])/3600, (case['keplerian_difference_wrt_tle'][:,1])/1.0e3, linestyle='-.')
    ax.set_xlabel('Time [hours since start of TLE]')
    ax.set_ylabel('Semi-major axis [km]')
    ax.grid()

    # eccentricity
    ax = fig.add_subplot(232)
    ax.set_title(f'Difference between propagated and TLE orbital elements')
    ax.plot((case['keplerian_states'][:, 0] - case['start_recording_day'])/3600, case['keplerian_difference_wrt_tle'][:,2], linestyle='-.')
    ax.set_xlabel('Time [hours since start of TLE]')
    ax.set_ylabel('Eccentricity [-]')
    ax.grid()

    # inclination
    ax = fig.add_subplot(233)
    ax.plot((case['keplerian_states'][:, 0] - case['start_recording_day'])/3600, case['keplerian_difference_wrt_tle'][:,3]/np.pi*180, linestyle='-.')
    ax.set_xlabel('Time [hours since start of TLE]')
    ax.set_ylabel('Inclination [deg]')
    ax.grid()

    # argument of periapsis
    ax = fig.add_subplot(234)
    ax.plot((case['keplerian_states'][:, 0] - case['start_recording_day'])/3600, (case['keplerian_difference_wrt_tle'][:,4])/np.pi*180, linestyle='-.')
    ax.set_xlabel('Time [hours since start of TLE]')
    ax.set_ylabel('Argument of perigee [deg]')
    ax.grid()

    # right ascension of the ascending node
    ax = fig.add_subplot(235)
    ax.plot((case['keplerian_states'][:, 0] - case['start_recording_day'])/3600, case['keplerian_difference_wrt_tle'][:,5]/np.pi*180, linestyle='-.')
    ax.set_xlabel('Time [hours since start of TLE]')
    ax.set_ylabel('RAAN [deg]')
    ax.grid()

    # true anomaly
    ax = fig.add_subplot(236)
    ax.plot((case['keplerian_states'][:, 0] - case['start_recording_day'])/3600, case['keplerian_difference_wrt_tle'][:,6]/np.pi*180, linestyle='-.')
    ax.set_xlabel('Time [hours since start of TLE]')
    ax.set_ylabel('True anomaly [deg]')
    ax.grid()
    fig.tight_layout()
    save_plot(fig, case['run_id'], 'plot_4')
    plt.close(fig)




    # ... PLOT 5 - DIFFERENCES BETWEEN PROPAGATED AND TLE POSITION IN RSW COORDINATES ...

    # Radial direction
    fig = plt.figure(figsize=(12,3))
    ax = fig.add_subplot(131)
    ax.plot((case['propagation_epochs'] - case['start_recording_day'])/3600, (case['rsw_difference_wrt_tle'][:,1])/1.0e3, linestyle='-.')
    ax.set_xlabel('Time [hours since start of TLE]')
    ax.set_ylabel('Radial diff [km]')
    ax.grid()

    # Along-track direction
    ax = fig.add_subplot(132)
    ax.set_title(f'Difference between propagated and TLE orbits in RSW')
    ax.plot((case['propagation_epochs'] - case['start_recording_day'])/3600, (case['rsw_difference_wrt_tle'][:,2])/1.0e3, linestyle='-.')
    ax.set_xlabel('Time [hours since start of TLE]')
    ax.set_ylabel('Along-track diff [km]')
    ax.grid()

    # Cross-track direction
    ax = fig.add_subplot(133)
    ax.plot((case['propagation_epochs'] - case['start_recording_day'])/3600, (case['rsw_difference_wrt_tle'][:,3])/1.0e3, linestyle='-.')
    ax.set_xlabel('Time [hours since start of TLE]')
    ax.set_ylabel('Cross-track diff [km]')
    ax.grid()
    fig.tight_layout()
    save_plot(fig, case['run_id'], 'plot_5')
    plt.close(fig)



    # ... PLOT 6 - GROUND TRACK AND VISIBILITY ...

    location_groundstation_lon = 4.3754
    location_groundstation_lat = 51.9899

    R = 6371360
    h =  547500;
    Lambda = np.arccos((R)/(R+h))
    Phi_E = np.linspace(0, 2*np.pi, num=1000)

    # Create hemisphere function
    mask_E = []
    for i in range(len(Phi_E)):
        val = (-Phi_E[i]) % 2*np.pi
        if val >= 0 and val < np.pi:
            mask_E.append(1.0)
        else:
            mask_E.append(-1.0)


    # Calculate horizon coordinates on the map.
    colat_horizon = np.arccos(np.cos(Lambda)*np.cos((90-location_groundstation_lat)/180*np.pi)+np.sin(Lambda)*np.sin((90-location_groundstation_lat)/180*np.pi)*np.cos(Phi_E % 2*np.pi))
    DL = ((mask_E * np.arccos((np.cos(Lambda)-np.cos(colat_horizon)*np.cos((90-location_groundstation_lat)/180*np.pi))/(np.sin((90-location_groundstation_lat)/180*np.pi)*np.sin(colat_horizon)))))

    latitude_horizon = (90-(colat_horizon/np.pi*180))
    longitude_horizon_abs = ((location_groundstation_lon/180*np.pi-DL)/np.pi*180)
    longitude_horizon = np.where(longitude_horizon_abs <= 180, longitude_horizon_abs, longitude_horizon_abs - 360)

    # Plot groundtrack with visibility area of ground station
    fig = plt.figure(figsize = (12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.scatter(case['longitudes']/np.pi*180, case['latitudes']/np.pi*180, label='Delfi-C3', marker='.',s=2)
    ax.scatter(location_groundstation_lon,location_groundstation_lat, color='red', marker='*',s=200)
    ax.plot(longitude_horizon,latitude_horizon,color ='red')
    ax.gridlines(draw_labels=True)
    plt.title('Ground track and visibility')
    save_plot(fig, case['run_id'], 'plot_6')
    plt.close(fig)




    # ... PLOT 7 - ELEVATION OF THE PASSES ...

    sin_rho = R/(R+h)
    DL_e = np.deg2rad(np.rad2deg(case['longitudes'][:,1])-location_groundstation_lon)
    Lambda_e = np.arccos(np.cos(case['latitudes'][:,1])*np.cos(np.deg2rad(location_groundstation_lat))+np.sin(case['latitudes'][:,1])*np.sin(np.deg2rad(location_groundstation_lat))*np.cos(DL_e))

    # Calculate elevation
    eta = np.arctan2(sin_rho*np.sin(Lambda_e),1-sin_rho*np.cos(Lambda_e))
    elevation_abs = np.rad2deg(np.arccos(np.sin(eta)/sin_rho))
    elevation_lambda_check = np.where(Lambda_e <= np.arccos(R/(R+h)), elevation_abs, 0)
    elevation = np.where(np.abs(DL_e)<= 0.5*np.pi, elevation_lambda_check, 0)

    # Plot elevation
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(211)
    ax.set_title(f'Azimuth and Elevation')
    ax.plot((case['longitudes'][:,0] - case['start_recording_day'])/3600, elevation, color='red')
    ax.set_xlim(7.0, 10.5)
    ax.set_xlabel('Time [hours since start of day]')
    ax.set_ylabel('Elevation [deg]')
    plt.grid()
    save_plot(fig, case['run_id'], 'plot_7')
    plt.close(fig)




    # ... PLOT 8 - AZIMUTH OF THE PASSES ...

    # Create hemisphere function
    mask_DL = []
    for i in range(len(DL_e)):
        val = (DL_e[i]) % 2*np.pi
        if val >= 0 and val < np.pi:
            mask_DL.append(1.0)
        else:
            mask_DL.append(-1.0)

    azimuth_abs = np.rad2deg(mask_DL*np.arccos((np.cos(np.deg2rad(location_groundstation_lat))-np.cos(case['latitudes'][:,1])*np.cos(Lambda_e))/(np.sin(case['latitudes'][:,1])*np.sin(Lambda_e))))
    azimuth_lambda_check = np.where(Lambda_e <= np.arccos(R/(R+h)), azimuth_abs, 0)
    azimuth = np.where(np.abs(DL_e)<= 0.5*np.pi, azimuth_lambda_check, 0)


    # Plot azimuth
    ax = fig.add_subplot(212)
    ax.plot((case['longitudes'][:,0] - case['start_recording_day'])/3600, azimuth % 360, color='red')
    ax.set_xlim(7.0, 10.5)
    ax.set_xlabel('Time [hours since start of day]')
    ax.set_ylabel('azimuth [deg]')
    plt.grid(True)
    save_plot(fig, case['run_id'], 'plot_8')
    plt.close(fig)
    case['azimuth'] = azimuth
    case['elevation'] = elevation






    # ... PLOT 9 - SIMULATED DOPPLER DATA ...
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(f'Simulated Doppler data')
    ax.plot((case['simulated_obs_times'] - case['start_recording_day'])/3600, case['simulated_doppler'], color='red', linestyle='none', marker='.')
    ax.set_xlabel('Time [hours since start of day]')
    ax.set_ylabel('Doppler [m/s]')
    plt.grid()
    save_plot(fig, case['run_id'], 'plot_9')
    plt.close(fig)



    # ... PLOT 10 - SIMULATED FREQUENCY DATA ...
    # With simplified doppler formula, calculate received frequency
    satellite_frequency = 145870000
    speed_of_light = 299792458

    frequency_received = (1-case['simulated_doppler']/speed_of_light)*satellite_frequency

    # Plot simulated frequency data
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(f'Simulated received frequency')
    ax.plot((case['simulated_obs_times'] - case['start_recording_day'])/3600, case['simulated_doppler'], color='red', linestyle='none', marker='.')
    ax.set_xlabel('Time [hours since start of day]')
    ax.set_ylabel('Radio frequency [Hz]')
    plt.grid()
    save_plot(fig, case['run_id'], 'plot_10a')
    plt.close(fig)


    # ... PLOT 10 - SIMULATED VS REAL DOPPLER ...

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot()
    ax.set_title(f'Doppler data')
    ax.plot((np.array(case['simulated_obs_times']) - case['start_recording_day'])/3600, case['simulated_doppler'], label='simulated', color='red', linestyle='none', marker='.')
    ax.plot((np.array(case['observation_times']) - case['start_recording_day'])/3600, case['real_doppler'], label='recorded', color='blue', linestyle='none', marker='.')
    ax.legend()
    ax.set_xlabel('Time [hours since start of day]')
    ax.set_ylabel('Doppler [m/s]')
    plt.grid()
    save_plot(fig, case['run_id'], 'plot_10b')
    plt.close(fig)




    ### PLOT 11 - SINGLE PASS OBSERVATIONS

    fig = plt.figure(figsize=(10,6), constrained_layout=True)

    ax1 = fig.add_subplot(2,2,1)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)

    ax1.plot((case['interpolated_times'] - case['start_recording_day'])/3600, case['interpolated_real_obs'][:,1], label='recorded', color='blue', linestyle='none', marker='.')
    ax1.plot((case['interpolated_times'] - case['start_recording_day'])/3600, case['interpolated_simulated_obs'][:,1], label='simulated', color='red', linestyle='none', marker='.')
    ax1.grid()
    ax1.set_title(f'Doppler')
    ax1.legend()
    ax1.set_xlabel('Time [hours since start of day]')
    ax1.set_ylabel('Doppler [m/s]')

    ax3.plot((case['interpolated_times'] - case['start_recording_day'])/3600, case['first_residual_obs'], label='residual', color='green', linestyle='none', marker='.')
    ax3.plot((np.linspace(case['interpolated_times'][0], case['interpolated_times'][len(case['interpolated_times'])-1]) - case['start_recording_day'])/3600, label='linear fit', color='black', linestyle='-')
    ax3.grid()
    ax3.set_title(f'First residual (recorded - simulated)')
    ax3.legend()
    ax3.set_xlabel('Time [hours since start of day]')
    ax3.set_ylabel('Residual [m/s]')

    ax4.plot((case['interpolated_times'] - case['start_recording_day'])/3600, case['second_residual_obs'], label='residual', color='purple', linestyle='none', marker='.')
    ax4.grid()
    ax4.set_title(f'Second residual (first residual - linear fit)')
    ax4.legend()
    ax4.set_xlabel('Time [hours since start of day]')
    ax4.set_ylabel('Residual [m/s]')
    save_plot(fig, case['run_id'], 'plot_11')
    plt.close(fig)
    plt.show()



import numpy as np
import matplotlib.pyplot as plt

def comparison_plots(cases_dict, run_ids, save=True):
    """
    Make comparison plots for the selected runs.

    Parameters
    ----------
    cases_dict : dict
        Mapping run_id (str) -> case (dict). Each case must contain the same keys
        you used in your single-run plotting function (cartesian_states, keplerian_states, etc.).
    run_ids : list[str]
        Example: ["00_BASE", "11_G_PM"]
    save : bool
        If True, calls save_plot(fig, run_id, plot_name). Requires save_plot to exist.

    Notes
    -----
    - Plots 1,2,3,4,5,10,11 are OVERLAID (comparison).
    - Plots 6,7,8,9 and frequency are ONE-FIGURE-PER-RUN (one map per run for Plot 6).
    - Cartopy is required only for Plot 6 (map). If not installed, Plot 6 is skipped.
    """

    # ---- build cases list safely (fixes "string indices must be integers" error) ----
    missing = [rid for rid in run_ids if rid not in cases_dict]
    if missing:
        raise KeyError(f"Missing run_ids in cases_dict: {missing}")

    cases = [cases_dict[rid] for rid in run_ids]

    def _label(case):
        return f"run {case.get('run_id', 'UNKNOWN')}"

    def _save(fig, run_id, name):
        if save:
            # save_plot must exist in your project
            dir = os.path.join('figures/comparison', run_id)
            os.makedirs(dir, exist_ok=True)
            fig.savefig(os.path.join(dir, f"{run_id}_{name}.png"), dpi=400, bbox_inches="tight")


    # =========================
    # PLOT 1 - 3D TRAJECTORY (comparison)
    # =========================
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Delfi-C3 trajectory around Earth (comparison)")

    for case in cases:
        ax.plot(
            case["cartesian_states"][:, 1] / 1.0e3,
            case["cartesian_states"][:, 2] / 1.0e3,
            case["cartesian_states"][:, 3] / 1.0e3,
            linestyle="-.",
            label=_label(case),
        )

    ax.scatter(0.0, 0.0, 0.0, label="Earth", marker="o")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_zlabel("z [km]")
    ax.legend()
    _save(fig, "COMPARISON", "plot_1_compare")
    plt.show()
    plt.close(fig)

    # =========================
    # PLOT 2 - ACCELERATIONS (comparison)
    # =========================
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot()
    ax.set_title("Accelerations on Delfi (comparison)")

    for case in cases:
        t_days = (case["saved_accelerations"][:, 0] - case["start_recording_day"]) / 86400.0
        for i in range(np.shape(case["saved_accelerations"])[1] - 1):
            ax.plot(
                t_days,
                case["saved_accelerations"][:, i + 1],
                linestyle="-",
                label=f"{_label(case)} | {case['accelerations_ids'][i]}",
            )

    ax.set_xlabel("Time [days since first recording day]")
    ax.set_ylabel("Acceleration [m/s²]")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend(fontsize=8, ncol=2)
    _save(fig, "COMPARISON", "plot_2_compare")
    plt.show()
    plt.close(fig)

    def _kepler_time_hours(case):
        return (case["keplerian_states"][:, 0] - case["start_recording_day"]) / 3600.0

    # =========================
    # PLOT 3 - KEPLER ELEMENTS (comparison)
    # =========================
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Propagated orbital elements of Delfi-C3 (comparison)")

    axes = [
        fig.add_subplot(231), fig.add_subplot(232), fig.add_subplot(233),
        fig.add_subplot(234), fig.add_subplot(235), fig.add_subplot(236)
    ]

    y_defs = [
        ("Semi-major axis [km]", lambda c: c["keplerian_states"][:, 1] / 1.0e3),
        ("Eccentricity [-]",     lambda c: c["keplerian_states"][:, 2]),
        ("Inclination [deg]",    lambda c: c["keplerian_states"][:, 3] / np.pi * 180.0),
        ("Arg. of perigee [deg]",lambda c: c["keplerian_states"][:, 4] / np.pi * 180.0),
        ("RAAN [deg]",           lambda c: c["keplerian_states"][:, 5] / np.pi * 180.0),
        ("True anomaly [deg]",   lambda c: c["keplerian_states"][:, 6] / np.pi * 180.0),
    ]

    for ax, (ylabel, yfun) in zip(axes, y_defs):
        for case in cases:
            ax.plot(_kepler_time_hours(case), yfun(case), linestyle="-.", label=_label(case))
        ax.set_xlabel("Time [hours since start of TLE]")
        ax.set_ylabel(ylabel)
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    _save(fig, "COMPARISON", "plot_3_compare")
    plt.show()
    plt.close(fig)

    # =========================
    # PLOT 4 - KEPLER DIFF WRT TLE (comparison)
    # =========================
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Difference between propagated and TLE orbital elements (comparison)")

    axes = [
        fig.add_subplot(231), fig.add_subplot(232), fig.add_subplot(233),
        fig.add_subplot(234), fig.add_subplot(235), fig.add_subplot(236)
    ]

    y_defs = [
        ("Δ a [km]",      lambda c: c["keplerian_difference_wrt_tle"][:, 1] / 1.0e3),
        ("Δ e [-]",       lambda c: c["keplerian_difference_wrt_tle"][:, 2]),
        ("Δ i [deg]",     lambda c: c["keplerian_difference_wrt_tle"][:, 3] / np.pi * 180.0),
        ("Δ ω [deg]",     lambda c: c["keplerian_difference_wrt_tle"][:, 4] / np.pi * 180.0),
        ("Δ RAAN [deg]",  lambda c: c["keplerian_difference_wrt_tle"][:, 5] / np.pi * 180.0),
        ("Δ ν [deg]",     lambda c: c["keplerian_difference_wrt_tle"][:, 6] / np.pi * 180.0),
    ]

    for ax, (ylabel, yfun) in zip(axes, y_defs):
        for case in cases:
            ax.plot(_kepler_time_hours(case), yfun(case), linestyle="-.", label=_label(case))
        ax.set_xlabel("Time [hours since start of TLE]")
        ax.set_ylabel(ylabel)
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    _save(fig, "COMPARISON", "plot_4_compare")
    plt.show()
    plt.close(fig)

    # =========================
    # PLOT 5 - RSW DIFF WRT TLE (comparison)
    # =========================
    fig = plt.figure(figsize=(12, 3))
    fig.suptitle("Difference between propagated and TLE orbits in RSW (comparison)")

    axR = fig.add_subplot(131)
    axS = fig.add_subplot(132)
    axW = fig.add_subplot(133)

    for case in cases:
        t_hr = (case["propagation_epochs"] - case["start_recording_day"]) / 3600.0
        axR.plot(t_hr, case["rsw_difference_wrt_tle"][:, 1] / 1.0e3, linestyle="-.", label=_label(case))
        axS.plot(t_hr, case["rsw_difference_wrt_tle"][:, 2] / 1.0e3, linestyle="-.", label=_label(case))
        axW.plot(t_hr, case["rsw_difference_wrt_tle"][:, 3] / 1.0e3, linestyle="-.", label=_label(case))

    axR.set_xlabel("Time [hours]"); axR.set_ylabel("Radial [km]"); axR.grid(True)
    axS.set_xlabel("Time [hours]"); axS.set_ylabel("Along-track [km]"); axS.grid(True)
    axW.set_xlabel("Time [hours]"); axW.set_ylabel("Cross-track [km]"); axW.grid(True)

    handles, labels = axR.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    _save(fig, "COMPARISON", "plot_5_compare")
    plt.show()
    plt.close(fig)

    # =========================
    # PLOT 10 - SIMULATED VS REAL DOPPLER (comparison)
    # =========================
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot()
    ax.set_title("Doppler data (comparison)")

    for case in cases:
        t_sim = (np.asarray(case["simulated_obs_times"]) - case["start_recording_day"]) / 3600.0
        t_real = (np.asarray(case["observation_times"]) - case["start_recording_day"]) / 3600.0

        ax.plot(t_sim, case["simulated_doppler"], linestyle="none", marker=".", label=f"{_label(case)} simulated")
        ax.plot(t_real, case["real_doppler"], linestyle="none", marker=".", label=f"{_label(case)} recorded")

    ax.set_xlabel("Time [hours since start of day]")
    ax.set_ylabel("Doppler [m/s]")
    ax.grid(True)
    ax.legend(fontsize=8)
    _save(fig, "COMPARISON", "plot_10_compare")
    plt.show()
    plt.close(fig)

    # =========================
    # PLOT 11 - SINGLE PASS OBSERVATIONS (comparison)
    # =========================
    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    fig.suptitle("Single pass observations (comparison)")

    ax1 = fig.add_subplot(2, 2, 1)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    for case in cases:
        t = (np.asarray(case["interpolated_times"]) - case["start_recording_day"]) / 3600.0

        ax1.plot(t, case["interpolated_real_obs"][:, 1], linestyle="none", marker=".", label=f"{_label(case)} recorded")
        ax1.plot(t, case["interpolated_simulated_obs"][:, 1], linestyle="none", marker=".", label=f"{_label(case)} simulated")

        ax3.plot(t, case["first_residual_obs"], linestyle="none", marker=".", label=_label(case))
        ax4.plot(t, case["second_residual_obs"], linestyle="none", marker=".", label=_label(case))

    ax1.set_title("Doppler"); ax1.set_xlabel("Time [hours]"); ax1.set_ylabel("Doppler [m/s]"); ax1.grid(True); ax1.legend(fontsize=8)
    ax3.set_title("First residual"); ax3.set_xlabel("Time [hours]"); ax3.set_ylabel("Residual [m/s]"); ax3.grid(True); ax3.legend(fontsize=8)
    ax4.set_title("Second residual"); ax4.set_xlabel("Time [hours]"); ax4.set_ylabel("Residual [m/s]"); ax4.grid(True); ax4.legend(fontsize=8)

    _save(fig, "COMPARISON", "plot_11_compare")
    plt.show()
    plt.close(fig)

    # ============================================================
    # REMAINING PLOTS: ONE PER RUN (one map per run for Plot 6)
    # ============================================================

    # --- Common ground station + horizon geometry (used in 6/7/8) ---
    location_groundstation_lon = 4.3754
    location_groundstation_lat = 51.9899

    R = 6371360.0
    h = 547500.0

    # =========================
    # PLOT 6 - MAP (ONE MAP PER RUN)
    # =========================
    try:
        import cartopy.crs as ccrs
        cartopy_ok = True
    except Exception:
        cartopy_ok = False

    if cartopy_ok:
        Lambda = np.arccos(R / (R + h))
        Phi_E = np.linspace(0, 2 * np.pi, num=1000)

        mask_E = np.where(((-Phi_E) % (2 * np.pi)) < np.pi, 1.0, -1.0)

        colat_horizon = np.arccos(
            np.cos(Lambda) * np.cos((90 - location_groundstation_lat) / 180 * np.pi)
            + np.sin(Lambda) * np.sin((90 - location_groundstation_lat) / 180 * np.pi) * np.cos(Phi_E % (2 * np.pi))
        )

        DL = mask_E * np.arccos(
            (np.cos(Lambda) - np.cos(colat_horizon) * np.cos((90 - location_groundstation_lat) / 180 * np.pi))
            / (np.sin((90 - location_groundstation_lat) / 180 * np.pi) * np.sin(colat_horizon))
        )

        latitude_horizon = 90 - (colat_horizon / np.pi * 180)
        longitude_horizon_abs = (location_groundstation_lon / 180 * np.pi - DL) / np.pi * 180
        longitude_horizon = np.where(longitude_horizon_abs <= 180, longitude_horizon_abs, longitude_horizon_abs - 360)

        for case in cases:
            fig = plt.figure(figsize=(12, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines()

            ax.scatter(case["longitudes"] / np.pi * 180,
                       case["latitudes"] / np.pi * 180,
                       marker=".", s=2, label=_label(case))

            ax.scatter(location_groundstation_lon, location_groundstation_lat,
                       color="red", marker="*", s=200, label="Ground station")

            ax.plot(longitude_horizon, latitude_horizon, color="red", linewidth=1.5, label="Visibility horizon")

            ax.gridlines(draw_labels=True)
            ax.legend(loc="lower left")
            plt.title(f"Ground track and visibility ({_label(case)})")

            _save(fig, case.get("run_id", "UNKNOWN"), "plot_6")
            plt.show()
            plt.close(fig)
    else:
        print("Cartopy not installed -> skipping Plot 6 (maps).")

    # =========================
    # PLOT 7 - ELEVATION (ONE FIGURE PER RUN)
    # =========================
    for case in cases:
        sin_rho = R / (R + h)
        DL_e = np.deg2rad(np.rad2deg(case["longitudes"][:, 1]) - location_groundstation_lon)
        Lambda_e = np.arccos(
            np.cos(case["latitudes"][:, 1]) * np.cos(np.deg2rad(location_groundstation_lat))
            + np.sin(case["latitudes"][:, 1]) * np.sin(np.deg2rad(location_groundstation_lat)) * np.cos(DL_e)
        )

        eta = np.arctan2(sin_rho * np.sin(Lambda_e), 1 - sin_rho * np.cos(Lambda_e))
        elevation_abs = np.rad2deg(np.arccos(np.sin(eta) / sin_rho))
        elevation_lambda_check = np.where(Lambda_e <= np.arccos(R / (R + h)), elevation_abs, 0)
        elevation = np.where(np.abs(DL_e) <= 0.5 * np.pi, elevation_lambda_check, 0)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.set_title(f"Elevation of passes ({_label(case)})")
        ax.plot((case["longitudes"][:, 0] - case["start_recording_day"]) / 3600.0, elevation)
        ax.set_xlabel("Time [hours since start of day]")
        ax.set_ylabel("Elevation [deg]")
        ax.grid(True)
        _save(fig, case.get("run_id", "UNKNOWN"), "plot_7")
        plt.show()
        plt.close(fig)

    # =========================
    # PLOT 8 - AZIMUTH (ONE FIGURE PER RUN)
    # =========================
    for case in cases:
        DL_e = np.deg2rad(np.rad2deg(case["longitudes"][:, 1]) - location_groundstation_lon)
        Lambda_e = np.arccos(
            np.cos(case["latitudes"][:, 1]) * np.cos(np.deg2rad(location_groundstation_lat))
            + np.sin(case["latitudes"][:, 1]) * np.sin(np.deg2rad(location_groundstation_lat)) * np.cos(DL_e)
        )

        mask_DL = np.where((DL_e % (2 * np.pi)) < np.pi, 1.0, -1.0)

        azimuth_abs = np.rad2deg(
            mask_DL
            * np.arccos(
                (np.cos(np.deg2rad(location_groundstation_lat)) - np.cos(case["latitudes"][:, 1]) * np.cos(Lambda_e))
                / (np.sin(case["latitudes"][:, 1]) * np.sin(Lambda_e))
            )
        )
        azimuth_lambda_check = np.where(Lambda_e <= np.arccos(R / (R + h)), azimuth_abs, 0)
        azimuth = np.where(np.abs(DL_e) <= 0.5 * np.pi, azimuth_lambda_check, 0)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.set_title(f"Azimuth of passes ({_label(case)})")
        ax.plot((case["longitudes"][:, 0] - case["start_recording_day"]) / 3600.0, azimuth % 360)
        ax.set_xlabel("Time [hours since start of day]")
        ax.set_ylabel("Azimuth [deg]")
        ax.grid(True)
        _save(fig, case.get("run_id", "UNKNOWN"), "plot_8")
        plt.show()
        plt.close(fig)

    # =========================
    # PLOT 9 - SIMULATED DOPPLER (ONE FIGURE PER RUN)
    # =========================
    for case in cases:
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        ax.set_title(f"Simulated Doppler data ({_label(case)})")
        ax.plot((case["simulated_obs_times"] - case["start_recording_day"]) / 3600.0,
                case["simulated_doppler"],
                linestyle="none", marker=".")
        ax.set_xlabel("Time [hours since start of day]")
        ax.set_ylabel("Doppler [m/s]")
        ax.grid(True)
        _save(fig, case.get("run_id", "UNKNOWN"), "plot_9")
        plt.show()
        plt.close(fig)

    # =========================
    # SIMULATED RECEIVED FREQUENCY (ONE FIGURE PER RUN)
    # =========================
    satellite_frequency = 145_870_000.0
    speed_of_light = 299_792_458.0

    for case in cases:
        frequency_received = (1 - case["simulated_doppler"] / speed_of_light) * satellite_frequency

        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        ax.set_title(f"Simulated received frequency ({_label(case)})")
        ax.plot((case["simulated_obs_times"] - case["start_recording_day"]) / 3600.0,
                frequency_received,
                linestyle="none", marker=".")
        ax.set_xlabel("Time [hours since start of day]")
        ax.set_ylabel("Radio frequency [Hz]")
        ax.grid(True)
        _save(fig, case.get("run_id", "UNKNOWN"), "plot_10_frequency")
        plt.show()
        plt.close(fig)