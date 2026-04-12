"""
estimation_utils.py — run estimation, save plots, summary table.
Plots are saved to disk, never displayed interactively.
"""

import sys
import os
import math
import statistics
import numpy as np
from matplotlib import pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))
for _p in [
    _here,
    os.path.abspath(os.path.join(_here, '..')),
    os.path.abspath(os.path.join(_here, '../..')),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from estimation_functions.estimation import (
    run_estimation, get_residuals_per_pass
)
from tudatpy.astro import element_conversion, frame_conversion
from propagation_functions.environment import define_environment
from propagation_functions.propagation import propagate_initial_state
from shared_config import MASS, REF_AREA, DRAG_COEF, SRP_COEF, SAT_NAME


def compute_rsw_rms(built, updated_params, arc_index=0):
    """
    Propagate estimated and TLE orbits for one arc and return
    RMS of RSW position components (R, S, W) in km.
    """
    arc_start_times = built['arc_start_times']
    arc_end_times   = built['arc_end_times']
    arc_wise_init   = built['arc_wise_initial_states']
    accelerations   = built['accelerations']

    bodies = define_environment(MASS, REF_AREA, DRAG_COEF, SRP_COEF,
                                SAT_NAME, multi_arc_ephemeris=False)

    est_state = updated_params[6*arc_index:(arc_index+1)*6]
    tle_state = arc_wise_init[arc_index]

    est_orbit = propagate_initial_state(est_state, arc_start_times[arc_index],
                                        arc_end_times[arc_index], bodies,
                                        accelerations, SAT_NAME)[0]
    tle_orbit = propagate_initial_state(tle_state, arc_start_times[arc_index],
                                        arc_end_times[arc_index], bodies,
                                        accelerations, SAT_NAME)[0]

    n = len(tle_orbit[:, 0])
    rsw = np.zeros((n, 3))
    for i in range(n):
        tle_s = tle_orbit[i, 1:]
        est_s = est_orbit[i, 1:]
        diff  = est_s - tle_s
        R     = frame_conversion.inertial_to_rsw_rotation_matrix(tle_s)
        rsw[i, :] = R @ diff[:3]

    rms_R = float(np.sqrt(np.mean(rsw[:, 0]**2))) / 1000.0
    rms_S = float(np.sqrt(np.mean(rsw[:, 1]**2))) / 1000.0
    rms_W = float(np.sqrt(np.mean(rsw[:, 2]**2))) / 1000.0

    return rms_R, rms_S, rms_W


def run_task(built, nb_iterations=10, label=""):
    estimator          = built['estimator']
    parameters_to_est  = built['parameters_to_estimate']
    observations_set   = built['observations_set']
    nb_arcs            = built['nb_arcs']
    observation_times  = built['observation_times']
    passes_start_times = built['passes_start_times']

    pod_output    = run_estimation(estimator, parameters_to_est,
                                   observations_set, nb_arcs, nb_iterations)
    residuals     = pod_output.residual_history
    formal_errors = pod_output.formal_errors
    final_res     = residuals[:, nb_iterations - 1]
    final_rms     = float(np.sqrt(np.mean(final_res**2)))

    residuals_per_pass = get_residuals_per_pass(
        observation_times, residuals, passes_start_times)

    updated_params    = parameters_to_est.parameter_vector
    arc_wise_init     = built['arc_wise_initial_states']
    arc_tle_distances = []
    for arc in range(nb_arcs):
        est  = updated_params[arc*6:(arc+1)*6]
        tle  = arc_wise_init[arc]
        dist = float(np.sqrt(np.sum((est[:3] - tle[:3])**2))) / 1000.0
        arc_tle_distances.append(dist)

    try:
        rms_R, rms_S, rms_W = compute_rsw_rms(built, updated_params, arc_index=0)
    except Exception as e:
        print(f"  [warning] RSW RMS computation failed: {e}")
        rms_R, rms_S, rms_W = None, None, None

    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  Final RMS residual : {final_rms:.4f} m/s")
        print(f"  Mean TLE distance  : {np.mean(arc_tle_distances):.2f} km")
        if rms_R is not None:
            print(f"  RSW RMS (arc 0)    : R={rms_R:.3f} km  S={rms_S:.3f} km  W={rms_W:.3f} km")
        print(f"{'='*60}")

    return dict(
        label=label,
        final_rms=final_rms,
        mean_residual=statistics.mean(final_res),
        std_residual=statistics.stdev(final_res),
        residuals=residuals,
        residuals_per_pass=residuals_per_pass,
        updated_parameters=updated_params,
        formal_errors=formal_errors,
        arc_tle_distances=arc_tle_distances,
        rsw_rms=(rms_R, rms_S, rms_W),
        nb_iterations=nb_iterations,
    )


def save_residuals(result, built, fig_dir):
    rpp = result['residuals_per_pass']
    n   = len(rpp)
    fig, axs = plt.subplots(math.ceil(n / 3), 3, figsize=(12, 8))
    for i, r in enumerate(rpp):
        ax = axs.flatten()[i] if hasattr(axs, "flatten") else axs[i]
        ax.plot(r, color='blue', linestyle='-.')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Residuals [m/s]')
        ax.set_title(f'Pass {i+1}')
        ax.grid()
    fig.suptitle(result.get('label', ''), fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'residuals.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_residual_histogram(result, fig_dir):
    final_res = result['residuals'][:, result['nb_iterations'] - 1]
    fig, ax = plt.subplots()
    ax.hist(final_res, bins=100)
    ax.set_xlabel('Doppler residuals [m/s]')
    ax.set_ylabel('Nb occurrences')
    ax.set_title(result.get('label', ''))
    ax.grid()
    fig.savefig(os.path.join(fig_dir, 'residuals_histogram.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_rsw_keplerian(result, built, fig_dir, arc_index=0):
    updated_params  = result['updated_parameters']
    arc_start_times = built['arc_start_times']
    arc_end_times   = built['arc_end_times']
    arc_wise_init   = built['arc_wise_initial_states']
    accelerations   = built['accelerations']

    bodies = define_environment(MASS, REF_AREA, DRAG_COEF, SRP_COEF,
                                SAT_NAME, multi_arc_ephemeris=False)
    est_state = updated_params[6*arc_index:(arc_index+1)*6]
    tle_state = arc_wise_init[arc_index]

    est_orbit = propagate_initial_state(est_state, arc_start_times[arc_index],
                                        arc_end_times[arc_index], bodies,
                                        accelerations, SAT_NAME)[0]
    tle_orbit = propagate_initial_state(tle_state, arc_start_times[arc_index],
                                        arc_end_times[arc_index], bodies,
                                        accelerations, SAT_NAME)[0]

    t  = tle_orbit[:, 0] - tle_orbit[0, 0]
    mu = bodies.get("Earth").gravitational_parameter

    rsw = np.zeros((len(t), 7))
    kep = np.zeros((len(t), 7))
    for i in range(len(t)):
        tle_s = tle_orbit[i, 1:]
        est_s = est_orbit[i, 1:]
        diff  = est_s - tle_s
        R     = frame_conversion.inertial_to_rsw_rotation_matrix(tle_s)
        rsw[i, 1:4] = R @ diff[:3]
        rsw[i, 4:7] = R @ diff[3:]
        kep[i, 1:7] = (element_conversion.cartesian_to_keplerian(est_s, mu)
                       - element_conversion.cartesian_to_keplerian(tle_s, mu))

    # RSW plot
    fig, axs = plt.subplots(3, 2, figsize=(10, 8))
    rsw_labels = ['Diff R [km]','Diff S [km]','Diff W [km]',
                  'Diff Vr [km/s]','Diff Vs [km/s]','Diff Vw [km/s]']
    for row, (ci, cj) in enumerate([(1,4),(2,5),(3,6)]):
        axs[row,0].plot(t, rsw[:,ci]/1000, color='blue', linestyle='-.')
        axs[row,0].set_ylabel(rsw_labels[row]); axs[row,0].grid()
        axs[row,1].plot(t, rsw[:,cj]/1000, color='blue', linestyle='-.')
        axs[row,1].set_ylabel(rsw_labels[row+3]); axs[row,1].grid()
    axs[0,0].set_title(f'Arc {arc_index+1} RSW — {result.get("label","")}')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'rsw.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Keplerian plot
    fig2, axs2 = plt.subplots(3, 2, figsize=(10, 8))
    kep_labels = [r'$\Delta a$ [km]', r'$\Delta e$ [-]', r'$\Delta i$ [deg]',
                  r'$\Delta\omega$ [deg]', r'$\Delta\Omega$ [deg]', r'$\Delta\theta$ [deg]']
    scales     = [1/1000, 1, np.degrees, np.degrees, np.degrees, np.degrees]
    positions  = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)]
    for idx, (r, c) in enumerate(positions):
        vals = scales[idx](kep[:,idx+1]) if callable(scales[idx]) else kep[:,idx+1]*scales[idx]
        axs2[r,c].plot(t, vals, color='blue', linestyle='-.')
        axs2[r,c].set_ylabel(kep_labels[idx]); axs2[r,c].grid()
    axs2[0,0].set_title(f'Arc {arc_index+1} Keplerian — {result.get("label","")}')
    fig2.tight_layout()
    fig2.savefig(os.path.join(fig_dir, 'keplerian.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)


def summary_table(results_list):
    print(f"\n{'Label':<35} {'RMS [m/s]':>12} {'TLE [km]':>10} {'dR [km]':>9} {'dS [km]':>9} {'dW [km]':>9}")
    print('-' * 87)
    for r in results_list:
        mean_d = float(np.mean(r['arc_tle_distances']))
        rR, rS, rW = r.get('rsw_rms', (None, None, None))
        rR_s = f"{rR:>9.3f}" if rR is not None else "       --"
        rS_s = f"{rS:>9.3f}" if rS is not None else "       --"
        rW_s = f"{rW:>9.3f}" if rW is not None else "       --"
        print(f"{r['label']:<35} {r['final_rms']:>12.4f} {mean_d:>10.3f} {rR_s} {rS_s} {rW_s}")


def save_summary_bar_chart(results_list, fig_dir, title="Summary"):
    """
    Bar chart comparing RMS and TLE distance across all configurations.
    Saves two side-by-side subplots: RMS [m/s] and TLE distance [km].
    Skips point-mass case in the bar chart if RMS > 1000 (diverged).
    """
    os.makedirs(fig_dir, exist_ok=True)

    filtered = [r for r in results_list if r['final_rms'] < 1000]
    labels   = [r['label'] for r in filtered]
    rms_vals = [r['final_rms'] for r in filtered]
    tle_vals = [float(np.mean(r['arc_tle_distances'])) for r in filtered]

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(x, rms_vals, color='steelblue', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax1.set_ylabel('Final RMS residual [m/s]')
    ax1.set_title('RMS Residual')
    ax1.grid(axis='y', alpha=0.4)

    ax2.bar(x, tle_vals, color='tomato', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax2.set_ylabel('Mean TLE distance [km]')
    ax2.set_title('TLE Distance')
    ax2.grid(axis='y', alpha=0.4)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'summary_bar.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_rsw_bar_chart(results_list, fig_dir, title="RSW RMS Components"):
    """
    Grouped bar chart of RSW RMS components (dR, dS, dW) across configurations.
    Skips diverged cases.
    """
    os.makedirs(fig_dir, exist_ok=True)

    filtered = [r for r in results_list
                if r.get('rsw_rms', (None,None,None))[0] is not None
                and r['final_rms'] < 1000
                and float(np.mean(r['arc_tle_distances'])) < 500]

    labels = [r['label'] for r in filtered]
    dR = [r['rsw_rms'][0] for r in filtered]
    dS = [r['rsw_rms'][1] for r in filtered]
    dW = [r['rsw_rms'][2] for r in filtered]

    x     = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, dR, width, label=r'$\Delta R$', color='steelblue',  alpha=0.8)
    ax.bar(x,         dS, width, label=r'$\Delta S$', color='tomato',     alpha=0.8)
    ax.bar(x + width, dW, width, label=r'$\Delta W$', color='seagreen',   alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('RSW RMS [km]')
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='y', alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'rsw_bar.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)