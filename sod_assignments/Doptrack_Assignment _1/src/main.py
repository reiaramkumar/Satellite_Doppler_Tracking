import os
import sys
import numpy as np
import pandas as pd
sys.path.append("../../")

# Load standard modules
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression  # linear regression module

# Import doptrack-estimate functions
from propagation_functions.environment import *
from propagation_functions.propagation import *
from utility_functions.time import *
from utility_functions.tle import *
from utility_functions.data import extract_tar
from estimation_functions.observations_data import *
from estimation_functions.estimation import *

# Load tudatpy modules
from tudatpy import constants
from tudatpy.interface import spice
from tudatpy.astro import element_conversion, frame_conversion
from tudatpy.dynamics import environment
import propagation_runner as pr
import plotting as pt
from pathlib import Path


def save_case_data(case):
    run_id = case['run_id']
    os.makedirs('case_data', exist_ok = True)
    np.savez_compressed(
        f'case_data/{run_id}.npz',
        propagation_epochs = case['propagation_epochs'],
        cartesian_states = case['cartesian_states'],
        keplerian_states = case['keplerian_states'],
        rsw_difference_wrt_tle = case['rsw_difference_wrt_tle'],
        keplerian_difference_wrt_tle = case['keplerian_difference_wrt_tle'],
        saved_accelerations = case['saved_accelerations'],
        accelerations_ids = case['accelerations_ids'],
    )


def case_summary_for_comp_table(case):

    def rms(value, axis = 0):
        return np.sqrt(np.mean(np.square(value), axis = axis))

    kep = case['keplerian_difference_wrt_tle']
    rsw = case['rsw_difference_wrt_tle']

    kep_values = kep[:,1:7]
    rsw_pos = rsw[:,1:4]
    rsw_vel = rsw[:,4:7]

    return {
        'run_id': case['run_id'],
        'rms_da_m': rms(kep_values[:,0]),
        'rms_de_nil': rms(kep_values[:,1]),
        'rms_di_rad': rms(kep_values[:,2]),
        'rms_dw_rad': rms(kep_values[:,3]),
        'rms_draan_rad': rms(kep_values[:,4]),
        'rms_dnu_rad': rms(kep_values[:,5]),

        'rms_dR_m': rms(rsw_pos[:,0]),
        'rms_dS_m': rms(rsw_pos[:,1]),
        'rms_dW_m': rms(rsw_pos[:,2]),

        'rms_dVR_ms': rms(rsw_vel[:, 0]),
        'rms_dVS_ms': rms(rsw_vel[:, 1]),
        'rms_dVW_ms': rms(rsw_vel[:, 2]),

        'rms_RSW_pos_m': rms(np.linalg.norm(rsw_pos, axis=1)),
        'rms_RSW_vel_ms': rms(np.linalg.norm(rsw_vel, axis=1)),


    }





# Extracting Data
HERE = Path(__file__).resolve().parent      # src/
HERE = HERE.parent                         # Doptrack_Assignment_1/
ROOT = HERE.parent

extract_tar(str(ROOT / "metadata.tar.xz"))
extract_tar(str(ROOT / "data.tar.xz"))

# Import Folder Paths
metadata_folder = 'metadata/'
data_folder = 'data/'


#  TLE Data
delfi_tle = environment.Tle("1 32789U 08021G   20092.14603172 +.00001512 +00000-0 +10336-3 0  9992",
                       "2 32789 097.4277 137.6209 0011263 214.0075 146.0432 15.07555919650162")

# Sensitivity Analysis
baseline_case = pr.propagate(delfi_tle)
baseline_case["run_id"] = "00_BASE"

# Gravitational Acceleration Cases
grav_acc_pm_case = pr.propagate(delfi_tle, grav_acc_type = 'pm')
grav_acc_pm_case["run_id"] = "11_G_PM"

grav_acc_sph_case = baseline_case.copy()                                       # --> as propagation(delfi_tle, grav_acc_type = 'sph') is the base case
grav_acc_sph_case["run_id"] = "12_G_SPH"
print('done')

# Atmospheric Drag Cases
atm_drag_on_case = baseline_case.copy()                                        # --> as propagation(delfi_tle, atm_drag_on = True) is the base case
atm_drag_on_case["run_id"] = "21_D_ON"

atm_drag_off_case = pr.propagate(delfi_tle, atm_drag_on = False)
atm_drag_off_case["run_id"] = "22_D_OFF"

atm_drag_10_case = pr.propagate(delfi_tle, drag_coef = (1.1 * 1.2))
atm_drag_10_case["run_id"] = "23_D_CD110"

atm_drag_50_case = pr.propagate(delfi_tle, drag_coef = (1.5 * 1.2))
atm_drag_50_case["run_id"] = "24_D_CD150"

atm_drag_100_case = pr.propagate(delfi_tle, drag_coef = (2.0 * 1.2))
atm_drag_100_case["run_id"] = "25_D_CD200"
print('done')


# Third Body Acceleration Cases
tbp_all_case = baseline_case.copy()                                            # --> as propagation(delfi_tle, tbp_type = 'all') is the base case
tbp_all_case["run_id"] = "31_TBP_ALL"

tbp_none_case = pr.propagate(delfi_tle, tbp_type = 'none')
tbp_none_case["run_id"] = "32_TBP_NONE"

tbp_moon_sun_case = pr.propagate(delfi_tle, tbp_type = 'moon_sun')
tbp_moon_sun_case["run_id"] = "33_TBP_MS"

tbp_moon_case = pr.propagate(delfi_tle, tbp_type = 'moon')
tbp_moon_case["run_id"] = "33_TBP_M"
print('done')


# Solar Radiation Pressure Cases
srp_on_case = baseline_case.copy()                                             # --> as propagation(delfi_tle, srp_on = True) is the base case
srp_on_case["run_id"] = "41_SRP_ON"

srp_off_case = pr.propagate(delfi_tle, srp_on = False)
srp_off_case["run_id"] = "42_SRP_OFF"

srp_10_case = pr.propagate(delfi_tle, srp_coef = (1.1 * 1.2))
srp_10_case["run_id"] = "43_SRP110"

srp_50_case = pr.propagate(delfi_tle, srp_coef = (1.5 * 1.2))
srp_50_case["run_id"] = "44_SRP150"

srp_100_case = pr.propagate(delfi_tle, srp_coef = (2.0 * 1.2))
srp_100_case["run_id"] = "45_SRP200"
print('done')

# Sensitivity Analysis Report
run_cases = [baseline_case, grav_acc_pm_case, grav_acc_sph_case, atm_drag_on_case,atm_drag_off_case, atm_drag_10_case, atm_drag_50_case, atm_drag_100_case, tbp_all_case, tbp_none_case, tbp_moon_sun_case, tbp_moon_case, srp_on_case, srp_off_case, srp_10_case, srp_50_case, srp_100_case]

sensitivity_table_rows = []

for case in run_cases:
    print('current')
    case['rsw_difference_wrt_tle'], case['keplerian_difference_wrt_tle'] = pr.residuals_rsw_and_kep(case['propagation_epochs'],
                                                                                                    case['delfi_ephemeris'],
                                                                                                    case['cartesian_states'],
                                                                                                    case['keplerian_states'],
                                                                                                    case['mu_earth'])

    dop = pr.doppler_simulation(
        bodies=case["bodies"],
        accelerations=case["accelerations_dict"],
        initial_state=case["initial_state"],
        initial_epoch=case["initial_epoch"],
        final_epoch=case["final_epoch"],
        start_recording_day=case["start_recording_day"],
        metadata_folder=metadata_folder,
        data_folder=data_folder,
        index_pass=0
    )
    print('2 - done')

    case.update(dop)
    save_case_data(case)
    pt.plots(case)
    sensitivity_table_rows.append(case_summary_for_comp_table(case))

df = pd.DataFrame.from_dict(sensitivity_table_rows)
os.makedirs('results', exist_ok = True)
df.to_csv('results/sensitivity_table.csv')


cases_dict = {
    # Baseline
    "00_BASE": baseline_case,

    # Gravity
    "11_G_PM": grav_acc_pm_case,

    # Atmospheric Drag
    "22_D_OFF": atm_drag_off_case,
    "23_D_CD110": atm_drag_10_case,
    "24_D_CD150": atm_drag_50_case,
    "25_D_CD200": atm_drag_100_case,

    # Third Body Perturbations
    "32_TBP_NONE": tbp_none_case,
    "33_TBP_MS": tbp_moon_sun_case,
    "34_TBP_M": tbp_moon_case,

    # Solar Radiation Pressure
    "42_SRP_OFF": srp_off_case,
    "43_SRP110": srp_10_case,
    "44_SRP150": srp_50_case,
    "45_SRP200": srp_100_case,
}


# %%
# Gravity
pt.comparison_plots(cases_dict, ["00_BASE", "11_G_PM"])

# Drag ON/OFF
pt.comparison_plots(cases_dict, ["00_BASE", "22_D_OFF"])

# Drag sensitivity
pt.comparison_plots(cases_dict, ["00_BASE", "23_D_CD110", "24_D_CD150", "25_D_CD200"])

# Third body ON/OFF
pt.comparison_plots(cases_dict, ["00_BASE", "32_TBP_NONE"])

# Third body breakdown
pt.comparison_plots(cases_dict, ["00_BASE", "33_TBP_MS", "34_TBP_M"])

# SRP ON/OFF
pt.comparison_plots(cases_dict, ["00_BASE", "42_SRP_OFF"])

# SRP sensitivity
pt.comparison_plots(cases_dict, ["00_BASE", "43_SRP110", "44_SRP150", "45_SRP200"])
# %%


