# ... IMPORTS ...
### IMPORT STATEMEMTS

import sys

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



def propagate(delfi_tle,
                mass = 2.2,
                ref_area = (4 * 0.3 * 0.1 + 2 * 0.1 * 0.1) / 4 ,
                srp_coef = 1.2,
                drag_coef = 1.2,
                *,
                grav_acc_type = 'sph',          # options --> 'sph', 'pm'
                atm_drag_on = True,
                tbp_type = 'all',               # options --> 'all', 'none', 'moon', 'moon_sun'
                srp_on = True):
    """
    Propagate spacecraft orbit with configurable dynamical model.

    Parameters
    ----------
    mass : float, optional      Spacecraft mass in kilograms. Default is 2.2 kg.
    ref_area : float, optional
        Reference cross-sectional area in square meters.
        Default corresponds to average projection area of a 3U CubeSat.
    srp_coef : float, optional      Solar radiation pressure coefficient.
    drag_coef : float, optional
        Atmospheric drag coefficient.
    grav_acc_on : bool, optional
        If True, Earth's gravitational acceleration is included.
    grav_acc_type : {"sph", "pm"}, optional
        Earth gravity model selection:
        - "sph": spherical harmonic gravity
        - "pm": point-mass gravity
    atm_drag_on : bool, optional
        If True, atmospheric drag is included.
    tbp_on : bool, optional
        If True, third-body perturbations are included.
    tbp_type : {"all", "none", "moon", "moon_sun"}, optional
        Third-body configuration:
        - "all": all third bodies (Sun, Moon, planets)
        - "none": no third-body perturbations
        - "moon": Moon only
        - "moon_sun": Moon and Sun only
    srp_on : bool, optional
        If True, solar radiation pressure is included.

    Returns
    -------
    dict
        Dictionary containing propagated state histories and dependent variables.
    """

    cfg = {
        "grav_acc_type": grav_acc_type,
        "atm_drag_on": atm_drag_on,
        "drag_coef": drag_coef,
        "tbp_type": tbp_type,
        "srp_on": srp_on,
        "srp_coef": srp_coef
    }

    # Epoch retrieval
    initial_epoch = delfi_tle.get_epoch()
    start_recording_day = get_start_next_day(initial_epoch)
    propagation_time = 1.0 * constants.JULIAN_DAY
    final_epoch = start_recording_day + propagation_time

    # Initial state retrieval
    delfi_ephemeris = environment.TleEphemeris("Earth", "J2000", delfi_tle, False)
    mid_epoch = (initial_epoch + final_epoch) / 2.0
    initial_state = delfi_ephemeris.cartesian_state(mid_epoch)

    bodies = define_environment(mass, ref_area, drag_coef, srp_coef, "Delfi")
    mu_earth = bodies.get("Earth").gravitational_parameter

    # Accelerations
    # ... TBP choices ...
    sun_on = (tbp_type in ['all', 'moon_sun'])
    moon_on = (tbp_type in ['all', 'moon', 'moon_sun'])
    planets_on = (tbp_type == 'all')

    # ... Gravity choices ...
    earth_pm = (grav_acc_type == 'pm')
    earth_sph = (grav_acc_type == 'sph')

    accelerations = dict(
        Sun={
            'point_mass_gravity': sun_on,
            'solar_radiation_pressure': srp_on
        },
        Moon={
            'point_mass_gravity': moon_on
        },
        Earth={
            'point_mass_gravity': earth_pm,
            'spherical_harmonic_gravity': earth_sph,
            'drag': atm_drag_on
        },
        Venus={
            'point_mass_gravity': planets_on
        },
        Mars={
            'point_mass_gravity': planets_on
        },
        Jupiter={
            'point_mass_gravity': planets_on
        }
    )

    # The propagation output is given in cartesian and keplerian states, and the latitude/longitude of the spacecraft are also saved.
    cartesian_states, keplerian_states, latitudes, longitudes, saved_accelerations = \
        propagate_initial_state(initial_state, initial_epoch, final_epoch, bodies, accelerations, "Delfi", True)

    # Retrieve accelerations
    accelerations_to_save, accelerations_ids = retrieve_accelerations_to_save(accelerations, "Delfi")

    # Retrieve propagation epochs (in seconds since J2000)
    propagation_epochs = cartesian_states[:, 0]

    return {
        'cfg': cfg,

        'initial_epoch': initial_epoch,
        'start_recording_day': start_recording_day,
        'propagation_time': propagation_time,
        'final_epoch': final_epoch,
        'propagation_epochs': propagation_epochs,
        'delfi_ephemeris': delfi_ephemeris,

        'initial_state': initial_state,
        'cartesian_states': cartesian_states,
        'keplerian_states': keplerian_states,
        'latitudes': latitudes,
        'longitudes': longitudes,

        'saved_accelerations': saved_accelerations,
        'accelerations_ids': accelerations_ids,
        'mu_earth': mu_earth,
        'bodies': bodies,
        'accelerations_dict': accelerations,


    }

### COMPUTE DIFFERENCE BETWEEN PROPAGATED ORBIT AND REFERENCE TLE EPHEMERIS


def residuals_rsw_and_kep(propagation_epochs,delfi_ephemeris,cartesian_states, keplerian_states, mu_earth):

    rsw_difference_wrt_tle = np.zeros((len(propagation_epochs), 7))
    keplerian_difference_wrt_tle = np.zeros((len(propagation_epochs), 7))

    # Parse all epochs in propagated state history
    for i in range(len(propagation_epochs)):

        current_epoch = propagation_epochs[i]
        rsw_difference_wrt_tle[i, 0] = current_epoch
        keplerian_difference_wrt_tle[i, 0] = current_epoch

        # Retrieve current TLE and propagated states
        current_tle_state = delfi_ephemeris.cartesian_state(current_epoch)
        current_propagated_state = cartesian_states[i, 1:7]

        # Compute difference in the inertial frame
        current_state_difference = current_propagated_state - current_tle_state
        current_position_difference = current_state_difference[0:3]
        current_velocity_difference = current_state_difference[3:6]

        # Compute the rotation matrix from inertial to RSW frames
        rotation_to_rsw = frame_conversion.inertial_to_rsw_rotation_matrix(current_tle_state)

        # Convert the state difference from inertial to RSW frames
        rsw_difference_wrt_tle[i, 1:4] = rotation_to_rsw @ current_position_difference
        rsw_difference_wrt_tle[i, 4:7] = rotation_to_rsw @ current_velocity_difference

        # Compute reference orbital elements from TLE ephemeris
        current_tle_keplerian = element_conversion.cartesian_to_keplerian(current_tle_state,
                                                                          mu_earth)

        # Compute difference in orbital elements
        keplerian_difference_wrt_tle[i, 1:7] = keplerian_states[i, 1:7] - current_tle_keplerian

    return rsw_difference_wrt_tle, keplerian_difference_wrt_tle
    # [epoch, x, y, z, v_x, v_y, v_z], [epoch, a, e, i, ω, Ω, ν]




def doppler_simulation(*,
                     bodies,
                     accelerations,
                     initial_state,
                     initial_epoch,
                     final_epoch,
                     start_recording_day,
                     metadata_folder,
                     data_folder,
                     metadata=None,
                     data=None,
                     obs_time_step=10.0,
                     index_pass=0):

    ### SIMULATE DOPPLER MEASUREMENTS

    # The predicted orbit and visibility information is used to simulate range-rate observations at the times of the predicted pass of the satellite over Delft. With the (simplified)
    # Doppler equation, the received radio frequency can be calculated from the simulated range-rate.

    # Create the DopTrack station
    define_doptrack_station(bodies)

    # Define observation settings
    observation_settings = define_ideal_doppler_settings(["DopTrackStation"], "Delfi")

    # Create list of observation times, with one Doppler measurement every 10 seconds
    possible_obs_times = []
    obs_time_step = 10.0
    current_time = start_recording_day
    while current_time < final_epoch:
        possible_obs_times.append(current_time)
        current_time = current_time + obs_time_step

    # Simulate (ideal) observations
    propagator_settings = create_propagator_settings(initial_state, initial_epoch, final_epoch, bodies, accelerations,
                                                     "Delfi")
    simulated_observations = simulate_observations("Delfi", possible_obs_times, observation_settings,
                                                   propagator_settings, bodies, initial_epoch, 0)

    simulated_obs_times = np.array(simulated_observations.concatenated_times)
    simulated_doppler = simulated_observations.concatenated_observations

    # Observation files to be uploaded
    metadata = ['Delfi-C3_32789_202004021953.yml', 'Delfi-C3_32789_202004022126.yml']
    data = ['Delfi-C3_32789_202004021953.csv', 'Delfi-C3_32789_202004022126.csv']

    # Compute recording start times
    recording_start_times = extract_recording_start_times_yml(metadata_folder, metadata, old_yml=False)

    # Process observations.
    # This loads the recorded observations and retrieve the start of each tracking pass
    passes_start_times, passes_end_times, observation_times, observations_set = load_and_format_observations(
        "Delfi", data_folder, data, recording_start_times, old_obs_format=False)

    # Retrieve measured Doppler values
    real_doppler = observations_set.concatenated_observations

    ###############################################################
    ### STEP 4. COMPARE SIMULATED AND RECORDED DATA FOR SINGLE PASS
    ###############################################################

    # From the previous figure it seems that the predicted model is able to capture the observed data quite accurate. To determine how accurate we select a single pass
    # with `index_pass` parameter and inspect this data.
    # In the DopTrack terminology, the difference between the observed range-rate and the predicted range-rate is called the first residual.
    # Index of the *recorded* pass of interest (warning: the number of recorded passes might differ from the number of simulated passes)

    index_pass = 0
    single_pass_start_time = passes_start_times[index_pass]
    single_pass_end_time = passes_end_times[index_pass]

    # Retrieve recorded Doppler data for single pass
    real_obs_single_pass = get_observations_single_pass(single_pass_start_time, single_pass_end_time, observations_set)

    # Retrieve simulated Doppler data for single pass
    simulated_obs_single_pass = get_observations_single_pass(single_pass_start_time, single_pass_end_time,
                                                             simulated_observations)

    # Interpolate simulated and recorded observations to identical times
    interpolated_simulated_obs, interpolated_real_obs = interpolate_obs(simulated_obs_single_pass, real_obs_single_pass)
    interpolated_times = interpolated_simulated_obs[:, 0]

    # Compute first residual between recorded and simulated observations
    first_residual_obs = interpolated_real_obs[:, 1] - interpolated_simulated_obs[:, 1]

    ### REMOVE LINEAR DRIFT FROM DOPPLER DATA

    # Over time we have found that in the Delfi-C3 data there is in many passes a linear drift in the frequency data, with respect to the predicted range-rate. We hypothesised that
    # there is a linear drift in the onboard oscillator or an other system that is responsible for a linear drift in the transmitted frequency. As a first data anaylysis, a linear drift
    # is fitted through the first residual. The resulting residual is called the second residual, which should be free of any onboard clock drifts.
    # Some background: To test this hypothesis we have observed over the years different satellites. One of them is the Nayif-1, which has a much better onboard clock and radio, less
    # affected by drifts. We have found that the Nayif-1 data has better quality and is less pronounced by clock drift. This partly confirms our hypothesis, but much is still unclear
    # and need to be studied.

    # Perform linear regression on first residual
    linear_fit = LinearRegression().fit(interpolated_times.reshape((-1, 1)), first_residual_obs)

    # Retrieve fit model
    fit = linear_fit.predict(
        np.linspace(interpolated_times[0], interpolated_times[len(interpolated_times) - 1]).reshape((-1, 1)))

    # Compute second residual after removing linear fit
    second_residual_obs = first_residual_obs - linear_fit.predict(interpolated_times.reshape((-1, 1)))


    return {
        # full-day simulated + recorded
        "simulated_observations": simulated_observations,
        "simulated_obs_times": simulated_obs_times,
        "simulated_doppler": simulated_doppler,
        "observation_times": observation_times,
        "observations_set": observations_set,
        "real_doppler": real_doppler,
        "passes_start_times": passes_start_times,
        "passes_end_times": passes_end_times,

        # single pass
        "index_pass": index_pass,
        "single_pass_start_time": single_pass_start_time,
        "single_pass_end_time": single_pass_end_time,
        "real_obs_single_pass": real_obs_single_pass,
        "simulated_obs_single_pass": simulated_obs_single_pass,
        "interpolated_times": interpolated_times,
        "interpolated_simulated_obs": interpolated_simulated_obs,
        "interpolated_real_obs": interpolated_real_obs,
        "first_residual_obs": first_residual_obs,
        "fit": fit,
        "second_residual_obs": second_residual_obs,
    }





    # --- TLE Initialization Summary ---
    # initial_epoch       : float (sec since J2000)  -> TLE reference time
    # start_recording_day : float (sec since J2000)  -> Next UTC day start
    # propagation_time    : float (sec)              -> 1 Julian day (86400 s)
    # final_epoch         : float (sec since J2000)  -> End of simulation
    # delfi_ephemeris     : TleEphemeris             -> SGP4 state generator
    # mid_epoch           : float (sec since J2000)  -> Midpoint of window
    # initial_state       : ndarray(6,) [m, m/s]     -> [x,y,z,vx,vy,vz]
    # -----------------------------------