from tudatpy import constants
from tudatpy.interface import spice
from tudatpy.astro import frame_conversion
from tudatpy.dynamics import environment
from tudatpy.dynamics import parameters
from tudatpy.estimation import estimation_analysis
from tudatpy.estimation.observable_models_setup import links, model_settings
from tudatpy.estimation.observations_setup import viability, random_noise, observations_simulation_settings, observations_wrapper
from tudatpy.estimation.observations import observations_processing

from propagation_functions.environment import *
from propagation_functions.propagation import *
from estimation_functions.estimation import *

from utility_functions.tle import *

spice.load_standard_kernels()

def get_default_tle():
    return environment.Tle( "1 32789U 08021G   20090.88491347 +.00001016 +00000-0 +70797-4 0  9997",
        "2 32789 097.4279 136.4027 0011143 218.6381 141.4051 15.07550601649972"
    )

def build_accelerations():
    return dict(
        Sun={
            'point_mass_gravity': True,
            'solar_radiation_pressure': True
        },
        Moon={
            'point_mass_gravity': True
        },
        Earth={
            'point_mass_gravity': False,
            'spherical_harmonic_gravity': True,
            'drag': True
        },
        Venus={
            'point_mass_gravity': True
        },
        Mars={
            'point_mass_gravity': True
        },
        Jupiter={
            'point_mass_gravity': True
        }
    )

def build_scenario(config):
    delfi_tle = get_default_tle()
    initial_epoch = delfi_tle.get_epoch()
    final_epoch = initial_epoch + config.propagation_time
    mid_epoch = (initial_epoch + final_epoch) / 2

    arc_start_times, arc_mid_times, arc_end_times = get_arc_times_definition(initial_epoch, final_epoch, config.arc_duration)
    nb_arcs = len(arc_mid_times)

    bodies = define_environment(
        config.mass,
        config.ref_area,
        config.drag_coef,
        config.srp_coef,
        'spacecraft',
        multi_arc_ephemeris = True
    )

    accelerations = build_accelerations()
    delfi_ephemeris = environment.TleEphemeris('Earth', 'J2000', delfi_tle, False)
    initial_state = delfi_ephemeris.cartesian_state(mid_epoch)
    global_orbit = propagate_initial_state( initial_state, initial_epoch, final_epoch, bodies, accelerations, 'spacecraft', save_ephemeris = False)
    arc_wise_initial_state = retrieve_arc_wise_states_from_orbit(global_orbit, arc_mid_times)
    multi_arc_propagation_settings = define_multi_arc_propagation_settings(arc_wise_initial_state, arc_start_times, arc_end_times, bodies, accelerations, 'spacecraft')
    define_doptrack_station(bodies)
    station_names = create_ground_stations(bodies, config.nb_fake_stations, config.stations_lon, config.stations_lat)
    link_definitions = create_link_ends_definitions(config.nb_fake_stations)

    observation_settings_list = []
    for link in link_definitions:
        observation_settings_list.append(model_settings.one_way_doppler_instantaneous(link))

    edge_buffer = 300.0  # seconds
    observation_times = np.arange(
        initial_epoch + edge_buffer,
        final_epoch - edge_buffer,
        config.observation_interval
    )

    observation_simulation_settings = []
    for i in range(config.nb_fake_stations + 1):
        observation_simulation_settings.append(
            observations_simulation_settings.tabulated_simulation_settings(
                model_settings.one_way_instantaneous_doppler_type,
                link_definitions[i],
                observation_times,
            )
                )

    random_noise.add_gaussian_noise_to_observable(
        observation_simulation_settings,
        config.noise_level,
        model_settings.one_way_instantaneous_doppler_type
    )

    for i in range(config.nb_fake_stations + 1):
        viability_setting = viability.elevation_angle_viability(
            ("Earth", station_names[i]),
            np.deg2rad(15)
        )
        viability.add_viability_check_to_observable_for_link_ends(
                                                                    [observation_simulation_settings[i]],
                                                                    [viability_setting],
                                                                    model_settings.one_way_instantaneous_doppler_type,
                                                                    link_definitions[i]
        )

    parameters_list = dict(
        initial_state={
            'estimate': config.estimate_initial_state
        },
        drag_coefficient={
            'estimate': config.estimate_drag,
            'type': 'global'
        },
        gravitational_parameter={
            'estimate': config.estimate_gravitational_parameter,
            'type': 'global'
        },
        C20={
            'estimate': config.estimate_C20,
            'type': 'global'
        },
        C22={
            'estimate': config.estimate_C22,
            'type': 'global'
        }
    )

    parameters_to_estimate = define_parameters(
        parameters_list,
        bodies,
        multi_arc_propagation_settings,
        "spacecraft",
        arc_start_times,
        arc_mid_times
    )

    truth_parameters = parameters_to_estimate.parameter_vector.copy()

    estimator = estimation_analysis.Estimator(bodies, parameters_to_estimate, observation_settings_list, multi_arc_propagation_settings)

    return {
        "config": config,
        "tle": delfi_tle,
        "initial_epoch": initial_epoch,
        "final_epoch": final_epoch,
        "mid_epoch": mid_epoch,
        "arc_start_times": arc_start_times,
        "arc_mid_times": arc_mid_times,
        "arc_end_times": arc_end_times,
        "nb_arcs": nb_arcs,
        "bodies": bodies,
        "accelerations": accelerations,
        "arc_wise_initial_states": arc_wise_initial_state,
        "multi_arc_propagation_settings": multi_arc_propagation_settings,
        "stations_names": station_names,
        "link_definitions": link_definitions,
        "observation_times": observation_times,
        "observation_settings_list": observation_settings_list,
        "observation_simulation_settings": observation_simulation_settings,
        "parameters_to_estimate": parameters_to_estimate,
        "truth_parameters": truth_parameters,
        "estimator": estimator,
    }

