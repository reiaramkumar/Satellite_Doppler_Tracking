from config import ScenarioConfig


def get_nominal_config():
    return ScenarioConfig(
        scenario_name=                      'nominal',
        propagation_time =                  86400.0,
        arc_duration =                      86400.0,
        observation_interval =              10.0,
        nb_fake_stations =                  0,
        stations_lat =                      [],
        stations_lon =                      [],
        station_layout_name =               'doptrack_only',
        noise_level =                       1.0,
        use_next_tle_as_perturbation =      True,
        use_manual_perturbation =           False,
        estimate_initial_state =            True,
        estimate_drag =                     True,
        estimate_gravitational_parameter =  True,
        estimate_C20 =                      False,
        estimate_C22 =                      False,
        nb_iterations =                     10,
    )