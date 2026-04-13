from scenarios.nominal import get_nominal_config
from scenarios.station_cases import get_far_station_config

def get_c20_c22_single_station():
    cfg = get_nominal_config()
    cfg.scenario_name = 'c20_c22_single_station'
    cfg.estimate_C20 = True
    cfg.estimate_C22 = True
    return cfg

def get_c20_c22_far_stations():
    cfg = get_nominal_config()
    cfg.scenario_name = 'c20_c22_far_stations'
    cfg.estimate_C20 = True
    cfg.estimate_C22 = True
    cfg.nb_fake_stations = 2
    cfg.stations_lat = [-25.0, -14.0]
    cfg.stations_lon = [134.0, -52.0]
    cfg.station_layout_name = 'far_stations'
    return cfg

def get_c20_c22_multi_arc_doptrack():
    cfg = get_nominal_config()
    cfg.scenario_name = 'c20_c22_multi_arc_doptrack'
    cfg.estimate_C20 = True
    cfg.estimate_C22 = True
    cfg.propagation_time = 86400.0 * 3  # 3 days total
    cfg.arc_duration = 86400.0          # 1 day per arc = 3 arcs
    return cfg