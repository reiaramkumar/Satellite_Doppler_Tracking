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
    return cfg
