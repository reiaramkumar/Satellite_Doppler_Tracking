from scenarios.nominal import get_nominal_config

def get_nearby_station_config():
    cfg = get_nominal_config()
    cfg.scenario_name = "nearby_stations"
    cfg.nb_fake_stations = 2
    cfg.stations_lat = [52.0705, 51.9244]
    cfg.stations_lon = [4.3007, 4.4777]
    cfg.station_layout_name = 'nearby'
    return cfg

def get_far_station_config():
    cfg = get_nominal_config()
    cfg.scenario_name = "far"
    cfg.nb_fake_stations = 2
    cfg.stations_lat = [-25.0, -14.0]
    cfg.stations_lon = [134.0, -52.0]
    cfg.station_layout_name = 'far'
    return cfg


