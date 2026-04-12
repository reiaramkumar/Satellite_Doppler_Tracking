from scenarios.nominal import get_nominal_config

def get_three_day_one_day_arcs():
    cfg = get_nominal_config()
    cfg.scenario_name = 'three_day_one_day_arcs'
    cfg.propagation_time = 3 * 86400.0
    cfg.arc_duration = 86400.0
    return cfg
