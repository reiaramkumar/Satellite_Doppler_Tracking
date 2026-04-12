from scenarios.nominal import get_nominal_config

def get_manual_perturbation_cases():
    cfg = get_nominal_config()
    cfg.scenario_name = 'manual_1km_1ms'
    cfg.use_next_tle_as_perturbation = False
    cfg.use_manual_perturbation = True
    cfg.manual_position_perturbation = 1000
    cfg.manual_velocity_perturbation = 1.0
    return cfg