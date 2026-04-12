from scenarios.nominal import get_nominal_config

def get_noise_scenarios():
    noise_levels = [0.1, 1.0, 10]
    configs = []

    for noise in noise_levels:
        cfg = get_nominal_config()
        cfg.noise_level = noise
        cfg.scenario_name = f"noise_{str(noise).replace('.', 'p')}"
        configs.append(cfg)

    return configs