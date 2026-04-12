from dataclasses import dataclass, field
from typing import List

@dataclass
class ScenarioConfig:
    scenario_name:  str = 'nominal'

    # Time Settings
    propagation_time:                   float = 86400.0 # s
    arc_duration:                       float = 86400.0 # s
    observation_interval:               float = 10.0    # s

    # Spacecraft Properties
    mass:                               float = 2.2 # kg
    ref_area:                           float = (4 * 0.3 * 0.1 + 2 *0.1 * 0.1)/4 # m^2
    drag_coef:                          float = 1.2
    srp_coef:                           float = 1.2

    # Ground Stations
    nb_fake_stations:                   int = 0
    stations_lat:                       List[float] = field(default_factory=list)   # deg
    stations_lon:                       List[float] = field(default_factory=list)   # deg
    station_layout_name:                str = 'doptrack_only'

    # Measurement Noise
    noise_level:                        float = 1.0 # m/s
    use_next_tle_as_perturbation:       bool = False
    use_manual_perturbation:            bool = False
    manual_position_perturbation:       float = 0.0
    manual_velocity_perturbation:       float = 0.0

    # Estimation iterations
    nb_iterations:                      int = 10


    # Estimation Settings
    estimate_initial_state:             bool = True
    estimate_drag:                      bool = True
    estimate_gravitational_parameter:   bool = True
    estimate_C20:                       bool = True
    estimate_C22:                       bool = True




