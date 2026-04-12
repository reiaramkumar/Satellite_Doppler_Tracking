import os

metadata = [
    'Delfi-C3_32789_202004011044.yml', 'Delfi-C3_32789_202004011219.yml',
    'Delfi-C3_32789_202004021953.yml', 'Delfi-C3_32789_202004022126.yml',
    'Delfi-C3_32789_202004031031.yml', 'Delfi-C3_32789_202004031947.yml',
    'Delfi-C3_32789_202004041200.yml',
    'Delfi-C3_32789_202004061012.yml', 'Delfi-C3_32789_202004062101.yml',
    'Delfi-C3_32789_202004072055.yml', 'Delfi-C3_32789_202004072230.yml',
    'Delfi-C3_32789_202004081135.yml'
]

data = [
    'Delfi-C3_32789_202004011044.csv', 'Delfi-C3_32789_202004011219.csv',
    'Delfi-C3_32789_202004021953.csv', 'Delfi-C3_32789_202004022126.csv',
    'Delfi-C3_32789_202004031031.csv', 'Delfi-C3_32789_202004031947.csv',
    'Delfi-C3_32789_202004041200.csv',
    'Delfi-C3_32789_202004061012.csv', 'Delfi-C3_32789_202004062101.csv',
    'Delfi-C3_32789_202004072055.csv', 'Delfi-C3_32789_202004072230.csv',
    'Delfi-C3_32789_202004081135.csv'
]

# Default: all passes except index 5 (day 5 missing)
DEFAULT_INDICES = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]

# Spacecraft physical properties
MASS       = 2.2
REF_AREA   = (4 * 0.3 * 0.1 + 2 * 0.1 * 0.1) / 4
SRP_COEF   = 1.2
DRAG_COEF  = 1.2
SAT_NAME   = "Delfi"

METADATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'metadata') + os.sep
DATA_FOLDER     = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data') + os.sep

# Default accelerations (full model)
DEFAULT_ACCELERATIONS = dict(
    Sun={
        'point_mass_gravity': True,
        'solar_radiation_pressure': True
    },
    Moon={'point_mass_gravity': True},
    Earth={
        'point_mass_gravity': False,
        'spherical_harmonic_gravity': True,
        'drag': True
    },
    Venus={'point_mass_gravity': True},
    Mars={'point_mass_gravity': True},
    Jupiter={'point_mass_gravity': True}
)