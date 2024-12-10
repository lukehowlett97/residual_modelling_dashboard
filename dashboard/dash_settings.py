import yaml
from pathlib import Path

# Define default settings
DEFAULT_SETTINGS = {
    'data_folder': "D:/CPEFTP/Dashboard_Data",
    'residual_types': ['res_oc1', 'reg_trop', 'reg_iono', 'ppprtk1'],
    'data_formats': {
        'raw_data': "",
        'diff': "_diff",
        'rolling_mean': "_diff_rolling_mean",
        'rolling_std': "_diff_rolling_std",
        'sg_filter': "_sg_filter"
    },
    'event_labels': ['stable', 'spikes', 'steps', 'unclassified', 'shimmering'],
    'event_colors': {
        'stable': 'LightGreen',
        'spikes': 'Red',
        'steps': 'Blue',
        'unclassified': 'Gray',
        'shimmering': 'Purple'
    }
}

# Load configuration from YAML file if it exists, otherwise use defaults
config_path = Path(__file__).parent / "dash_config.yaml"
if config_path.is_file():
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
else:
    config = {}         

# Extract configuration values or use defaults if keys are missing
DATA_FOLDER = Path(config.get('data_folder', DEFAULT_SETTINGS['data_folder']))
residual_types = config.get('residual_types', DEFAULT_SETTINGS['residual_types'])
data_formats = config.get('data_formats', DEFAULT_SETTINGS['data_formats'])
event_labels = config.get('event_labels', DEFAULT_SETTINGS['event_labels'])
event_colors = config.get('event_colors', DEFAULT_SETTINGS['event_colors'])

print(DATA_FOLDER)
# Available folders based on data folder path
available_folders = [f.name for f in DATA_FOLDER.iterdir() if f.is_dir()]
