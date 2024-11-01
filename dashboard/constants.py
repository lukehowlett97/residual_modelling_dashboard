# constants.py
from pathlib import Path

# Your main data folder path
# DATA_FOLDER = Path(r"C:\Users\chcuk\Work\Projects\residual_modelling\data\processed")
DATA_FOLDER = Path(r"D:\CPEFTP\Dashboard_Data")

# Residual types
residual_types = ['res_oc1', 'reg_trop', 'reg_iono', 'ppprtk1']

# Data formats
data_formats = {
    'raw data': '',
    'diff': '_diff',
    'rolling mean': '_diff_rolling_mean',
    'rolling std': '_diff_rolling_std',
    'sg filter': '_sg_filter'
}

# Event labels
event_labels = ['stable', 'spikes', 'steps', 'unclassified', 'shimmering']

# Color mapping for event labels
event_colors = {
    'stable': 'LightGreen',
    'spikes': 'Red',
    'steps': 'Blue',
    'unclassified': 'Gray',
    'shimmering': 'Purple'
}

# Available folders
available_folders = [f.name for f in DATA_FOLDER.iterdir() if f.is_dir()]

