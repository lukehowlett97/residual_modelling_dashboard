import numpy as np
import pandas as pd
from pathlib import Path
import json

from Readers.i2gResRead import ReadI2GRes

from scripts.stable_period_detector import StablePeriodDetector
from scripts.segment_feature_extraction import FeatureExtractor
from scripts.sharp_events_detector import SharpEventDetector
from scripts.calculate_single_sat_stats import single_sat_statistics, save_single_sat_statistics

def calculate_segmentation_statistics(group_df, config):
    """
    Skeleton function to calculate segmentation statistics.
    Replace with actual implementation.
    """
    # detect stable periods
    stable_periods = StablePeriodDetector(
        initial_window_size = 10,
        max_window_size = 5000,
        variance_threshold = 0.01,
        moving_variance_window = 30,
        step_size = 10,
        expansion_step = 5
        ).detect_stable_periods(group_df['reg_iono'])
    
    unstable_periods = StablePeriodDetector().get_unstable_periods(stable_periods, len(group_df))
    
    stable_segment_features_df = FeatureExtractor().extract_features(group_df['reg_iono'], stable_periods, 'stable')
    unstable_segment_features_df = FeatureExtractor().extract_features(group_df['reg_iono'], unstable_periods, 'unstable')
    segment_features_combined_df = pd.concat([stable_segment_features_df, unstable_segment_features_df], ignore_index=True)
    
    sharp_events_dict = SharpEventDetector().events_detection(group_df['reg_iono'])
    
    dataset_summary = single_sat_statistics(group_df['reg_iono'], segment_features_combined_df, sharp_events_dict)
    
    
    year =str(config['year'])
    year_int =int(year)
    doy = int(config['doy'])
    station = config['station']
    sys = int(group_df['sys'].iloc[0])
    num = int(group_df['num'].iloc[0])
    sys_map = {1: 'G', 2: 'R', 3: 'C', 4: 'E'}
    prn_tag = f"{sys_map[sys]}{num:02d}"
    
    save_folder_path = Path(config['output_folder']) / year / f"{doy:03d}" / station 
    
    save_seg_file_path =  save_folder_path / f'seg_features_{year_int}_{doy:03d}_{station}_{prn_tag}.pkl'
    save_sat_stats_path = save_folder_path /  f'sat_stats_{year_int}_{doy:03d}_{station}_{prn_tag}.json'
    save_sharp_events_path = save_folder_path /  f'sharp_events_{year_int}_{doy:03d}_{station}_{prn_tag}.json'
    
    save_folder_path.mkdir(parents=True, exist_ok=True)
    save_segmentation_statistics(segment_features_combined_df, save_seg_file_path)
    save_single_sat_statistics(dataset_summary, save_sat_stats_path)
    save_sharp_events(sharp_events_dict, save_sharp_events_path)
    
    return segment_features_combined_df

def save_sharp_events(sharp_events_dict, save_sharp_events_path):
    """
    Save sharp events as a JSON file.
    """

    with open(save_sharp_events_path, 'w') as f:
        json.dump(sharp_events_dict, f, indent=4)    
    

def save_segmentation_statistics(segment_features_df, save_seg_file_path):
    """
    Save segmentation statistics as a pickle file.

    Parameters:
    - segment_features_df (pd.DataFrame): DataFrame containing the segmentation features.
    - save_path (Path or str): Path where the pickle file will be saved.
    """
    
    segment_features_df.to_pickle(save_seg_file_path)
    

def process_group(group_df, config, system_map):
    """
    Process a single group of data, calculate rolling statistics,
    and save the processed data and segmentation statistics.

    Parameters:
    - group_df (pd.DataFrame): The DataFrame group to process.
    - config (dict): Configuration parameters.
    - system_map (dict): Mapping of system codes to labels.

    Returns:
    - pd.DataFrame: The processed group DataFrame.
    """
    group_df = group_df.reset_index(drop=True)

    # Add station, year, and day of year to group_df
    group_df['station'] = config['station']
    group_df['year'] = config['year']
    group_df['doy'] = config['doy']

    # Calculate rolling statistics
    for col in config['columns_to_process']:
        group_df[f'{col}_diff'] = group_df[col].diff()
        group_df[f'{col}_diff_rolling_mean'] = group_df[f'{col}_diff'].rolling(window=config['rolling_window']).mean()
        group_df[f'{col}_diff_rolling_std'] = group_df[f'{col}_diff'].rolling(window=config['rolling_window']).std()
        group_df[f'{col}_sg_filter'] = group_df[col].rolling(window=config['rolling_window']).apply(
            lambda x: np.polyfit(range(config['rolling_window']), x, config['poly_order'])[0]
            if len(x) == config['rolling_window'] else np.nan
        )

    # Calculate segmentation statistics and save to output folder
    calculate_segmentation_statistics(group_df, config)
    
    # Determine prn_tag
    prn_tag = system_map.get(int(group_df['sys'].iloc[0]), 'Unknown') + f"{int(group_df['num'].iloc[0]):02d}"

    # Construct filenames with the new naming convention
    proc_filename = f"proc_res_{config['year']}_{config['doy']:03d}_{config['station']}_{prn_tag}.pkl"

    # Construct the directory path: output_folder / station / year / doy
    save_directory = config['output_folder'] / f"{config['year']}" / f"{config['doy']:03d}" / config['station'] 
    save_directory.mkdir(parents=True, exist_ok=True)

    # Save transformed group data
    group_save_path = save_directory / proc_filename
    group_df.to_pickle(group_save_path)

    return group_df

def extract_file_info(filename):
    """
    Extract station, year, and day of year from filename.
    Assumes format: CMDN_2024002_intg.res

    Parameters:
    - filename (Path): The Path object of the file.

    Returns:
    - tuple: (station (str), year (int), day_of_year (int))
    """
    base = filename.stem  # e.g., CMDN_2024002_intg
    parts = base.split('_')
    if len(parts) < 2:
        raise ValueError(f"Filename {filename} does not match expected format.")
    station = parts[0]
    year_day = parts[1]
    if len(year_day) != 7:
        raise ValueError(f"Year and day in filename {filename} do not match expected format.")
    year = int(year_day[:4])
    day_of_year = int(year_day[4:])
    return station, year, day_of_year

def preprocess_data(input_folder, output_folder, config=None):
    """
    Preprocess all .res files in the input_folder and save results to output_folder.

    Parameters:
    - input_folder (str or Path): Path to the input directory containing .res files.
    - output_folder (str or Path): Path to the directory where processed data will be saved.
    - config (dict): Configuration parameters.
    """
    # Default configuration
    default_config = {
        'columns_to_process': ["res_oc1", "reg_trop", "reg_iono", "ppprtk1"],
        'rolling_window': 10,
        'poly_order': 2,
        'input_extension': '.res',
        'file_pattern': '*_???????_intg.res',  # Adjust as needed
        'max_files': None,  # Set to None for no limit
    }

    if config is None:
        config = default_config
    else:
        # Update default config with provided config
        for key, value in default_config.items():
            config.setdefault(key, value)

    input_folder = Path(input_folder)
    output_folder = Path(output_folder) / f"processed_{input_folder.stem}"

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder {input_folder} does not exist.")

    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    # Define system_map
    system_map = {1: 'G', 2: 'R', 3: 'C', 4: 'E'}

    # Initialize file counter
    file_count = 0
    max_files = config.get('max_files', None)

    # Loop through all .res files in input_folder
    for res_file in input_folder.glob(f'*{config["input_extension"]}'):
        if not res_file.is_file():
            continue

        if max_files is not None and file_count >= max_files:
            print(f"Reached the maximum number of files to process: {max_files}")
            break

        file_count += 1

        try:
            station, year, day_of_year = extract_file_info(res_file)
            print(f"Processing file: {res_file.name} | Station: {station}, Year: {year}, Day: {day_of_year}")
        except ValueError as e:
            print(f"Skipping file {res_file.name}: {e}")
            continue

        # Read data
        try:
            df = ReadI2GRes(res_file).get_fix_s_data()
            df = df[['epoch', 'sys', 'num', *config['columns_to_process']]]
        except Exception as e:
            print(f"Error reading file {res_file.name}: {e}")
            continue

        # Group by sys and num and process each group
        try:
            grouped = df.groupby(['sys', 'num'])

            # Update config with file metadata and output folder
            group_config = config.copy()
            group_config['output_folder'] = output_folder
            group_config['station'] = station
            group_config['year'] = year
            group_config['doy'] = day_of_year

            # Apply processing to each group
            for [sys,num], group_df in grouped:
                try:
                    group_df.reset_index(inplace=True, drop=True)
                    process_group(group_df, group_config, system_map)
                except Exception as e:
                    print(f"Error processing group {sys}, {num} in file {res_file.name}: {e}")
                    continue
        except Exception as e:
            print(f"Error processing groups in file {res_file.name}: {e}")
            continue
        
    # generate dataset statistics file
    generate_dataset_statistics(output_folder)
    
def generate_dataset_statistics(output_folder):
    combined_data = []

    for file in output_folder.glob('**/*.json'):
        if 'sat_stats' in file.name:
            file_properties = file.stem.split('_')
            year = file_properties[2]
            doy = file_properties[3]
            station = file_properties[4]
            prn = file_properties[5]

            with open(file, 'r') as f:
                data_dict = json.load(f)

            data_dict.update({
                'year': year,
                'doy': doy,
                'station': station,
                'prn': prn
            })

            combined_data.append(data_dict)

    if combined_data:
        combined_df = pd.DataFrame(combined_data)
        combined_df[['system', 'prn']] = combined_df['prn'].str.extract(r'([A-Z])(\d+)')
        
        # rename columns
        combined_df.columns = ['rng', 'iqr', 'sd', 'stab%', 'spikes',
       'steps', 'misc',
       'num_shim', 'shim%', 'mean',
       'medi', 'kurt', 'skew', 'year', 'doy', 'stn',
       'prn', 'sys']
        
        # reorder columns
        combined_df = combined_df[['year', 'doy', 'stn', 'sys', 'prn', 'rng', 'iqr', 'sd', 'stab%', 'spikes', 'steps', 'misc', 'num_shim', 'shim%', 'mean', 'medi', 'kurt', 'skew']]
        
        save_path_dir = output_folder / r"_etc/"
        save_path_dir.mkdir(parents=True, exist_ok=True)
        combined_df.to_pickle(save_path_dir / "dataset_statistics.pkl")
        print(f"Dataset statistics saved to {save_path_dir}")
    else:
        print("No satellite statistics files found.")

def main():
    """
    Main function to execute the preprocessing.
    """
    # Example usage:
    input_dir = r"\\meetingroom\Integrity\SWASQC\res20240122"
    output_dir = r"C:\Users\chcuk\Work\Projects\residual_modelling\data\processed"

    # Optional: Define custom configuration
    custom_config = {
        'rolling_window': 10,
        'poly_order': 2,
        'columns_to_process': ["res_oc1", "reg_trop", "reg_iono", "ppprtk1"],
        'max_files': 0,  # Uncomment to limit the number of files processed
    }

    preprocess_data(input_dir, output_dir, config=custom_config)

if __name__ == "__main__":
    main()
