import pandas as pd
from typing import List, Dict, Any
import numpy as np
from pathlib import Path
import json

class StatisticsManager:
    def __init__(self, logger = None):
        self.logger = logger
        
    def _log(self, message):
        if self.logger:
            self.logger.write_log(message)
        else:
            pass
    
    def single_sat_statistics(
        self,
        data_df: pd.Series,
        segment_features_df: pd.DataFrame,
        sharp_events_dict: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculates overall statistics for the dataset based on raw data, segment features, and detected sharp events.

        Parameters:
            data_series (pd.Series): The raw time-series data.
            segment_features_df (pd.DataFrame): DataFrame containing features of each segment.
            sharp_events_dict (Dict[str, Any]): Dictionary containing classified sharp events.
            stats_to_calculate (List[str], optional): List of statistics to calculate. If None, calculates all.

        Returns:
            Dict[str, Any]: Dictionary containing calculated statistics with values rounded to 3 decimal places and as standard floats.
        """
        
        stats_dict = {}
        
        for col in config['columns_to_process']:
            if col not in data_df.columns:
                raise ValueError(f"Column '{col}' not found in data.")
            stats_dict[col] = self._calculate_statistics(data_df[col], segment_features_df, sharp_events_dict, config['stats_to_calculate'])
            
        return stats_dict
        
    def _calculate_statistics(self, data_series, segment_features_df, sharp_events_dict, stats_to_calculate):
        statistics = {}
        
        # Define all possible statistics
        all_stats = {
            'range': lambda: data_series.max() - data_series.min(),
            'iqr': lambda: data_series.quantile(0.75) - data_series.quantile(0.25),
            'std_dev': lambda: data_series.std(),
            'stability_percentage': lambda: (segment_features_df[segment_features_df['label'] == 'stable']['length'].sum() / len(data_series)) * 100,
            'number_of_spikes': lambda: len(sharp_events_dict.get('spikes', {})),
            'number_of_steps': lambda: len(sharp_events_dict.get('steps', {})),
            'number_of_unclassified_events': lambda: len(sharp_events_dict.get('unclassified', {})),
            'number_of_shimmering_periods': lambda: len(sharp_events_dict.get('shimmering', [])),
            'shimmering_percentage': lambda: (sum([end - start for start, end in sharp_events_dict.get('shimmering', [])]) / len(data_series)) * 100,
            'mean': lambda: np.nanmean(data_series),
            'median': lambda: np.nanmedian(data_series),
            'kurtosis': lambda: np.nanmean(segment_features_df['kurtosis']),
            'skewness': lambda: np.nanmean(segment_features_df['skewness'])
        }
        
        # If no specific stats are requested, calculate all
        if stats_to_calculate is None:
            stats_to_calculate = all_stats.keys()
        
        # Calculate requested statistics
        for stat in stats_to_calculate:
            if stat in all_stats:
                try:
                    value = all_stats[stat]()
                    if isinstance(value, (np.floating, float)):
                        # Round to 3 decimal places and convert to float
                        statistics[stat] = round(float(value), 3)
                    else:
                        statistics[stat] = value
                except Exception as e:
                    statistics[stat] = f"Error calculating {stat}: {e}"
            else:
                statistics[stat] = "Statistic not defined."
        
        return statistics

    def save_single_sat_statistics(
        statistics: Dict[str, Any],
        save_path: Path
    ) -> None:
        """
        Saves the statistics dictionary to a JSON file.

        Parameters:
            statistics (Dict[str, Any]): The statistics dictionary to save.
            save_path (Path): The file path where the JSON will be saved.
        """
        try:
            # Ensure the directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the JSON file with indentation for readability
            with open(save_path, 'w') as json_file:
                json.dump(statistics, json_file, indent=4)
            
            print(f"Statistics successfully saved to {save_path}")
        except Exception as e:
            print(f"Failed to save statistics: {e}")
            

    def generate_dataset_statistics(self, save_folder_path: Path):
        """
        Reads all satellite statistics files in the output folder and generates a dataset statistics file.
        
        Parameters:
            output_folder_path (Path): The path to the output folder containing satellite statistics JSON files.
        """
        combined_data: List[Dict[str, any]] = []

        self._log("Starting to generate dataset statistics.")

        # Iterate through all JSON files containing satellite statistics
        for file in save_folder_path.glob('**/*.json'):
            if 'sat_stats' in file.name:
                self._log(f"Processing file: {file}")

                # Extract properties from the filename
                try:
                    file_properties = file.stem.split('_')
                    if len(file_properties) < 6:
                        self._log(f"Filename {file.name} does not conform to expected format. Skipping.")
                        continue

                    year = file_properties[2]
                    doy = file_properties[3]
                    station = file_properties[4]
                    prn = file_properties[5]
                except Exception as e:
                    self._log(f"Error parsing filename {file.name}: {e}. Skipping.")
                    continue

                # Load the JSON data
                try:
                    with open(file, 'r') as f:
                        data_dict = json.load(f)
                except json.JSONDecodeError as e:
                    self._log(f"JSON decode error in file {file.name}: {e}. Skipping.")
                    continue
                except Exception as e:
                    self._log(f"Error reading file {file.name}: {e}. Skipping.")
                    continue

                # Update the data dictionary with additional properties
                data_dict.update({
                    'year': year,
                    'doy': doy,
                    'station': station,
                    'sysnum': prn
                })

                combined_data.append(data_dict)
                self._log(f"Added data from file {file.name}.")

        if not combined_data:
            self._log("No satellite statistics files found. Exiting method.")
            print("No satellite statistics files found.")
            return

        # Normalize the combined data to flatten nested dictionaries
        try:
            # This will flatten the nested dictionaries like 'res_oc1': {...} into 'res_oc1_range', etc.
            combined_df = pd.json_normalize(combined_data, sep='-')
        except Exception as e:
            self._log(f"Error during data normalization: {e}. Exiting method.")
            print(f"Error during data normalization: {e}")
            return

        # Extract 'system' and 'prn_number' from 'prn'
        try:
            combined_df[['system', 'prn']] = combined_df['sysnum'].str.extract(r'([A-Z])(\d+)', expand=True)
            combined_df.drop(columns='sysnum', inplace=True)
            self._log("Extracted 'system' and 'prn_number' from 'prn'.")
        except Exception as e:
            self._log(f"Error extracting 'system' and 'prn_number' from 'prn': {e}.")
            # Depending on requirements, you may choose to continue or exit

        # Save the DataFrame as a pickle file
        save_path_dir = save_folder_path / "_etc/"
        save_path_dir.mkdir(parents=True, exist_ok=True)
        save_file_path = save_path_dir / "dataset_statistics.pkl"

        try:
            combined_df.to_pickle(save_file_path)
            self._log(f"Dataset statistics saved to {save_file_path}")
            print(f"Dataset statistics saved to {save_file_path}")
        except Exception as e:
            self._log(f"Error saving dataset statistics to pickle file: {e}")
            print(f"Error saving dataset statistics to pickle file: {e}")


 # Rename columns for clarity
        # Define a mapping from original column names to desired names
        rename_mapping = {}
        # metrics = ['range', 'iqr', 'std_dev', 'stability_percentage', 'number_of_spikes',
        #         'number_of_steps', 'number_of_unclassified_events', 'number_of_shimmering_periods',
        #         'shimmering_percentage', 'mean', 'median', 'kurtosis', 'skewness']
        
        # # metrics = set([metric.split('-')[0] for metric in metrics])

        # # # Iterate through each feature and create rename mappings
        # # for feature in metrics:
        # #     for column in config['columns_to_process']:
        # #         original_col = f"{column}_{feature}"
        # #         # Create a concise column name, e.g., 'res_oc1_range'
        # #         new_col = f"{column}-{feature}"
        # #         rename_mapping[original_col] = new_col

        # # # Apply the renaming
        # # combined_df.rename(columns=rename_mapping, inplace=True)
        # # self._log("Renamed columns for clarity.")

        # # # Reorder columns
        # # # Define the desired order of columns
        # # desired_order = [
        # #     'res_oc1-range', 'res_oc1-iqr', 'res_oc1-std_dev', 'res_oc1-stability_percentage',
        # #     'res_oc1-number_of_spikes', 'res_oc1-number_of_steps', 'res_oc1-number_of_unclassified_events',
        # #     'res_oc1-number_of_shimmering_periods', 'res_oc1-shimmering_percentage', 'res_oc1-mean', 'res_oc1-median',
        # #     'res_oc1-kurtosis', 'res_oc1-skewness', 'reg_trop-range', 'reg_trop-iqr', 'reg_trop-std_dev',
        # #     'reg_trop-stability_percentage', 'reg_trop-number_of_spikes', 'reg_trop-number_of_steps',
        # #     'reg_trop-number_of_unclassified_events', 'reg_trop-number_of_shimmering_periods',
        # #     'reg_trop-shimmering_percentage', 'reg_trop-mean', 'reg_trop-median', 'reg_trop-kurtosis', 'reg_trop-skewness',
        # #     'reg_iono-range', 'reg_iono-iqr', 'reg_iono-std_dev', 'reg_iono-stability_percentage',
        # #     'reg_iono-number_of_spikes', 'reg_iono-number_of_steps', 'reg_iono-number_of_unclassified_events',
        # #     'reg_iono-number_of_shimmering_periods', 'reg_iono-shimmering_percentage', 'reg_iono-mean', 'reg_iono-median',
        # #     'reg_iono-kurtosis', 'reg_iono-skewness', 'ppprtk1-range', 'ppprtk1-iqr', 'ppprtk1-std_dev',
        # #     'ppprtk1-stability_percentage', 'ppprtk1-number_of_spikes', 'ppprtk1-number_of_steps',
        # #     'ppprtk1-number_of_unclassified_events', 'ppprtk1-number_of_shimmering_periods',
        # #     'ppprtk1-shimmering_percentage', 'ppprtk1-mean', 'ppprtk1-median', 'ppprtk1-kurtosis', 'ppprtk1-skewness'
        # # ]

        # # # Ensure all desired columns are present
        # # missing_cols = [col for col in desired_order if col not in combined_df.columns]
        # # if missing_cols:
        # #     self._log(f"The following expected columns are missing and will be added with NaN values: {missing_cols}")
        # #     for col in missing_cols:
        # #         combined_df[col] = np.nan  # Add missing columns with NaN values

        # # Reorder the columns
        # combined_df = combined_df[desired_order]
        # self._log("Reordered columns as per the desired layout.")
