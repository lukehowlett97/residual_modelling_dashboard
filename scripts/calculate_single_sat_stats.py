import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import json
from pathlib import Path

def single_sat_statistics(
    data_series: pd.Series,
    segment_features_df: pd.DataFrame,
    sharp_events_dict: Dict[str, Any],
    stats_to_calculate: List[str] = None
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
        'mean': lambda: data_series.mean(),
        'median': lambda: data_series.median(),
        'mean_kurtosis': lambda: segment_features_df['kurtosis'].mean(),
        'mean_skewness': lambda: segment_features_df['skewness'].mean()
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
