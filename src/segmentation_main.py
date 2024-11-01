
import pandas as pd
from typing import Dict
from src.refinement_process import refine_fixed_intervals, refine_cpa_intervals

def apply_segmentation(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Apply segmentation to the DataFrame based on the provided configuration.

    Args:
        df (pd.DataFrame): The input DataFrame.
        config (Dict): Configuration settings for segmentation.

    Returns:
        pd.DataFrame: The DataFrame with segmentation columns added.
    """
    # Group the data by 'sys' and 'num' (constellation and satellite number)
    grouped = df.groupby(['sys', 'num'])

    # Apply segmentation for each group
    processed_groups = []
    for name, group in grouped:
        processed_group = process_single_group(group, config)
        processed_groups.append(processed_group)

    # Concatenate the results
    result_df = pd.concat(processed_groups)
    return result_df

def process_single_group(group_df: pd.DataFrame, config: Dict): 
    """
    Process a single group of data based on the provided configuration.

    Args:
        group_df (pd.DataFrame): The input DataFrame group.
        config (Dict): Configuration settings for segmentation.

    Returns:
        pd.DataFrame: The DataFrame group with segmentation columns added.
    """
    # Initialize the result DataFrame
    result_df = group_df.copy()
    
    # Fixed Interval Segmentation
    initial_interval_len = '1H'
    refinement_factor = 2
    max_loops = 5
    # refined_intervals = refine_fixed_intervals(group_df, initial_interval_len, refinement_factor, max_loops)

    # change point analysis
    cpa_intervals = refine_cpa_intervals(group_df)
    # evaluate
    
    # clustering
    # apply segmentation
    # evaluate
    
    # wavelet transform
    # apply segmentation
    # evaluate
    
    return result_df    
    

    