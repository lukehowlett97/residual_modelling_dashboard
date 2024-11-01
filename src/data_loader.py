from pathlib import Path
from typing import List, Optional, Callable
import numpy as np
import pandas as pd

# Placeholder for the custom reader import, adjust based on actual reader versions
from Readers import ReadI2GRes

def load_single_file(file, constellations=None, excluded_svs=None, residuals=None, normalize_data = True):
    
    constellation_map = {'G': 1, 'R':2, 'C':3, 'E':4}

    # Load the file using the appropriate reader
    df = ReadI2GRes(file).get_fix_s_data()

    # Filter by constellations if provided
    if constellations:
        constellation_nums = [constellation_map[c] for c in constellations]
        df = df[df['sys'].isin(constellation_nums)]

    # Filter by satellite vehicle (SV) numbers if provided
    if excluded_svs is not None:
        for excluded_sv in excluded_svs:
            
            constellation_code = excluded_sv[0]  # First character is the constellation code
            sv_number = int(excluded_sv[1:])  # The rest is the satellite vehicle number

            # Map the constellation code to its corresponding number
            constellation_num = constellation_map[constellation_code]

            # Exclude rows where both sys and num match the excluded constellation and SV number
            df = df[~((df['sys'] == constellation_num) & (df['num'] == sv_number))]
        

    # Extract only specified residuals if provided
    if residuals:
        df = df[['epoch', 'sys', 'num'] + residuals]

    # Normalize specified columns if needed
    if normalize_data:
        for col in residuals:
            if col in df.columns:
                df[col] = normalize(df[col])

    df.reset_index(drop=True, inplace=True)
    
    return df

def process_file(file: Path,
                 constellations: Optional[List[int]] = None,
                 svs: Optional[List[int]] = None,
                 residuals: Optional[List[str]] = None,
                 normalize_columns: Optional[List[str]] = None,
                 processing_function: Optional[Callable[[pd.DataFrame, Path], None]] = None) -> None:
    """
    Process a single GNSS residual file, applying filtering, normalization, and user-specified processing.

    Args:
        file (Path): The file to process.
        constellations (List[int], optional): List of constellation IDs to filter by.
        svs (List[int], optional): List of satellite vehicle (SV) numbers to filter by.
        residuals (List[str], optional): List of residual column names to extract.
        normalize_columns (List[str], optional): List of columns to normalize.
        processing_function (Callable, optional): Custom function to process the DataFrame.
                                                 It receives the DataFrame and the file path.
    """
    print(f"Processing file: {file}")

    # Load the file using the appropriate reader
    df = ReadI2GRes(file).get_fix_s_data()

    # Filter by constellations if provided
    if constellations:
        df = df[df['sys'].isin(constellations)]

    # Filter by satellite vehicle (SV) numbers if provided
    if svs:
        df = df[df['num'].isin(svs)]

    # Extract only specified residuals if provided
    if residuals:
        df = df[['epoch', 'sys', 'num'] + residuals]

    # Normalize specified columns if needed
    if normalize_columns:
        for col in normalize_columns:
            if col in df.columns:
                df[col] = normalize(df[col])

    df.reset_index(drop=True, inplace=True)

    # Pass the processed DataFrame to the user-specified processing function
    if processing_function:
        processing_function(df, file)
    else:
        # Default action: save to a CSV (or handle in some other way)
        output_file = file.with_suffix('.processed.csv')
        df.to_csv(output_file, index=False)
        print(f"Saved processed data to {output_file}")

def normalize(v: pd.Series) -> pd.Series:
    """
    Normalize a Pandas Series.

    Args:
        v (pd.Series): Series to normalize.

    Returns:
        pd.Series: Normalized Series.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def load_and_process_files(input_folder: str,
                           constellations: Optional[List[int]] = None,
                           svs: Optional[List[int]] = None,
                           residuals: Optional[List[str]] = None,
                           normalize_columns: Optional[List[str]] = None,
                           processing_function: Optional[Callable[[pd.DataFrame, Path], None]] = None,
                           file_pattern: str = "*.res") -> None:
    """
    Load and process GNSS residual data from files in a specified directory, file-by-file.

    Args:
        input_folder (str): Path to the directory containing residual files.
        constellations (List[int], optional): List of constellation IDs to filter by.
        svs (List[int], optional): List of satellite vehicle (SV) numbers to filter by.
        residuals (List[str], optional): List of residual column names to extract.
        normalize_columns (List[str], optional): List of columns to normalize.
        processing_function (Callable, optional): Custom function to process the DataFrame.
                                                 It receives the DataFrame and the file path.
        file_pattern (str, optional): Glob pattern to match files. Default is "*.res".

    Returns:
        None
    """
    input_path = Path(input_folder)

    # Iterate through files in the specified directory
    for file in input_path.glob(file_pattern):
        try:
            # Process each file individually
            process_file(file, constellations, svs, residuals, normalize_columns, processing_function)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
