import pandas as pd
import numpy as np
from scipy import stats
from scipy.fftpack import fft
from pathlib import Path

def calculate_central_tendencies(group):
    return {
        'mean': group.mean(),
        'median': group.median(),
        'mode': group.mode()[0] if not group.mode().empty else np.nan
    }

def calculate_variability(group):
    return {
        'std': group.std(),
        'variance': group.var(),
        'min': group.min(),
        'max': group.max(),
        'range': group.max() - group.min(),
        'IQR': group.quantile(0.75) - group.quantile(0.25)
    }

def calculate_shape_metrics(group):
    return {
        'skewness': group.skew(),
        'kurtosis': group.kurtosis()
    }

def calculate_autocorrelation(group, lags=20):
    return [group.autocorr(lag) for lag in range(1, lags + 1)]

def calculate_fourier_transform(group):
    N = len(group)
    T = 1.0  # Assuming uniform time spacing of 1 unit
    yf = fft(group.dropna())
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    return 2.0 / N * np.abs(yf[:N // 2])

def calculate_outliers(group, z_threshold=3):
    zscore = (group - group.mean()) / group.std()
    outliers = np.abs(zscore) > z_threshold
    time_between_outliers = np.diff(group.index[outliers]) if outliers.sum() > 1 else np.nan
    return {
        'outlier_count': outliers.sum(),
        'outlier_proportion': outliers.mean(),
        'time_between_outliers': time_between_outliers.mean() if not np.isnan(time_between_outliers).all() else np.nan
    }

def calculate_temporal_trend(group):
    time = np.arange(len(group))
    slope, intercept, _, _, _ = stats.linregress(time, group.fillna(0))
    return {'slope': slope}

def statistical_sweep(df, residuals_to_process, station_name, output_dir='./output'):
    """
    Perform the full statistical sweep on the given dataframe, grouped by ['sys', 'num'],
    and combine all results into one CSV file.
    
    Parameters:
    df (pd.DataFrame): Input dataframe.
    residuals_to_process (list): List of columns to process.
    station_name (str): Name of the station for output file identification.
    output_dir (str): Directory to save the combined output file.
    
    Returns:
    pd.DataFrame: Combined dataframe with all statistical results.
    """
    # Ensure the output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract process date info
    df['epoch'] = pd.to_datetime(df['epoch'])
    process_year = df['epoch'].dt.year.iloc[0]
    day_of_year = df['epoch'].dt.dayofyear.iloc[0]

    # Initialize storage for all results
    combined_results = []

    # Group by ['sys', 'num'] (each satellite)
    grouped_df = df.groupby(['sys', 'num'])

    # Process each group
    for (sys, num), group in grouped_df:
        for col in residuals_to_process:
            result = {'sys': sys, 'num': num, 'column': col, 'epoch_count': group['epoch'].count()}

            # Central tendencies
            result.update(calculate_central_tendencies(group[col]))

            # Variability
            result.update(calculate_variability(group[col]))

            # Shape metrics
            result.update(calculate_shape_metrics(group[col]))

            # Outliers
            result.update(calculate_outliers(group[col]))

            # Temporal trend
            result.update(calculate_temporal_trend(group[col]))

            combined_results.append(result)

    # Convert combined results to DataFrame
    combined_df = pd.DataFrame(combined_results)

    # Save the combined results as a CSV
    output_filename = f'{station_name}_combined_stats_{process_year}_{day_of_year:02d}.csv'
    combined_df.to_csv(output_path / output_filename, index=False)

    # Return the combined DataFrame
    return combined_df
