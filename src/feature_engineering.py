import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.tsa.seasonal import STL
from scipy.stats import entropy
from statsmodels.tsa.stattools import acf
import ruptures as rpt


# -------------------------
# Feature Generation
# -------------------------

def generate_epoch_range(min_epoch, max_epoch, interval_seconds=5) -> List[pd.Timestamp]:
    """Generate a range of epochs between min_epoch and max_epoch with a specified interval."""
    return [min_epoch + i * timedelta(seconds=interval_seconds) 
            for i in range(int((max_epoch - min_epoch).total_seconds() / interval_seconds) + 1)]

def calculate_deltas(df: pd.DataFrame, column: str = 'reg_iono') -> pd.Series:
    """Calculate the difference between consecutive values."""
    return df[column].diff()

def normalize(v: pd.Series) -> pd.Series:
    """Normalize a Pandas Series."""
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm


# -------------------------
# Statistical Analysis
# -------------------------

def calculate_statistics(segment_data: pd.DataFrame, segment_data_prev: Optional[pd.DataFrame] = None, 
                         column: str = 'reg_iono', delta_column: str = 'delta', 
                         percentile_indexes: List[float] = [0.01, 0.05, 0.1, 0.25, 0.75, 0.9, 0.95, 0.99]) -> Dict[str, float]:
    """Calculate various statistics for a segment of data, handling errors gracefully."""
    stats = {}
    # Basic statistics
    stats['Mean'] = segment_data[column].mean()
    stats['Median'] = segment_data[column].median()
    stats['Std'] = segment_data[column].std()
    stats['Skewness'] = segment_data[column].skew()
    stats['Kurtosis'] = segment_data[column].kurtosis()
    stats['Percentiles'] = segment_data[column].quantile(percentile_indexes)
    stats['Available Epochs'] = len(segment_data.dropna())
    stats['Availability'] = len(segment_data.dropna()) / len(segment_data) * 100
    
    # Delta statistics
    stats['Mean Delta'] = segment_data[delta_column].mean()
    stats['Std Delta'] = segment_data[delta_column].std()
    stats['Skewness Delta'] = segment_data[delta_column].skew()
    stats['Kurtosis Delta'] = segment_data[delta_column].kurtosis()
    stats['Delta Percentiles'] = segment_data[delta_column].quantile(percentile_indexes)
    
    # Jump detection
    jump_threshold = 3 * stats['Std Delta']
    stats['Jumps Count'] = len(segment_data[segment_data[delta_column].abs() > jump_threshold])
    
    # Trend analysis
    linear_trend, slope, intercept, r2 = linear_trend_analysis(segment_data)
    stats.update({
        'Linear Trend': linear_trend,
        'Slope': slope,
        'Intercept': intercept,
        'R2': r2
    })
    
    # Additional statistics
    stats['Range'] = segment_data[column].max() - segment_data[column].min()
    stats['IQR'] = stats['Percentiles'][0.75] - stats['Percentiles'][0.25]

    # Rolling statistics
    rolling_stats = calculate_rolling_statistics(segment_data)
    stats.update(rolling_stats)

    # Autocorrelation, Entropy, FFT
    stats['Autocorrelation'] = calculate_autocorrelation(segment_data)
    stats['Entropy'] = calculate_entropy(segment_data)
    fft_values, fft_freq, dominant_freq, total_power, spectral_entropy = calculate_fft(segment_data)
    stats.update({
        'FFT Values': fft_values,
        'FFT Frequencies': fft_freq,
        'Dominant Frequency': dominant_freq,
        'Total Power': total_power,
        'Spectral Entropy': spectral_entropy
    })

    # Z-Scores, CUSUM
    stats['Z Scores'] = calculate_z_scores(segment_data)
    stats['CUSUM'] = calculate_cusum(segment_data)

    # Stability and Volatility
    stats['Stability Index'] = calculate_stability_index(stats)
    stats['Volatility Index'] = calculate_volatility_index(stats)
    
    # Trend decomposition
    trend, seasonal, resid = decompose_trend(segment_data)
    stats.update({
        'Trend': trend,
        'Seasonal': seasonal,
        'Residual': resid
    })

    # Segment comparison
    if segment_data_prev is not None:
        diff_mean, diff_std = compare_segments(segment_data, segment_data_prev)
    else:
        diff_mean, diff_std = np.nan, np.nan
    stats.update({
        'Mean Difference': diff_mean,
        'Std Difference': diff_std
    })
    
    return stats


# -------------------------
# Trend and Anomaly Detection
# -------------------------

def linear_trend_analysis(segment_data: pd.DataFrame, time_column: str = 'epoch', value_column: str = 'reg_iono') -> Tuple[str, float, float, float]:
    """Perform linear regression on a segment to determine the trend."""
    X = np.array((segment_data[time_column] - segment_data[time_column].min()).dt.total_seconds()).reshape(-1, 1)
    y = segment_data[value_column].values

    valid_indices = ~np.isnan(y)
    X = X[valid_indices]
    y = y[valid_indices]

    if len(X) == 0 or len(y) == 0:
        return 'stable', 0.0, 0.0, 0.0

    model = LinearRegression().fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    if intercept > 0:
        trend = 'converging' if slope < 0 else 'diverging'
    elif intercept < 0:
        trend = 'converging' if slope > 0 else 'diverging'
    else:
        trend = 'stable'
    
    return trend, slope, intercept, r2

def calculate_rolling_statistics(segment_data: pd.DataFrame, column: str = 'reg_iono', window_size: int = 5) -> Dict[str, pd.Series]:
    """Calculate rolling statistics for a segment of data."""
    rolling_stats = {
        'Rolling Mean': segment_data[column].rolling(window=window_size, min_periods=1).mean(),
        'Rolling Std': segment_data[column].rolling(window=window_size, min_periods=1).std(),
        'Rolling Skewness': segment_data[column].rolling(window=window_size, min_periods=1).skew(),
        'Rolling Kurtosis': segment_data[column].rolling(window=window_size, min_periods=1).kurt()
    }
    return rolling_stats

def calculate_autocorrelation(segment_data: pd.DataFrame, column: str = 'reg_iono', nlags: int = 10) -> np.ndarray:
    """Calculate autocorrelation for a segment of data."""
    return acf(segment_data[column].dropna(), nlags=nlags)

def calculate_entropy(segment_data: pd.DataFrame, column: str = 'reg_iono') -> float:
    """Calculate entropy of the segment data."""
    value_counts = segment_data[column].value_counts(normalize=True, bins=10)
    return entropy(value_counts)

def calculate_fft(segment_data: pd.DataFrame, column: str = 'reg_iono') -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """Calculate FFT of the segment data."""
    fft_values = np.fft.fft(segment_data[column].dropna())
    fft_freq = np.fft.fftfreq(len(fft_values))
    positive_freq_indices = fft_freq > 0
    dominant_freq = fft_freq[positive_freq_indices][np.argmax(np.abs(fft_values[positive_freq_indices]))]
    total_power = np.sum(np.abs(fft_values[positive_freq_indices]) ** 2)
    spectral_entropy = entropy(np.abs(fft_values[positive_freq_indices]))
    return fft_values, fft_freq, dominant_freq, total_power, spectral_entropy

def calculate_z_scores(segment_data: pd.DataFrame, column: str = 'reg_iono') -> pd.Series:
    """Calculate Z-scores of the segment data."""
    return (segment_data[column] - segment_data[column].mean()) / segment_data[column].std()

def calculate_cusum(segment_data: pd.DataFrame, column: str = 'reg_iono') -> pd.Series:
    """Calculate the cumulative sum (CUSUM) of the deviations from the mean."""
    mean_value = segment_data[column].mean()
    return (segment_data[column] - mean_value).cumsum()

def calculate_stability_index(stats: Dict[str, float]) -> float:
    """Calculate a stability index based on various statistics."""
    return (1 / stats['Std']) * (1 / np.abs(stats['Skewness'])) * (1 / stats['Entropy'])

def calculate_volatility_index(stats: Dict[str, float]) -> float:
    """Calculate a volatility index based on various statistics."""
    return stats['Std'] * stats['Kurtosis'] * stats['Jumps Count']

def decompose_trend(segment_data: pd.DataFrame, column: str = 'reg_iono', period: int = 30) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Decompose the trend using STL decomposition."""
    stl = STL(segment_data[column].dropna(), period=period)
    result = stl.fit()
    return result.trend, result.seasonal, result.resid

def compare_segments(segment_data1: pd.DataFrame, segment_data2: pd.DataFrame, column: str = 'reg_iono') -> Tuple[float, float]:
    """Compare two segments and return differences in statistics."""
    diff_mean = segment_data1[column].mean() - segment_data2[column].mean()
    diff_std = segment_data1[column].std() - segment_data2[column].std()
    return diff_mean, diff_std


# -------------------------
# Segment Processing
# -------------------------

def process_segment(segment_id: int, df: pd.DataFrame) -> Dict[str, float]:
    """Process a single segment, computing statistics."""
    segment_data = df[df['segment'] == segment_id]
    
    min_epoch = segment_data['epoch'].min()
    max_epoch = segment_data['epoch'].max()
    epoch_range = generate_epoch_range(min_epoch, max_epoch)
    
    segment_data = segment_data.set_index('epoch').reindex(epoch_range).reset_index()

    if segment_id == 0:
        segment_data_prev = None
    else:
        segment_data_prev = df[df['segment'] == segment_id - 1]
        min_epoch_prev = segment_data_prev['epoch'].min()
        max_epoch_prev = segment_data_prev['epoch'].max()
        epoch_range_prev = generate_epoch_range(min_epoch_prev, max_epoch_prev)
        segment_data_prev = segment_data_prev.set_index('epoch').reindex(epoch_range_prev).reset_index()
    
    segment_stats = calculate_statistics(segment_data, segment_data_prev)
    segment_stats['Segment'] = int(segment_id)
    
    return segment_stats

def process_all_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Process all segments and compile the statistics."""
    segment_stats = [process_segment(segment_id, df) for segment_id in df['segment'].unique()]
    return compile_statistics_df(segment_stats)

def compile_statistics_df(segment_stats: List[Dict[str, float]]) -> pd.DataFrame:
    """Compile the statistics into a pandas DataFrame."""
    percentiles_df = pd.DataFrame(segment_stats)
    
    # Expand percentile data into separate columns
    percentiles_df[['1%', '5%', '10%', '25%', '75%', '90%', '95%', '99%']] = pd.DataFrame(percentiles_df['Percentiles'].tolist(), index=percentiles_df.index)
    percentiles_df[['1_d%', '5_d%', '10_d%','25_d%', '75_d%', '90_d%', '95_d%', '99_d%']] = pd.DataFrame(percentiles_df['Delta Percentiles'].tolist(), index=percentiles_df.index)
    
    # Drop the original percentile columns
    percentiles_df.drop(['Percentiles', 'Delta Percentiles'], axis=1, inplace=True)
    
    return percentiles_df


# -------------------------
# Main Interface
# -------------------------

def feature_engineering_pipeline(df: pd.DataFrame, segment_column: str = 'segment') -> pd.DataFrame:
    """
    Main function to apply feature engineering to all segments in the DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        segment_column (str): The column name that identifies segments in the DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with calculated features.
    """
    df['delta'] = calculate_deltas(df)
    processed_df = process_all_segments(df)
    
    return processed_df


# -------------------------
# Example Usage
# -------------------------

# Example of how to use the feature_engineering_pipeline:
# df = pd.read_csv('your_data.csv')  # Load your data here
# df['epoch'] = pd.to_datetime(df['epoch'])  # Ensure 'epoch' is in datetime format
# result_df = feature_engineering_pipeline(df)
# result_df.to_csv('processed_features.csv', index=False)
