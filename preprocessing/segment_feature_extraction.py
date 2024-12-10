# feature_extractor.py

import numpy as np
import pandas as pd
# from scipy.stats import kurtosis, skew
from typing import List, Dict, Optional
from FileLogging.simple_logger import SimpleLogger


class FeatureExtractor:
    """
    A class to extract statistical and temporal features from labeled segments of time-series data.
    """

    def __init__(self, features: Optional[List[str]] = None, logger: Optional[SimpleLogger] = None):
        """
        Initialize the FeatureExtractor with a list of features to extract.

        Parameters:
            features (List[str], optional): List of feature names to extract. 
                Defaults to ['mean', 'std', 'length', 'slope', 'kurtosis', 'skewness'].
            logger (SimpleLogger, optional): Instance of the custom logger.
        """
        if features is None:
            self.features = ['mean', 'std', 'length', 'slope', 'kurtosis', 'skewness']
        else:
            self.features = features
        self.logger = logger

    def _log(self, message: str):
        """
        Internal method to handle logging.

        Parameters:
            message (str): The message to log.
        """
        if self.logger:
            self.logger.write_log(message)
        else:
            pass  # Alternatively, you can use print(message) for console output

    def extract_features(
        self, 
        data: pd.Series, 
        labeled_periods: List[Dict[str, int]], 
        label: str,
        column: str
    ) -> pd.DataFrame:
        """
        Extract features from the labeled periods of the data.

        Parameters:
            data (pd.Series): The raw time-series data.
            labeled_periods (List[Dict[str, int]]): List of dictionaries with 'start' and 'end' indices.
            label (str): The label for the periods (e.g., 'stable' or 'unstable').
            column (str): The name of the column being processed.

        Returns:
            pd.DataFrame: A DataFrame where each row corresponds to a segment with extracted features, label, and column name.
        """
        features_list = []

        for period in labeled_periods:
            start = period['start']
            end = period['end']
            segment = data[start:end+1]

            # Handle potential empty segments
            if len(segment) == 0:
                self._log(f"Empty segment detected for column '{column}' at indices {start}-{end}. Skipping.")
                continue

            try:
                segment_features = self._compute_features(segment)
                segment_features['start'] = start
                segment_features['end'] = end
                segment_features['length'] = end - start + 1
                segment_features['label'] = label
                segment_features['column'] = column

                features_list.append(segment_features)
            except Exception as e:
                self._log(f"Error extracting features for column '{column}' segment {start}-{end}: {e}")
                continue

        return pd.DataFrame(features_list)

    def _compute_features(self, segment: pd.Series) -> Dict[str, float]:
        """
        Compute the specified features for a data segment.

        Parameters:
            segment (pd.Series): A segment of the time-series data.

        Returns:
            Dict[str, float]: A dictionary of computed features.
        """
        features = {}
        x = np.arange(len(segment))  # Time component for slope calculation

        if 'mean' in self.features:
            features['mean'] = segment.mean()

        if 'std' in self.features:
            features['std'] = segment.std()

        if 'kurtosis' in self.features:
            features['kurtosis'] = kurtosis(segment, fisher=True, bias=False)

        if 'skewness' in self.features:
            features['skewness'] = skew(segment, bias=False)

        if 'slope' in self.features:
            # Using linear regression to find the slope
            slope = self._calculate_slope(x, segment.values)
            features['slope'] = slope

        # Additional features can be added here following the same pattern

        return features

    @staticmethod
    def _calculate_slope(x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the slope of the linear regression line for the given data.

        Parameters:
            x (np.ndarray): Independent variable.
            y (np.ndarray): Dependent variable.

        Returns:
            float: The slope of the regression line.
        """
        if len(x) == 0:
            return np.nan
        try:
            model = np.polyfit(x, y, 1)
            return model[0]  # Slope
        except np.RankWarning as e:
            # Handle cases where polyfit might fail
            return np.nan

    def extract_features_from_segments(
        self, 
        data: pd.DataFrame, 
        segmented_periods_dict: Dict[str, Dict[str, List[Dict[str, int]]]]
    ) -> pd.DataFrame:
        """
        Extract features from both stable and unstable periods for all specified columns.

        Parameters:
            data (pd.DataFrame): The DataFrame containing time-series data for multiple columns.
            segmented_periods_dict (Dict[str, Dict[str, List[Dict[str, int]]]]): 
                Dictionary containing 'stable_periods' and 'unstable_periods' for each column.

        Returns:
            pd.DataFrame: A DataFrame containing features for all segments across all specified columns.
        """
        features_list = []

        for column, periods in segmented_periods_dict.items():
            self._log(f"Extracting features for column: {column}")

            stable_periods = periods.get('stable_periods', [])
            unstable_periods = periods.get('unstable_periods', [])

            if not stable_periods and not unstable_periods:
                self._log(f"No stable or unstable periods found for column '{column}'. Skipping feature extraction.")
                continue

            # Extract features for stable periods
            if stable_periods:
                stable_features = self.extract_features(data[column], stable_periods, label='stable', column=column)
                features_list.append(stable_features)
                self._log(f"Extracted features for {len(stable_features)} stable segments in column '{column}'.")

            # Extract features for unstable periods
            if unstable_periods:
                unstable_features = self.extract_features(data[column], unstable_periods, label='unstable', column=column)
                features_list.append(unstable_features)
                self._log(f"Extracted features for {len(unstable_features)} unstable segments in column '{column}'.")

        if features_list:
            all_features = pd.concat(features_list, ignore_index=True)
            self._log(f"Total features extracted: {len(all_features)}.")
            return all_features.reset_index(drop=True)
        else:
            self._log("No features extracted from any columns.")
            return pd.DataFrame()  # Return an empty DataFrame if no features were extracted
