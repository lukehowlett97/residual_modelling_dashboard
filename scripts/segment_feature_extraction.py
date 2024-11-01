# feature_extraction.py

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from typing import List, Dict, Optional


class FeatureExtractor:
    """
    A class to extract statistical and temporal features from labeled segments of time-series data.
    """

    def __init__(self, features: Optional[List[str]] = None):
        """
        Initialize the FeatureExtractor with a list of features to extract.

        Parameters:
            features (List[str], optional): List of feature names to extract. 
                Defaults to ['mean', 'std', 'length', 'slope', 'kurtosis', 'skewness'].
        """
        if features is None:
            self.features = ['mean', 'std', 'length', 'slope', 'kurtosis', 'skewness']
        else:
            self.features = features

    def extract_features(
        self, 
        data: pd.Series, 
        labeled_periods: List[Dict[str, int]], 
        label: str
    ) -> pd.DataFrame:
        """
        Extract features from the labeled periods of the data.

        Parameters:
            data (pd.Series): The raw time-series data.
            labeled_periods (List[Dict[str, int]]): List of dictionaries with 'start' and 'end' indices.
            label (str): The label for the periods (e.g., 'stable' or 'unstable').

        Returns:
            pd.DataFrame: A DataFrame where each row corresponds to a segment with extracted features and label.
        """
        features_list = []

        for period in labeled_periods:
            start = period['start']
            end = period['end']
            segment = data[start:end+1]

            segment_features = self._compute_features(segment)
            segment_features['start'] = start
            segment_features['end'] = end
            segment_features['length'] = end - start + 1
            segment_features['label'] = label

            features_list.append(segment_features)

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
        model = np.polyfit(x, y, 1)
        return model[0]  # Slope

    def extract_all_features(
        self, 
        data: pd.Series, 
        stable_periods: List[Dict[str, int]], 
        unstable_periods: List[Dict[str, int]]
    ) -> pd.DataFrame:
        """
        Extract features from both stable and unstable periods.

        Parameters:
            data (pd.Series): The raw time-series data.
            stable_periods (List[Dict[str, int]]): List of stable periods.
            unstable_periods (List[Dict[str, int]]): List of unstable periods.

        Returns:
            pd.DataFrame: A DataFrame containing features for all segments with labels.
        """
        stable_features = self.extract_features(data, stable_periods, label='stable')
        unstable_features = self.extract_features(data, unstable_periods, label='unstable')

        all_features = pd.concat([stable_features, unstable_features], ignore_index=True)
        return all_features.reset_index(drop=True)
