# stable_period_detection.py

import numpy as np
import pandas as pd
from typing import List, Dict


class StablePeriodDetector:
    """
    A class to detect stable periods in time-series data using an adaptive sliding window approach.
    """

    def __init__(
        self,
        initial_window_size: int = 10,
        max_window_size: int = 50,
        variance_threshold: float = 0.01,
        moving_variance_window: int = 10,
        step_size: int = 10,
        expansion_step: int = 5
    ):
        """
        Initialize the StablePeriodDetector with configurable parameters.

        Parameters:
            initial_window_size (int): The starting size of the sliding window.
            max_window_size (int): The maximum allowed window size.
            variance_threshold (float): The variance threshold to qualify a segment as stable.
            moving_variance_window (int): The window size for calculating moving variance during expansion.
            step_size (int): The step size to move the window when a stable segment is not found.
            expansion_step (int): The step size to expand the window during the expansion phase.
        """
        self.initial_window_size = initial_window_size
        self.max_window_size = max_window_size
        self.variance_threshold = variance_threshold
        self.moving_variance_window = moving_variance_window
        self.step_size = step_size
        self.expansion_step = expansion_step

    def detect_stable_periods(self, data: pd.Series) -> List[Dict[str, int]]:
        """
        Detect stable periods in the provided data series.

        Parameters:
            data (pd.Series): The time-series data to analyze.

        Returns:
            List[Dict[str, int]]: A list of dictionaries with 'start', 'end', and 'window_size' keys.
        """
        stable_segments = []
        start_idx = 0
        last_end_idx = -1  # To prevent overlapping segments

        data_values = data.values  # Convert to NumPy array for efficiency
        data_length = len(data_values)

        while start_idx < data_length:
            window_size = self.initial_window_size

            window_size = self._expand_window(
                data_values, start_idx, window_size
            )

            # Adjust window size if it exceeds the data length
            if start_idx + window_size > data_length:
                window_size = data_length - start_idx

            # Calculate the variance for the current segment
            current_variance = np.var(data_values[start_idx: start_idx + window_size])

            # If the segment qualifies as stable, refine and add to the list
            if current_variance < self.variance_threshold:
                stable_start_idx, end_idx = self._refine_backward(
                    data_values, start_idx, window_size
                )

                # Ensure no overlap with the last detected segment
                if stable_start_idx > last_end_idx:
                    if (end_idx - stable_start_idx + 1) >= self.initial_window_size:
                        stable_segments.append({
                            'start': stable_start_idx,
                            'end': end_idx,
                            'window_size': end_idx - stable_start_idx + 1
                        })
                        last_end_idx = end_idx  # Update the last end index

                # Move to the next segment after the current stable one
                start_idx = end_idx + 1
            else:
                # Move to the next unprocessed segment
                start_idx += self.step_size

        return stable_segments

    def _expand_window(
        self,
        data: np.ndarray,
        start_idx: int,
        window_size: int
    ) -> int:
        """
        Expand the window size while the variance remains below the threshold.

        Parameters:
            data (np.ndarray): The data array.
            start_idx (int): The starting index of the window.
            window_size (int): The current window size.

        Returns:
            int: The expanded window size.
        """
        max_data_index = len(data)

        while (start_idx + window_size <= max_data_index):
            segment = data[start_idx: start_idx + window_size]
            variance = np.var(segment)

            # Expand window if variance is below the threshold and max size not reached
            if variance < self.variance_threshold and window_size < self.max_window_size:
                # Calculate moving variance for the forward segment expansion
                end_of_current_window = start_idx + window_size
                if end_of_current_window + self.moving_variance_window <= max_data_index:
                    moving_segment = data[end_of_current_window : end_of_current_window + self.moving_variance_window]
                    moving_variance = np.var(moving_segment)
                else:
                    moving_variance = variance

                if moving_variance < self.variance_threshold:
                    window_size += self.expansion_step
                else:
                    break
            else:
                break

            # Early exit if window size reaches max_window_size
            if window_size >= self.max_window_size:
                break

        return window_size

    def _refine_backward(
        self,
        data: np.ndarray,
        start_idx: int,
        window_size: int
    ) -> (int, int):
        """
        Refine the stable segment by expanding backwards to include more stable points.

        Parameters:
            data (np.ndarray): The data array.
            start_idx (int): The starting index of the window.
            window_size (int): The size of the window.

        Returns:
            Tuple[int, int]: The refined start and end indices of the stable segment.
        """
        end_idx = start_idx + window_size - 1
        back_idx = start_idx - 1

        while back_idx >= 0:
            # Determine the window for moving variance calculation
            window_start = max(back_idx, 0)
            window_end = back_idx + self.moving_variance_window
            if window_end > len(data):
                window_end = len(data)

            moving_segment = data[window_start:window_end]
            moving_variance = np.var(moving_segment)

            if moving_variance < self.variance_threshold:
                back_idx -= 1
            else:
                break

        stable_start_idx = back_idx + 1  # Set the final stable start index after backward expansion
        return stable_start_idx, end_idx

    def get_unstable_periods(self, stable_periods: List[Dict[str, int]], data_length: int) -> List[Dict[str, int]]:
        """
        Identify unstable periods between stable periods.

        Parameters:
            stable_periods (List[Dict[str, int]]): A list of stable periods with 'start' and 'end' indices.
            data_length (int): The total number of data points in the dataset.

        Returns:
            List[Dict[str, int]]: A list of dictionaries with 'start' and 'end' keys representing unstable periods.
        """
        if not stable_periods:
            return [{'start': 0, 'end': data_length - 1}] if data_length > 0 else []

        # Sort stable_periods by start index to ensure correct order
        sorted_stable = sorted(stable_periods, key=lambda x: x['start'])

        unstable_periods = []
        prev_end = -1

        for period in sorted_stable:
            current_start = period['start']
            current_end = period['end']

            # Check for gap between previous end and current start
            if current_start > prev_end + 1:
                unstable_start = prev_end + 1
                unstable_end = current_start - 1
                unstable_periods.append({
                    'start': unstable_start,
                    'end': unstable_end
                })

            prev_end = current_end

        # Check for any unstable period after the last stable period
        if prev_end < data_length - 1:
            unstable_periods.append({
                'start': prev_end + 1,
                'end': data_length - 1
            })

        return unstable_periods