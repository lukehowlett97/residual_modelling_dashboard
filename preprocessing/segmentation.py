# segmentation.py

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from FileLogging.simple_logger import SimpleLogger


class Segmentation:
    def __init__(
        self,
        initial_window_size: int = 10,
        max_window_size: int = 50,
        variance_threshold: float = 0.01,
        moving_variance_window: int = 10,
        step_size: int = 10,
        expansion_step: int = 5,
        logger: Optional[SimpleLogger] = None
    ):
        """
        Initialize the Segmentation class with configuration parameters.

        Parameters:
            initial_window_size (int): Initial size of the window for stability detection.
            max_window_size (int): Maximum allowed window size.
            variance_threshold (float): Variance threshold to determine stability.
            moving_variance_window (int): Window size for calculating moving variance.
            step_size (int): Step size to move the window forward.
            expansion_step (int): Step size to expand the window during stability checking.
            logger (SimpleLogger, optional): Instance of the custom logger.
        """
        self.logger = logger
        self.initial_window_size = initial_window_size
        self.max_window_size = max_window_size
        self.variance_threshold = variance_threshold
        self.moving_variance_window = moving_variance_window
        self.step_size = step_size
        self.expansion_step = expansion_step

    def _log(self, message: str):
        """
        Internal method to handle logging.

        Parameters:
            message (str): The message to log.
        """
        if self.logger:
            self.logger.write_log(message)
        else:
            print(message)

    def detect_stable_periods(
        self, data: pd.DataFrame, config: Dict[str, any]
    ) -> Dict[str, Dict[str, List[Dict[str, int]]]]:
        """
        Detect stable periods for multiple columns in the provided DataFrame.

        Parameters:
            data (pd.DataFrame): The DataFrame containing time-series data for multiple columns.
            config (Dict[str, any]): Configuration dictionary containing 'columns_to_process'.

        Returns:
            Dict[str, Dict[str, List[Dict[str, int]]]]: 
                A dictionary where each key is a column name, and its value is another dictionary 
                containing 'stable_periods' and 'unstable_periods'.
        """
        combined_stable_periods = {}
        columns = config.get('columns_to_process', [])

        if not columns:
            self._log("No columns specified for processing.")
            return combined_stable_periods

        for col in columns:
            if col not in data.columns:
                self._log(f"Column '{col}' not found in data. Skipping.")
                continue

            try:
                stable_periods = self._stable_detection(data[col])
                combined_stable_periods[col] = {'stable_periods': stable_periods}
            except Exception as e:
                self._log(f"Error processing column '{col}': {e}")
                continue

        return combined_stable_periods

    def _stable_detection(self, data: pd.Series) -> List[Dict[str, int]]:
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
                    moving_segment = data[end_of_current_window: end_of_current_window + self.moving_variance_window]
                    moving_variance = np.var(moving_segment)
                else:
                    moving_segment = data[end_of_current_window:]
                    moving_variance = np.var(moving_segment) if len(moving_segment) > 0 else variance

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
            window_start = max(back_idx - self.expansion_step + 1, 0)
            window_end = back_idx + self.moving_variance_window

            if window_end > len(data):
                window_end = len(data)

            moving_segment = data[window_start:window_end]
            moving_variance = np.var(moving_segment)


            if moving_variance < self.variance_threshold:
                back_idx -= self.expansion_step
            else:
                break

        stable_start_idx = back_idx + self.expansion_step  # Adjust back to last valid index
        stable_start_idx = max(stable_start_idx, 0)  # Ensure non-negative index

        return stable_start_idx, end_idx

    def get_unstable_periods(
        self,
        data: pd.DataFrame,
        stable_periods_dict: Dict[str, Dict[str, List[Dict[str, int]]]]
    ) -> Dict[str, Dict[str, List[Dict[str, int]]]]:
        """
        Identify unstable periods between stable periods for multiple columns and update the dictionary.

        Parameters:
            data (pd.DataFrame): The DataFrame containing the time-series data.
            stable_periods_dict (Dict[str, Dict[str, List[Dict[str, int]]]]): 
                Dictionary containing stable periods for each column.

        Returns:
            Dict[str, Dict[str, List[Dict[str, int]]]]: 
                The updated dictionary with 'unstable_periods' added for each column.
        """
        for col, periods in stable_periods_dict.items():
            try:
                stable_periods = periods.get('stable_periods', [])
                data_length = len(data[col])
                unstable_periods = self._get_unstable_periods(stable_periods, data_length)
                stable_periods_dict[col]['unstable_periods'] = unstable_periods
            except Exception as e:
                self._log(f"Error identifying unstable periods for column '{col}': {e}")
                stable_periods_dict[col]['unstable_periods'] = []
                continue

        return stable_periods_dict

    def _get_unstable_periods(
        self,
        stable_periods: List[Dict[str, int]],
        data_length: int
    ) -> List[Dict[str, int]]:
        """
        Identify unstable periods between stable periods.

        Parameters:
            stable_periods (List[Dict[str, int]]): A list of stable periods with 'start' and 'end' indices.
            data_length (int): The total number of data points in the dataset.

        Returns:
            List[Dict[str, int]]: A list of dictionaries with 'start' and 'end' keys representing unstable periods.
        """
        unstable_periods = []
        if not stable_periods:
            if data_length > 0:
                unstable_periods.append({'start': 0, 'end': data_length - 1})
            return unstable_periods

        # Sort stable_periods by start index to ensure correct order
        sorted_stable = sorted(stable_periods, key=lambda x: x['start'])

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
            unstable_start = prev_end + 1
            unstable_end = data_length - 1
            unstable_periods.append({
                'start': unstable_start,
                'end': unstable_end
            })

        return unstable_periods

    def analyse_periods(
        self, data: pd.DataFrame, config: Dict[str, any]
    ) -> Dict[str, Dict[str, List[Dict[str, int]]]]:
        """
        Analyze the provided data to detect both stable and unstable periods for each specified column.

        This method orchestrates the detection of stable periods followed by the identification of
        unstable periods, compiling the results into a comprehensive dictionary.

        Parameters:
            data (pd.DataFrame): The DataFrame containing time-series data for multiple columns.
            config (Dict[str, any]): Configuration dictionary containing 'columns_to_process'.

        Returns:
            Dict[str, Dict[str, List[Dict[str, int]]]]: 
                A dictionary where each key is a column name, and its value is another dictionary 
                containing both 'stable_periods' and 'unstable_periods'.
        """
        self._log("Starting analysis of stable and unstable periods.")

        try:
            # Detect stable periods
            stable_periods_dict = self.detect_stable_periods(data, config)

            # Detect unstable periods and update the dictionary
            complete_periods_dict = self.get_unstable_periods(data, stable_periods_dict)

            self._log("Completed analysis of stable and unstable periods.")
            return complete_periods_dict

        except Exception as e:
            self._log(f"Error during analysis of periods: {e}")
            raise
