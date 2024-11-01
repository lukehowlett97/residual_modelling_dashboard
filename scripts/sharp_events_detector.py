# event_detection.py

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

class SharpEventDetector:
    """
    A class to detect sharp events such as shifts, spikes, steps, and shimmering periods in time-series GNSS residual data.
    """

    def __init__(
        self,
        shift_threshold: float = 0.5,
        classification_tolerance: float = 0.1,
        stability_window: int = 2,
        shimmering_tolerance: float = 0.1,
        min_shimmering_window: int = 3
    ):
        """
        Initialize the EventDetector with configurable parameters.

        Parameters:
            shift_threshold (float): Threshold for detecting significant shifts.
            classification_tolerance (float): Tolerance for classifying similar shifts.
            stability_window (int): Number of points to check for stability after a shift.
            shimmering_tolerance (float): Tolerance level for detecting similar alternating shifts in shimmering periods.
            min_shimmering_window (int): Minimum number of shifts required to qualify as a shimmering period.
        """
        self.shift_threshold = shift_threshold
        self.classification_tolerance = classification_tolerance
        self.stability_window = stability_window
        self.shimmering_tolerance = shimmering_tolerance
        self.min_shimmering_window = min_shimmering_window

    def events_detection(self, data_series: pd.Series) -> Dict[str, any]:
        """
        Detects sharp events in the data series and classifies them into spikes, steps, and shimmering periods.

        Parameters:
            data_series (pd.Series): The raw time-series data.

        Returns:
            Dict[str, any]: Dictionary containing shift amplitudes, shift classifications, and shimmering periods.
        """
        data_diff_series = data_series.diff().fillna(0)

        # Detect jumps
        jump_amplitudes = self.detect_jumps(data_diff_series)

        # Classify Shimmering 
        if len(jump_amplitudes) > 0:
            shimmering_periods, jump_amplitudes = self.classify_shimmering(data_series, jump_amplitudes)
            
        # # Classify shifts into spikes and steps
        
        if len(jump_amplitudes) > 0:
            jump_classification = self.classify_jumps(data_series, jump_amplitudes)

        sharp_events_dict = self.combine_events_into_dict(shimmering_periods, jump_classification)
        
        return sharp_events_dict

    def detect_jumps(self, data_diff: pd.Series) -> np.ndarray:
        """
        Detects shifts in the data using the difference series.

        Parameters:
            data_diff (pd.Series): Series containing differences between consecutive data points.

        Returns:
            np.ndarray: Array of shift amplitudes that exceed the specified threshold.
        """
        jump_indices = np.where(np.abs(data_diff) > self.shift_threshold)[0]
        jump_amplitudes = data_diff.iloc[jump_indices].values
        
        # Combine jump indices and amplitudes into a 2D array
        jumps = np.column_stack((jump_indices, jump_amplitudes))

        print(jump_indices[:5])
        print(jump_amplitudes[:5])
        print(jumps[:5])
        print(jumps[0][0])
        return jumps

    def classify_jumps(
        self,
        data_series: pd.Series,
        jump_amplitudes_arr: np.ndarray
    ) -> Dict[int, Dict[str, any]]:
        """
        Classifies jumps into spikes, steps, or unclassified based on jump amplitudes.
        """
        classification = {}
        classified_indices = set()
        
        # Ensure jump_amplitudes_arr is sorted by the first column (jump indices)
        jump_amplitudes_arr = jump_amplitudes_arr[jump_amplitudes_arr[:, 0].argsort()]
        
        # Extract jump indices and amplitudes
        shift_indices = jump_amplitudes_arr[:, 0].astype(int)
        amplitudes = jump_amplitudes_arr[:, 1]
        num_jumps = len(amplitudes)
        
        # Create a mapping from index to amplitude
        index_to_amplitude = dict(zip(shift_indices, amplitudes))
        
        i = 0
        while i < num_jumps - 1:
            current_index = shift_indices[i]
            current_amp = amplitudes[i]
            next_index = shift_indices[i + 1]
            next_amp = amplitudes[i + 1]
            
            index_gap = next_index - current_index

            # Spike detection: opposite signs, similar magnitudes, and consecutive indices
            if (
                index_gap == 1 and
                ((current_amp > 0 and next_amp < 0) or (current_amp < 0 and next_amp > 0)) and
                np.isclose(abs(current_amp), abs(next_amp), rtol=self.classification_tolerance)
            ):
                # Determine the spike peak
                spike_top_index = current_index if data_series.iloc[current_index] > data_series.iloc[next_index] else next_index
                classification[spike_top_index] = {
                    'index': spike_top_index,
                    'magnitude': index_to_amplitude[spike_top_index],
                    'type': 'spike'
                }
                # Mark these jumps as classified
                classified_indices.update([current_index, next_index])
                # Skip the next jump as it's part of the spike
                i += 2
                continue

            # Step detection: significant shift followed by stability
            # Check if there's enough data after current to check stability
            if current_index + self.stability_window < len(data_series):
                # Define the window after the current jump
                post_shift_start = current_index + 1
                post_shift_end = post_shift_start + self.stability_window

                # Ensure window does not exceed data bounds
                if post_shift_end > len(data_series):
                    post_shift_end = len(data_series)

                # Extract the window
                post_shift_values = data_series.iloc[post_shift_start:post_shift_end]

                # Check for stability (values close to the post-shift value)
                post_shift_mean = data_series.iloc[current_index]
                stability_check = np.all(
                    np.isclose(post_shift_values, post_shift_mean, atol=self.classification_tolerance)
                )

                if stability_check:
                    classification[current_index] = {
                        'index': current_index,
                        'magnitude': index_to_amplitude[current_index],
                        'type': 'step'
                    }
                    # Mark this jump as classified
                    classified_indices.add(current_index)
            
            i += 1  # Move to the next jump

        # Handle unclassified jumps
        for idx in range(num_jumps):
            index = shift_indices[idx]
            if index not in classified_indices:
                classification[index] = {
                    'index': index,
                    'magnitude': amplitudes[idx],
                    'type': 'unclassified'
                }

        return classification



    def classify_shimmering(
        self,
        data_series: pd.Series,
        jumps: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Detects periods of shimmering and removes the jumps that are part of shimmering periods from the jumps array.
        """
        shimmering_periods = []
        shimmering_jumps = []
        i = 0
        total_jumps = len(jumps)

        while i < total_jumps:
            current_index, current_amp = jumps[i]
            current_index = int(current_index)
            alternating_shifts = [jumps[i]]
            j = i + 1

            while j < total_jumps:
                next_index, next_amp = jumps[j]
                next_index = int(next_index)

                # Check for alternating directions and similar magnitudes
                if (
                    (current_amp > 0 and next_amp < 0) or 
                    (current_amp < 0 and next_amp > 0)
                ) and np.isclose(abs(current_amp), abs(next_amp), rtol=self.shimmering_tolerance):
                    # Check if the jumps are close enough in time
                    if next_index - current_index <= 2:  # Adjust as needed
                        alternating_shifts.append(jumps[j])
                        current_amp = next_amp
                        current_index = next_index
                        j += 1
                    else:
                        break
                else:
                    break

            # Check if the number of alternating shifts meets the minimum shimmering window
            if len(alternating_shifts) >= self.min_shimmering_window:
                start_shift = int(alternating_shifts[0][0])
                end_shift = int(alternating_shifts[-1][0])

                # Define the shimmering period boundaries
                start_index = start_shift - 1 if start_shift > 0 else start_shift
                end_index = end_shift + 1 if (end_shift + 1) < len(data_series) else end_shift

                shimmering_periods.append((start_index, end_index))

                # Collect shimmering jumps for removal
                shimmering_jumps.extend(alternating_shifts)

                # Move the pointer past the shimmering period
                i = j
            else:
                i += 1

        # Remove shimmering jumps from jumps array
        if shimmering_jumps:
            shimmering_indices = [int(jump[0]) for jump in shimmering_jumps]
            remaining_jumps = jumps[~np.isin(jumps[:, 0].astype(int), shimmering_indices)]
        else:
            remaining_jumps = jumps.copy()

        return shimmering_periods, remaining_jumps


    def combine_events_into_dict(
        self,
        shimmering_periods: List[Tuple[float, float]],
        jump_classification: Dict[int, Dict[str, any]]
    ) -> Dict[str, any]:
        """
        Combines shimmering periods and jump classifications into a single dictionary with headers.

        Parameters:
            shimmering_periods (List[Tuple[float, float]]): List of tuples indicating shimmering periods.
            jump_classification (Dict[int, Dict[str, any]]): Dictionary with jump classifications.

        Returns:
            Dict[str, any]: Combined dictionary with headers 'spikes', 'steps', 'unclassified', and 'shimmering'.
                            Each header contains indexes and magnitudes, except 'shimmering' which contains period tuples.
        """
        combined_dict = {
            'spikes': {},
            'steps': {},
            'unclassified': {},
            'shimmering': shimmering_periods
        }

        for jump_info in jump_classification.values():
            jump_type = jump_info['type']
            jump_index = int(jump_info['index'])
            jump_magnitude = float(jump_info['magnitude'])

            if jump_type == 'spike':
                combined_dict['spikes'][jump_index] = jump_magnitude
            elif jump_type == 'step':
                combined_dict['steps'][jump_index] = jump_magnitude
            elif jump_type == 'unclassified':
                combined_dict['unclassified'][jump_index] = jump_magnitude

        return combined_dict