import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.segmentation_tools.fixed_interval_segmentation import fixed_interval_segmentation
from src.evaluation_tools.variance_reduction import variance_reduction
from src.evaluation_tools.segment_homogeneity import segment_homogeneity


import matplotlib.pyplot as plt


#############
# FIXED INTERVALS
###############
def refine_fixed_intervals(data, initial_interval_length, refinement_factor, max_loops, homogeneity_threshold=0.01, plot=False):
    """
    Refine the fixed interval segmentation by shrinking the time intervals
    when the homogeneity score is low, up to a max number of loops.
    
    :param data: DataFrame with GNSS residual data (including 'epoch' column)
    :param initial_interval_length: Initial time interval for segmentation (e.g., '1H' for 1 hour)
    :param refinement_factor: Factor by which to reduce the interval length for refinement
    :param max_loops: Maximum number of refinement iterations
    :param homogeneity_threshold: Minimum improvement in homogeneity to continue refinement
    :param plot: Boolean flag to enable or disable plotting of segmentation after each loop
    :return: Final list of segments after refinement and a list of segmentations per loop
    """
    interval_length = initial_interval_length
    refined_segments = []
    previous_homogeneity = None
    segmentation_history = []  # To store segmentations at each loop
    evaluation_metrics = []  # To store evaluation metrics (interval length, homogeneity score)

    for loop in range(max_loops):
        # Segment the data into fixed time intervals
        segments = fixed_interval_segmentation(data, interval_length)
        
        # Evaluate the homogeneity of each segment
        homogeneity_scores = segment_homogeneity(segments)
        mean_homogeneity = np.nanmean(homogeneity_scores)

        # Save segmentation results and evaluation metrics for plotting later
        segmentation_history.append(segments)
        evaluation_metrics.append({
            'interval_length': interval_length,
            'mean_homogeneity': mean_homogeneity
        })

        # Check for convergence based on homogeneity improvement threshold
        if previous_homogeneity is not None:
            if homogeneity_has_converged(previous_homogeneity, mean_homogeneity, homogeneity_threshold):
                print(f"Convergence reached at loop {loop}. Stopping refinement.")
                refined_segments = segments
                break

        # Dynamically adjust the interval length based on homogeneity scores
        refined_interval_length = adjust_interval_length(interval_length, homogeneity_scores)
        
        # Sliding window segmentation
        # window_size = refined_interval_length  # Ensure window_size is in seconds or minutes
        # windows = sliding_window_segmentation(data, window_size, step_size=1)
        
        # Find the segment with the lowest homogeneity score
        min_score_idx = np.argmin(homogeneity_scores)
        
        # If the least homogeneous segment's score is below the threshold (indicating low homogeneity)
        if homogeneity_scores[min_score_idx] < 0.5:
            # Refine the segment by splitting it into smaller time intervals
            refined_segment = variance_reduction(segments[min_score_idx], interval_length, refinement_factor)
            
            # Update the segments list by replacing the low-homogeneity segment with the refined segments
            segments.pop(min_score_idx)
            segments.extend(refined_segment)
            
            # Dynamically adjust the refinement factor
            refinement_factor = adjust_refinement_factor(previous_homogeneity, mean_homogeneity, refinement_factor)
            
            # Reduce the interval length by the refinement factor
            interval_length = refine_interval_length(interval_length, refinement_factor)
        
        # Update the previous homogeneity score for the next iteration
        previous_homogeneity = mean_homogeneity
    
    # If no convergence, set the final segments
    if not refined_segments:
        refined_segments = segments
    
    return refined_segments, segmentation_history, evaluation_metrics


def plot_all_segmentations(data, segmentation_history, evaluation_metrics):
    """
    Plot all segmentations from different loops on a multi-subplot figure.
    :param data: DataFrame with the GNSS residual data (including 'epoch' and value columns)
    :param segmentation_history: List of segmentations from each loop
    :param evaluation_metrics: List of dictionaries containing interval length and homogeneity score for each loop
    """
    num_plots = len(segmentation_history)
    
    # Create subplots, reduce y-direction size, and add more space between them
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(10, 4 * num_plots))
    
    if 'epoch' in data.columns:
        data = data.set_index('epoch')

    for i, (segments, metrics) in enumerate(zip(segmentation_history, evaluation_metrics)):
        ax = axes[i]
        
        # Plot the original data
        ax.plot(data.index, data['reg_iono'], label='Original Data', color='gray', alpha=0.5)

        # Plot each segment
        for segment in segments:
            if 'epoch' in segment.columns:
                segment = segment.set_index('epoch')
            ax.plot(segment.index, segment['reg_iono'], label=f'Segment {i}', marker='o')

        # Set the title to include the interval length and homogeneity score
        ax.set_title(f"Segmentation at Loop {i} - Interval: {metrics['interval_length']}, Mean Homogeneity: {metrics['mean_homogeneity']:.4f}")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')

    plt.tight_layout()
    plt.show()

def refine_interval_length(interval_length, refinement_factor):
    """
    Reduce the interval length based on the refinement factor. Handles time-based intervals (e.g., '1H', '30T').
    """
    interval_duration = pd.Timedelta(interval_length).total_seconds()
    refined_duration = interval_duration / refinement_factor

    if refined_duration >= 3600:
        refined_freq = f"{int(refined_duration // 3600)}H"
    elif refined_duration >= 60:
        refined_freq = f"{int(refined_duration // 60)}T"
    else:
        refined_freq = f"{int(refined_duration)}S"
    
    return refined_freq


def adjust_interval_length(interval_length, homogeneity_scores):
    """
    Adjust interval length based on segment homogeneity. Longer for homogeneous segments, shorter for others.
    """
    if np.nanmean(homogeneity_scores) > 0.8:
        refinement_factor = 0.5  # Increase interval length
    elif np.nanmean(homogeneity_scores) < 0.3:
        refinement_factor = 2  # Decrease interval length
    else:
        refinement_factor = 1  # Keep same
    
    return refine_interval_length(interval_length, refinement_factor)


def sliding_window_segmentation(data, window_size, step_size=1, time_column='epoch'):
    """
    Apply sliding window segmentation, detecting variance in each window.
    """
    window_duration = pd.Timedelta(window_size)
    data[time_column] = pd.to_datetime(data[time_column])
    
    windows = []
    start_idx = 0
    while start_idx < len(data):
        start_time = data[time_column].iloc[start_idx]
        end_time = start_time + window_duration
        window = data[(data[time_column] >= start_time) & (data[time_column] < end_time)]
        
        if not window.empty:
            windows.append(window)
        
        start_idx += step_size
    
    return windows


def homogeneity_has_converged(previous_homogeneity, current_homogeneity, threshold=0.01):
    """
    Check if the refinement process has converged based on homogeneity improvement.
    """
    return abs(current_homogeneity - previous_homogeneity) < threshold


def adjust_refinement_factor(previous_homogeneity, current_homogeneity, refinement_factor):
    """
    Adjust the refinement factor dynamically based on improvement in homogeneity scores.
    """
    if current_homogeneity > previous_homogeneity:
        return refinement_factor * 0.9  # Decrease refinement if homogeneity improves
    return refinement_factor * 1.1  # Increase refinement if homogeneity does not improve
