import ruptures as rpt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def iterative_change_point_detection(data, max_loops=5, initial_penalty=10, penalty_adjust_factor=1.1, homogeneity_threshold=0.01):
    """
    Iteratively detects change points and refines parameters based on evaluation metrics.
    
    :param data: Array of normalized GNSS residuals (e.g., reg_iono)
    :param max_loops: Maximum number of refinement loops
    :param initial_penalty: Initial penalty value for change point detection
    :param penalty_adjust_factor: Factor to adjust penalty in each iteration
    :param homogeneity_threshold: Threshold to stop if homogeneity does not improve
    :return: Final list of detected change points and evaluation metrics
    """
    penalty = initial_penalty
    previous_homogeneity = None
    change_point_history = []
    evaluation_metrics = []
    
    data = bfill_missing_data(data)
    
    for loop in range(max_loops):
        print(f"Iteration {loop + 1} - Penalty: {penalty}")
        
        # Step 1: Detect change points
        change_points = detect_change_points(data, pen=penalty)
        change_point_history.append(change_points)
        
        # Step 2: Evaluate the segmentation (homogeneity, variance, etc.)
        homogeneity_score = evaluate_segments(data, change_points)
        evaluation_metrics.append({
            'penalty': penalty,
            'homogeneity': homogeneity_score
        })
        
        # Step 3: Check for stopping criteria (convergence or max loops)
        if previous_homogeneity is not None:
            improvement = abs(homogeneity_score - previous_homogeneity)
            if improvement < homogeneity_threshold:
                print(f"Convergence reached at iteration {loop + 1}. Stopping refinement.")
                break
        
        # Step 4: Refine parameters based on evaluation
        penalty = refine_penalty(penalty, homogeneity_score, previous_homogeneity, penalty_adjust_factor)
        
        # Update previous homogeneity score
        previous_homogeneity = homogeneity_score
    
    # Final results
    return change_point_history, evaluation_metrics


def detect_change_points(data, pen=10):
    """
    Detect change points in the data using the PELT algorithm.
    
    :param data: Array of normalized GNSS residuals
    :param pen: Penalty value for change point detection
    :return: List of indices where change points occur
    """
    algo = rpt.Binseg(model="l2").fit(data['reg_iono'].values)
    change_points = algo.predict(n_bkps=pen)
    # change_points = algo.predict(pen=pen)
    return change_points


def evaluate_segments(data, change_points):
    """
    Evaluate the quality of the segments based on homogeneity or variance.
    
    :param data: Array of residuals
    :param change_points: Detected change points
    :return: Homogeneity score or other evaluation metric
    """
    segments = np.split(data, change_points[:-1])  # Split data into segments
    homogeneity_scores = [np.var(segment['reg_iono']) for segment in segments]  # Example: variance-based homogeneity
    mean_homogeneity = np.nanmean(homogeneity_scores)
    return mean_homogeneity


def refine_penalty(penalty, current_homogeneity, previous_homogeneity, adjust_factor):
    """
    Adjust the penalty parameter based on the homogeneity evaluation.
    
    :param penalty: Current penalty
    :param current_homogeneity: Current homogeneity score
    :param previous_homogeneity: Previous homogeneity score
    :param adjust_factor: Factor to increase/decrease penalty
    :return: New refined penalty value
    """
    if previous_homogeneity is not None:
        if current_homogeneity < previous_homogeneity:
            penalty /= adjust_factor  # Make penalty more sensitive
        else:
            penalty *= adjust_factor  # Make penalty less sensitive
    return penalty

def plot_change_point_iterations(data, change_point_history, evaluation_metrics):
    """
    Plot the data and change points for each iteration on separate subplots.
    
    :param data: Array of residuals (e.g., reg_iono)
    :param change_point_history: List of change points detected in each iteration
    :param evaluation_metrics: List of evaluation metrics (penalty, homogeneity) for each iteration
    """
    num_plots = len(change_point_history)
    
    # Create subplots
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(10, 3 * num_plots))
    
    if num_plots == 1:
        axes = [axes]  # Handle case where only one subplot is created
    
    for i, (change_points, metrics) in enumerate(zip(change_point_history, evaluation_metrics)):
        ax = axes[i]
        
        # Plot the original data
        ax.plot(data, label="Original Data", color='gray', alpha=0.5)
        
        # Highlight the change points
        for cp in change_points:
            ax.axvline(x=cp, color='red', linestyle='--', label='Change Point' if i == 0 else "")
        
        # Display the penalty and homogeneity for this iteration
        ax.set_title(f"Iteration {i+1} - Penalty: {metrics['penalty']}, Homogeneity: {metrics['homogeneity']:.4f}")
        ax.set_xlabel('Time')
        ax.set_ylabel('Residuals')
    
    plt.tight_layout()
    plt.show()


def bfill_missing_data(data, resample_freq = '5s'):
    """
    Backfills missing 'reg_iono' values based on missing epochs in the 'epoch' column.
    The function resamples the DataFrame at regular intervals and applies backward fill (bfill)
    to the 'reg_iono' column.

    :param data: DataFrame with 'epoch' and 'reg_iono' columns
    :param resample_freq: The frequency for resampling the epochs (e.g., '5S' for 5-second intervals)
    :return: DataFrame with backward-filled 'reg_iono' values
    """
    
    data = data.copy()
    # Ensure the 'epoch' column is a datetime type
    data['epoch'] = pd.to_datetime(data['epoch'])
    
    # Set 'epoch' as the index
    data = data.set_index('epoch')
    
    # Resample based on the specified frequency (e.g., 5 seconds)
    resampled_data = data.resample(resample_freq).asfreq()
    
    # Apply backward fill to the 'reg_iono' column
    resampled_data['reg_iono'] = resampled_data['reg_iono'].bfill()
    
    # Reset index to bring 'epoch' back as a column
    resampled_data = resampled_data.reset_index()
    
    return resampled_data