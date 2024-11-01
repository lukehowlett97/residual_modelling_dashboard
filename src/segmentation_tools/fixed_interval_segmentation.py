import pandas as pd

def fixed_interval_segmentation(data, interval_length):
    
    data = data.copy()
    # Ensure that 'epoch' is in datetime format
    data['epoch'] = pd.to_datetime(data['epoch'])
    
    # Set the 'epoch' column as the index for easier time-based segmentation
    data = data.set_index('epoch')
    
    # Resample the data into time intervals based on the provided interval_length
    # interval_length should be a pandas frequency string, e.g., '5T' for 5 minutes, '1H' for 1 hour
    segments = [group for _, group in data.resample(interval_length)]
    
    return segments

