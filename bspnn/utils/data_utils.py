"""
Utility functions for data loading, saving, and preprocessing.
"""

import pickle
import numpy as np
import tensorflow as tf


# Track if GPU has been configured to avoid reconfiguration warnings
_gpu_configured = False

def configure_gpu():
    """
    Configure GPU settings for TensorFlow.
    Uses TensorFlow 2.x GPU configuration (no InteractiveSession needed).
    Checks if GPU is available and configures memory growth.
    This function is idempotent and can be called multiple times safely.
    """
    global _gpu_configured
    
    # Only configure once to avoid warnings
    if _gpu_configured:
        return
    
    # Check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            _gpu_configured = True
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            # This is OK if it's already been set
            if "already been set" not in str(e):
                print(f"Warning: Could not set GPU memory growth: {e}")
            _gpu_configured = True
    else:
        # No GPU available, mark as configured to avoid repeated messages
        _gpu_configured = True


def pickle_data(fileN_p, dt_p):
    """
    Save data to a pickle file.
    
    Args:
        fileN_p: File path to save data
        dt_p: Data to save
    """
    file = open(fileN_p, 'wb')
    pickle.dump(dt_p, file)
    file.close()


def normalize_data(data):
    """
    Normalize data by replacing NaN values with 0.
    
    Args:
        data: numpy array with potential NaN values
        
    Returns:
        Normalized numpy array
    """
    # Convert to numpy array if not already
    data = np.asarray(data)
    
    # Create a writable copy if the array is read-only
    # This handles cases where data comes from memory-mapped files or read-only views
    if not data.flags.writeable:
        data = data.copy()
    
    return np.nan_to_num(data, 0)


def clean_file_list(file_list):
    """
    Clean and split comma-separated file list.
    
    Args:
        file_list: List of file names (may contain comma-separated strings)
        
    Returns:
        List of cleaned file names
    """
    if file_list is None:
        return []
    result = []
    for item in file_list:
        # Split by comma if it's a comma-separated string
        items = item.split(',') if isinstance(item, str) else [item]
        for fname in items:
            cleaned = fname.strip().rstrip(',')
            if cleaned:  # Only add non-empty strings
                result.append(cleaned)
    return result


def split_comma_separated(arg_list):
    """
    Split comma-separated strings in a list. 
    Handles cases where shell passes comma-separated values as single string.
    
    Args:
        arg_list: List that may contain comma-separated strings
        
    Returns:
        List with comma-separated strings split, or None if input is None
    """
    if arg_list is None:
        return None
    result = []
    for item in arg_list:
        # Split by comma and strip whitespace
        split_items = [s.strip() for s in str(item).split(',')]
        # Remove empty strings
        split_items = [s for s in split_items if s]
        result.extend(split_items)
    return result if result else None


def get_importance_index_flag(y_test_p, sampleC_for_importance_p, dataset_name=""):
    """
    Get balanced sample indices for importance analysis.
    
    Args:
        y_test_p: Test labels (binary, shape: [n_samples, 2])
        sampleC_for_importance_p: Total number of samples to select
        dataset_name: Name of dataset for error messages
        
    Returns:
        Array of selected indices
    """
    y_test_sum = sum(y_test_p)
    importance_sampleC_1 = int(sampleC_for_importance_p / 2)
    importance_sampleC_2 = int(sampleC_for_importance_p / 2)
    
    if y_test_sum[0] < importance_sampleC_1 and y_test_sum[1] >= importance_sampleC_2:
        print(f'Importance sample count error\nTest data in {dataset_name} has only {y_test_sum[0]} counts in the first label')
        importance_sampleC_1 = y_test_sum[0]
        importance_sampleC_2 = int(sampleC_for_importance_p / 2) + importance_sampleC_1 - y_test_sum[0]
    
    if y_test_sum[1] < importance_sampleC_2 and y_test_sum[0] >= importance_sampleC_1:
        print(f'Importance sample count error\nTest data in {dataset_name} has only {y_test_sum[1]} counts in the second label')
        importance_sampleC_1 = int(sampleC_for_importance_p / 2) + importance_sampleC_2 - y_test_sum[1]
        importance_sampleC_2 = y_test_sum[1]
    
    if y_test_sum[1] < importance_sampleC_2 and y_test_sum[0] < importance_sampleC_1:
        print(f'Importance sample count error\nTest data in {dataset_name} has only {y_test_sum[0]} counts in the first label and {y_test_sum[1]} counts in the second label')
        importance_sampleC_1 = y_test_sum[0]
        importance_sampleC_2 = y_test_sum[1]
    
    condition_indices = np.where(y_test_p[:, 0] == 1)[0]
    np.random.seed(609)
    sampled_indices = np.random.choice(condition_indices, size=importance_sampleC_1, replace=False)
    
    condition_indices = np.where(y_test_p[:, 1] == 1)[0]
    sampled_indices = np.concatenate((
        np.random.choice(condition_indices, size=importance_sampleC_2, replace=False),
        sampled_indices
    ), axis=0)
    
    return sampled_indices
