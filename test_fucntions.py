import numpy as np
from math import sqrt

def standardize_data(arr):
    # Your code goes here
    row_means = np.mean(arr, axis=1, keepdims=True)    # shape = (n_rows, 1)
    row_stds  = np.std(arr,  axis=1, keepdims=True)    # shape = (n_rows, 1)
    
    # Subtract each row’s mean and divide by that row’s std
    return (arr - row_means) / row_stds

def min_max_scaler(values):
    """
    “Min‐max” normalization: for each x, compute (x – min) / (max – min).
    Returns a Python list of floats.
    """
    # Find min and max
    min_val = min(values)
    max_val = max(values)
    denom = max_val - min_val
    # Avoid division by zero if all values are identical
    if denom == 0:
        # If all values are the same, map them all to 0.0 (or 1.0—either is acceptable,
        # but 0.0 is a common convention when the range is zero).
        return [0.0 for _ in values]

    normalized = []
    for x in values:
        normalized.append((x - min_val) / denom)
    return normalized


def standardize(values):
    """
    “Z‐score” standardization: for each x, compute (x – μ) / σ,
    where μ is the mean of `values` and σ is the population standard deviation.
    Returns a Python list of floats.
    """
    n = len(values)
    # Compute mean
    mean_val = sum(values) / n

    # Compute population‐std = sqrt( (1/n) * Σ (x – mean)² )
    sum_sq = 0.0
    for x in values:
        diff = x - mean_val
        sum_sq += diff * diff
    variance = sum_sq / n
    stddev = sqrt(variance)

    # If stddev is zero (all values identical), return zeros
    if stddev == 0:
        return [0.0 for _ in values]

    standardized = []
    for x in values:
        standardized.append((x - mean_val) / stddev)
    return standardized