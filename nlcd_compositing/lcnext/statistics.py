import numpy as np

from typing import Callable


########################################################
# Generalized Functionality
########################################################
def zonal(value_array: np.ndarray, label_array: np.ndarray, stat_func: Callable, **kwargs) -> dict:
    """
    Calculate statistics from the value array for each region identified in the label_array

    Where the unique values in the label_array identify which region in the value_array a
    pixel belongs to

    Both input arrays must be of the same size/shape

    The stat_func must take in a ndarray as the first argument, and all other KWARGS
    will be passed to it
    """
    return {val: stat_func(value_array[label_array == val], **kwargs)
            for val in nanunique(label_array)}


def nanunique(arr: np.ndarray, **kwargs) -> np.ndarray:
    """
    Return the unique values in the given array, excluding any nan values
    kwargs are passed to numpy.unique
    """
    return np.unique(arr[~np.isnan(arr)], **kwargs)
