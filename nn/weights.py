"""Utility functions for weights."""

from typing import Tuple

import numpy as np

def initailise_weights(size : Tuple) -> np.ndarray:
    """Randomly initialise a weight matrix of a given size

    Parameters
    ----------
    size : tuple of int
        The input x output size of the weight matrix

    Returns
    -------
    weights : ndarray
        The weight matrix

    """
    return np.random.normal(0, 1, size)