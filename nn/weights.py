"""Utility functions for weights."""

from typing import Tuple

import numpy as np


def initialise_weights(size: Tuple, method: str='gauss') -> np.ndarray:
    """Randomly initialise a weight matrix of a given size

    Parameters
    ----------
    size : tuple of int
        The input x output size of the weight matrix
    method : {'gauss', 'xavier', 'xavier-average', 'he'}
        The initialization method,
        if none given a values will be drawn from a Uniform(-1, 1)
        - gauss : drawn from a Normal(0, 1)
        - xavier : drawn from a Normal(0, 1 / N_in)
        - xavier-average : drawn from a Normal(0, 2 / (N_in + N_out))
        - he : drawn from a Normal(0, 2/N_in)

    Returns
    -------
    weights : ndarray
        The weight matrix

    """
    if method == 'gauss':
        return np.random.normal(0, 1, size)
    elif method == 'xavier':
        std_dev = np.sqrt(1/size[0])
        return np.random.normal(0, std_dev, size)
    elif method == 'xavier-average':
        std_dev = 2/(size[0] + size[1])
        return np.random.normal(0, std_dev, size)
    elif method == 'he':
        std_dev = 2/size[0]
        return np.random.normal(0, std_dev, size)
    else:
        return np.random.uniform(-1, 1, size)
