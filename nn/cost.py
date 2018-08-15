"""Cost functions."""

import numpy as np


def MSE(output: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Calculate the mean squared error between the output and target array.

    Parameters
    ----------
    output : ndarray of float
    target : ndarray of float

    Returns
    -------
    error : ndarray of float

    """
    return (1/2) * (output - target) ** 2


def MSE_derivative(output: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Calculate first derivative of the mean square error between the output and target arrays,
    with respect to the output.

    Parameters
    ----------
    output : ndarray of float
    target : ndarray of float

    Returns
    -------
    d_error : ndarray of float
        MSE derivative

    """
    return (output - target)
