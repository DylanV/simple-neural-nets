"""Nonlinear activation functions."""

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute the logistic function for a given input x.

    Parameters
    ----------
    x : ndarray of float

    Returns
    -------
    activation : ndarray
       the sigmoid of x

    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Compute the derivative of the sigmoid function for the given input x.

    Parameters
    ----------
    x : ndarray of float
        Where the derivative will be evaluated

    Returns
    -------
    derivative : ndarray of float
        The derivative of the sigmoid function at x

    """
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x: np.ndarray) -> np.ndarray:
    """Compute the nonlinear tanh function for the input x.

    Parameters
    ----------
    x : ndarray of float

    Returns
    -------
    activation : ndarray of float

    """
    return -1 + 2 / (1 + np.exp(-2 * x))


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Compute the derivate of the tanh function at x.

    Parameters
    ----------
    x : ndarray of float

    Returns
    -------
    derivative : ndarray of float

    """
    return 1 - tanh(x) ** 2
