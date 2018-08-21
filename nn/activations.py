"""Nonlinear activation functions."""

import numpy as np


class Sigmoid:
    """Logistic activation"""

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
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
        return 1 / (1 + np.exp(-x)) * (1 - 1 / (1 + np.exp(-x)))


class Tanh:
    """Tanh activation"""

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        """Compute the nonlinear tanh function for the input x.

        Parameters
        ----------
        x : ndarray of float

        Returns
        -------
        activation : ndarray of float

        """
        return -1 + 2 / (1 + np.exp(-2 * x))

    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        """Compute the derivative of the tanh function at x.

        Parameters
        ----------
        x : ndarray of float

        Returns
        -------
        derivative : ndarray of float

        """
        return 1 - (-1 + 2 / (1 + np.exp(-2 * x))) ** 2


class ReLU:
    """Rectified Linear Unit activation"""

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit.
        Computes max(0, x) i.e. negative values of x are set to zero.

        Parameters
        ----------
        x : ndarray of float

        Returns
        -------
        out : ndarray of float

        """
        return np.maximum(0, x)

    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        """Compute the derivative of the ReLU function with respect to it's input at x.

        Parameters
        ----------
        x : ndarray of float

        Returns
        -------
        derivative : ndarray of float

        """
        return np.asarray(x >= 0, np.float)
