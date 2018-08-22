"""Nonlinear activation functions."""

import numpy as np
from abc import ABC, abstractmethod


class Activation(ABC):

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, x: np.ndarray) -> np.ndarray:
        pass


class Sigmoid(Activation):
    """Logistic sigmoid activation"""

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the logistic function for a given input x.

        Parameters
        ----------
        x : ndarray of float

        Returns
        -------
        activation : ndarray
           the sigmoid of x

        """
        return self._sigmoid(x)

    def backward(self, x: np.ndarray) -> np.ndarray:
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
        return self._sigmoid(x) * (1 - self._sigmoid(x))


class Tanh(Activation):
    """Tanh activation"""

    @staticmethod
    def _tanh(x: np.ndarray) -> np.ndarray:
        return -1 + 2 / (1 + np.exp(-2 * x))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the nonlinear tanh function for the input x.

        Parameters
        ----------
        x : ndarray of float

        Returns
        -------
        activation : ndarray of float

        """
        return self._tanh(x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        """Compute the derivative of the tanh function at x.

        Parameters
        ----------
        x : ndarray of float

        Returns
        -------
        derivative : ndarray of float

        """
        return 1 - self._tanh(x) ** 2


class ReLU(Activation):
    """Rectified Linear Unit activation"""

    def forward(self, x: np.ndarray) -> np.ndarray:
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

    def backward(self, x: np.ndarray) -> np.ndarray:
        """Compute the derivative of the ReLU function with respect to it's input at x.

        Parameters
        ----------
        x : ndarray of float

        Returns
        -------
        derivative : ndarray of float

        """
        return np.asarray(x >= 0, np.float)
