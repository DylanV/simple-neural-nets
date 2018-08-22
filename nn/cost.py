"""Cost functions."""

from abc import ABC, abstractmethod

import numpy as np


class Cost(ABC):

    @abstractmethod
    def loss(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        pass


class MSE(Cost):
    """Mean squared error"""

    def loss(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Calculate the mean squared error between the output and target array.

        Parameters
        ----------
        output : ndarray of float
        target : ndarray of float

        Returns
        -------
        error : ndarray of float

        """
        return (1 / 2) * (output - target) ** 2

    def forward(self, x: np.ndarray) -> np.ndarray:
        """MSE final layer does nothing to the activations."""
        return x

    def backward(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
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
        return output - target
