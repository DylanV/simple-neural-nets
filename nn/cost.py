"""Cost functions."""

from abc import ABC, abstractmethod

import numpy as np


class Cost(ABC):

    @property
    def trainable(self) -> bool:
        return False

    @abstractmethod
    def loss(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def forward(self, x: np.ndarray, mode: str='eval') -> np.ndarray:
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
        return (1 / 2) * np.sum((output - target) ** 2)

    def forward(self, x: np.ndarray, mode='eval') -> np.ndarray:
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


class SoftmaxCrossEntropy(Cost):
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
        return np.sum(np.nan_to_num(-target * np.log(output) - (1 - target) * np.log(1 - output)))

    def forward(self, x: np.ndarray, mode='eval') -> np.ndarray:
        """MSE final layer does nothing to the activations."""
        norm = np.sum(x, axis=1)[:, np.newaxis] + 1e-16
        return x / norm

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
