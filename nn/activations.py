"""Nonlinear activation functions."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from nn.weights import initialise_weights


class Activation(ABC):

    @property
    def trainable(self) -> bool:
        return False

    @property
    def gradients(self) -> List[np.ndarray]:
        return []

    @property
    def parameters(self) -> List[np.ndarray]:
        return []

    @abstractmethod
    def forward(self, x: np.ndarray, mode: str='eval') -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, x: np.ndarray) -> np.ndarray:
        pass


class Linear(Activation):
    """Linear layer"""

    def __init__(self, in_size: int, out_size: int, initialisation_method: str='xavier-average'):
        self.weights = initialise_weights((in_size, out_size), method=initialisation_method)
        self.biases = np.zeros((1, out_size))

        self._cached_input = None
        self._weight_gradients = np.zeros(self.weights.shape)
        self._bias_gradients = np.zeros(self.biases.shape)

    @property
    def trainable(self) -> bool:
        return True

    @property
    def gradients(self) -> List[np.ndarray]:
        return [self._weight_gradients, self._bias_gradients]

    @property
    def parameters(self) -> List[np.ndarray]:
        return [self.weights, self.biases]

    def forward(self, x: np.ndarray, mode: str='eval') -> np.ndarray:
        activations = np.dot(x, self.weights) + self.biases
        self._cached_input = x
        return activations

    def backward(self, error: np.ndarray) -> np.ndarray:
        # Update the gradients
        self._bias_gradients = np.sum(error, 0)
        self._weight_gradients = np.dot(self._cached_input.T, error)
        return np.dot(error, self.weights.T)


class Sigmoid(Activation):
    """Logistic sigmoid activation"""

    def __init__(self):
        self.cached_input = None

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def forward(self, x: np.ndarray, mode: str='eval') -> np.ndarray:
        """Compute the logistic function for a given input x.

        Parameters
        ----------
        x : ndarray of float
        mode : {'eval', 'train'}
            Whether the model is in training or not

        Returns
        -------
        activation : ndarray
           the sigmoid of x

        """
        self.cached_input = x
        return self._sigmoid(x)

    def backward(self, error: np.ndarray) -> np.ndarray:
        """Compute the derivative of the sigmoid function for the given input x.

        Parameters
        ----------
        error : ndarray of float
            Where the derivative will be evaluated

        Returns
        -------
        derivative : ndarray of float
            The derivative of the sigmoid function at x

        """
        derivative = self._sigmoid(self.cached_input) * (1 - self._sigmoid(self.cached_input))
        return error * derivative


class Tanh(Activation):
    """Tanh activation"""

    def __init__(self):
        self._cached_input = None

    @staticmethod
    def _tanh(x: np.ndarray) -> np.ndarray:
        return -1 + 2 / (1 + np.exp(-2 * x))

    def forward(self, x: np.ndarray, mode: str='eval') -> np.ndarray:
        """Compute the nonlinear tanh function for the input x.

        Parameters
        ----------
        x : ndarray of float
        mode : {'eval', 'train'}
            Whether the model is in training or not

        Returns
        -------
        activation : ndarray of float

        """
        self._cached_input = x
        return self._tanh(x)

    def backward(self, error: np.ndarray) -> np.ndarray:
        """Compute the derivative of the tanh function at x.

        Parameters
        ----------
        error : ndarray of float

        Returns
        -------
        derivative : ndarray of float

        """
        return error * (1 - self._tanh(self._cached_input) ** 2)


class ReLU(Activation):
    """Rectified Linear Unit activation"""

    def __init__(self):
        self.cached_input = None

    def forward(self, x: np.ndarray, mode: str='eval') -> np.ndarray:
        """Rectified Linear Unit.
        Computes max(0, x) i.e. negative values of x are set to zero.

        Parameters
        ----------
        x : ndarray of float
        mode : {'eval', 'train'}
            Whether the model is in training or not

        Returns
        -------
        out : ndarray of float

        """
        self.cached_input = x
        return np.maximum(0, x)

    def backward(self, error: np.ndarray) -> np.ndarray:
        """Compute the derivative of the ReLU function with respect to it's input at x.

        Parameters
        ----------
        error : ndarray of float

        Returns
        -------
        derivative : ndarray of float

        """
        return error * np.asarray(self.cached_input >= 0, np.float)
