"""Nonlinear activation functions."""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from nn.weights import initialise_weights


class Activation(ABC):

    @property
    def trainable(self) -> bool:
        return False

    @property
    def gradients(self) -> Optional[np.ndarray]:
        return None

    @property
    def parameters(self) -> Optional[np.ndarray]:
        return None

    @abstractmethod
    def forward(self, x: np.ndarray, mode: str='eval') -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, x: np.ndarray) -> np.ndarray:
        pass


class Linear(Activation):
    """Linear layer"""

    def __init__(self, in_size: int, out_size: int, initialisation_method: str='xavier-average'):
        self.weights = initialise_weights((in_size+1, out_size), method=initialisation_method)

        self._cached_input = None
        self._weight_gradients = np.zeros(self.weights.shape)

    @property
    def trainable(self) -> bool:
        return True

    @property
    def gradients(self) -> np.ndarray:
        return self._weight_gradients

    @property
    def parameters(self) -> np.ndarray:
        return self.weights

    def forward(self, x: np.ndarray, mode: str='eval') -> np.ndarray:
        # Add the bias ones to the input for the bias trick
        self._cached_input = np.hstack((np.ones((x.shape[0], 1)), x))
        activations = np.dot(self._cached_input, self.weights)
        return activations

    def backward(self, error: np.ndarray) -> np.ndarray:
        # Update the gradients
        self._weight_gradients = np.dot(self._cached_input.T, error)
        # Propagate back through all the weights except the bias weights
        return np.dot(error, self.weights.T)[:, 1:]


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


class Dropout(Activation):

    def __init__(self, probability):
        self.cached_input = None
        self.probability = probability

    @property
    def trainable(self) -> bool:
        return False

    def forward(self, x: np.ndarray, mode: str='eval') -> np.ndarray:
        if mode == 'eval':
            return x
        elif mode == 'train':
            dropout_mask = (np.random.rand(*x.shape) < self.probability) / self.probability
            return x * dropout_mask

    def backward(self, x: np.ndarray) -> np.ndarray:
        return x


class BatchNorm(Activation):

    def __init__(self, D, momentum=0.9):
        self.batch_mean = None
        self.batch_var = None

        self.running_mean = None
        self.running_var = None
        self.momentum = momentum

        self.gamma_beta = np.vstack((np.ones(D), np.zeros(D)))
        self.gamma_beta_grad = np.zeros(self.gamma_beta.shape)

        self.eps = 1e-8
        self.cached_input = None

    @property
    def trainable(self) -> bool:
        return True

    @property
    def parameters(self):
        return self.gamma_beta

    @property
    def gradients(self):
        return self.gamma_beta_grad

    def forward(self, x: np.ndarray, mode: str='eval') -> np.ndarray:

        gamma = self.gamma_beta[0, :]
        beta = self.gamma_beta[1, :]
        self.cached_input = x

        if mode == 'eval':
            if self.running_mean is None:
                return x
            return gamma * ((x - self.running_mean) / np.sqrt(self.running_var + self.eps)) + beta

        if mode == 'train':
            self.batch_mean = np.mean(x, axis=0)
            self.batch_var = np.var(x, axis=0)

            self.running_mean = self.batch_mean if self.running_mean is None else self.running_mean
            self.running_mean = self.momentum * self.batch_mean + (1 - self.momentum) * self.running_mean
            self.running_var = self.batch_var if self.running_var is None else self.running_var
            self.running_var = self.momentum * self.batch_var + (1 - self.momentum) * self.running_var

            return gamma * ((x - self.batch_mean) / np.sqrt(self.batch_var + self.eps)) + beta

    def backward(self, error: np.ndarray):
        tmp = error * (self.cached_input - self.batch_mean) * (self.batch_var + self.eps)**(-1/2)
        self.gamma_beta_grad[0, :] = np.sum(tmp, axis=0)
        self.gamma_beta_grad[1, :] = np.sum(error, axis=0)

        BS, _ = error.shape
        gamma = self.gamma_beta[0, :]
        x = self.cached_input
        x_mu = x - self.batch_mean
        var_eps = self.batch_var + self.eps
        dx = ((1/BS) * gamma * (var_eps ** (-1/2)) * BS * error - np.sum(error, 0)
              - x_mu * var_eps ** (-1) * np.sum(error * x_mu, 0))
        return dx


