"""Tests for nonlinear activation functions."""

import numpy as np
import numpy.testing as npt

from nn.activations import sigmoid, sigmoid_derivative
from nn.utils import finite_difference_derivative


def test_sigmoid():
    # Intercept at exactly 0.5
    npt.assert_equal(np.full(2, 0.5), sigmoid(np.zeros(2)))
    # Check limits
    npt.assert_allclose(sigmoid(np.array([-6, 6])), np.array([0, 1]), atol=1e-2)
    npt.assert_allclose(sigmoid(np.array([-10, 10])), np.array([0, 1]), atol=1e-4)
    # Check preserves shape
    assert sigmoid(np.zeros((2, 1))).shape == (2, 1)
    assert sigmoid(np.zeros((2, ))).shape == (2, )


def test_sigmoid_derivative():
    npt.assert_equal(sigmoid_derivative(0), 0.25)
    npt.assert_almost_equal(sigmoid_derivative(0), finite_difference_derivative(sigmoid, 0))
    npt.assert_almost_equal(sigmoid_derivative(3), finite_difference_derivative(sigmoid, 3))
    npt.assert_almost_equal(sigmoid_derivative(-3), finite_difference_derivative(sigmoid, -3))
