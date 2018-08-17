"""Tests for nonlinear activation functions."""

import numpy as np
import numpy.testing as npt

from nn.activations import sigmoid, sigmoid_derivative, tanh, tanh_derivative, relu, relu_derivative
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
    npt.assert_array_equal(sigmoid_derivative(np.asarray([0])), np.asarray([0.25]))
    npt.assert_almost_equal(sigmoid_derivative(np.asarray([0])), finite_difference_derivative(sigmoid, 0))
    npt.assert_almost_equal(sigmoid_derivative(np.asarray([3])), finite_difference_derivative(sigmoid, 3))
    npt.assert_almost_equal(sigmoid_derivative(np.asarray([-3])), finite_difference_derivative(sigmoid, -3))


def test_tanh():
    # Intercept at exactly 0.5
    npt.assert_equal(np.zeros(2), tanh(np.zeros(2)))
    # Check limits
    npt.assert_allclose(tanh(np.array([-6, 6])), np.array([-1, 1]), atol=1e-2)
    npt.assert_allclose(tanh(np.array([-10, 10])), np.array([-1, 1]), atol=1e-4)
    # Check preserves shape
    assert tanh(np.zeros((2, 1))).shape == (2, 1)
    assert tanh(np.zeros((2, ))).shape == (2, )


def test_tanh_derivative():
    npt.assert_array_equal(tanh_derivative(np.zeros(2)), np.ones(2))
    npt.assert_almost_equal(tanh_derivative(np.asarray([0])), finite_difference_derivative(tanh, 0))
    npt.assert_almost_equal(tanh_derivative(np.asarray([3])), finite_difference_derivative(tanh, 3))
    npt.assert_almost_equal(tanh_derivative(np.asarray([-3])), finite_difference_derivative(tanh, -3))


def test_relu():
    npt.assert_array_equal(relu(np.full((3, ), -1)), np.zeros(3))
    npt.assert_array_equal(relu(np.full((3, 1), 2)), np.full((3, 1), 2))
    x = np.asarray([[1, 2, -3], [-4, 5, -6], [7, 8, 9]])
    y = np.asarray([[1, 2, 0], [0, 5, 0], [7, 8, 9]])
    npt.assert_array_equal(relu(x), y)


def test_relu_derivative():
    npt.assert_array_equal(relu_derivative(np.full((3, ), -1)), np.zeros(3))
    npt.assert_array_equal(relu_derivative(np.full((3, 1), 2)), np.ones((3, 1)))
    npt.assert_array_equal(relu_derivative(np.zeros((3, 3))), np.ones((3, 3)))
    x = np.asarray([[0, 2, -3], [-4, 5, -6], [7, 8, 9]])
    d_x = np.asarray([[1, 1, 0], [0, 1, 0], [1, 1, 1]])
    npt.assert_array_equal(relu_derivative(x), d_x)
    npt.assert_almost_equal(relu_derivative(np.asarray([3])), finite_difference_derivative(relu, 3))
    npt.assert_almost_equal(relu_derivative(np.asarray([-3])), finite_difference_derivative(relu, -3))
