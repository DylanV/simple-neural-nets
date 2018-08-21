"""Tests for nonlinear activation functions."""

import numpy as np
import numpy.testing as npt

from nn.activations import Sigmoid, Tanh, ReLU
from nn.utils import finite_difference_derivative


def test_sigmoid():
    sigmoid = Sigmoid()
    # Intercept at exactly 0.5
    npt.assert_equal(np.full(2, 0.5), sigmoid.forward(np.zeros(2)))
    # Check limits
    npt.assert_allclose(sigmoid.forward(np.array([-6, 6])), np.array([0, 1]), atol=1e-2)
    npt.assert_allclose(sigmoid.forward(np.array([-10, 10])), np.array([0, 1]), atol=1e-4)
    # Check preserves shape
    assert sigmoid.forward(np.zeros((2, 1))).shape == (2, 1)
    assert sigmoid.forward(np.zeros((2, ))).shape == (2, )


def test_sigmoid_derivative():
    sigmoid = Sigmoid()
    npt.assert_array_equal(sigmoid.backward(np.asarray([0])), np.asarray([0.25]))
    npt.assert_almost_equal(sigmoid.backward(np.asarray([0])), finite_difference_derivative(sigmoid.forward, 0))
    npt.assert_almost_equal(sigmoid.backward(np.asarray([3])), finite_difference_derivative(sigmoid.forward, 3))
    npt.assert_almost_equal(sigmoid.backward(np.asarray([-3])), finite_difference_derivative(sigmoid.forward, -3))


def test_tanh():
    tanh = Tanh()
    # Intercept at exactly 0.5
    npt.assert_equal(np.zeros(2), tanh.forward(np.zeros(2)))
    # Check limits
    npt.assert_allclose(tanh.forward(np.array([-6, 6])), np.array([-1, 1]), atol=1e-2)
    npt.assert_allclose(tanh.forward(np.array([-10, 10])), np.array([-1, 1]), atol=1e-4)
    # Check preserves shape
    assert tanh.forward(np.zeros((2, 1))).shape == (2, 1)
    assert tanh.forward(np.zeros((2, ))).shape == (2, )


def test_tanh_derivative():
    tanh = Tanh()
    npt.assert_array_equal(tanh.backward(np.zeros(2)), np.ones(2))
    npt.assert_almost_equal(tanh.backward(np.asarray([0])), finite_difference_derivative(tanh.forward, 0))
    npt.assert_almost_equal(tanh.backward(np.asarray([3])), finite_difference_derivative(tanh.forward, 3))
    npt.assert_almost_equal(tanh.backward(np.asarray([-3])), finite_difference_derivative(tanh.forward, -3))


def test_relu():
    relu = ReLU()
    npt.assert_array_equal(relu.forward(np.full((3, ), -1)), np.zeros(3))
    npt.assert_array_equal(relu.forward(np.full((3, 1), 2)), np.full((3, 1), 2))
    x = np.asarray([[1, 2, -3], [-4, 5, -6], [7, 8, 9]])
    y = np.asarray([[1, 2, 0], [0, 5, 0], [7, 8, 9]])
    npt.assert_array_equal(relu.forward(x), y)


def test_relu_derivative():
    relu = ReLU()
    npt.assert_array_equal(relu.backward(np.full((3, ), -1)), np.zeros(3))
    npt.assert_array_equal(relu.backward(np.full((3, 1), 2)), np.ones((3, 1)))
    npt.assert_array_equal(relu.backward(np.zeros((3, 3))), np.ones((3, 3)))
    x = np.asarray([[0, 2, -3], [-4, 5, -6], [7, 8, 9]])
    d_x = np.asarray([[1, 1, 0], [0, 1, 0], [1, 1, 1]])
    npt.assert_array_equal(relu.backward(x), d_x)
    npt.assert_almost_equal(relu.backward(np.asarray([3])), finite_difference_derivative(relu.forward, 3))
    npt.assert_almost_equal(relu.backward(np.asarray([-3])), finite_difference_derivative(relu.forward, -3))
