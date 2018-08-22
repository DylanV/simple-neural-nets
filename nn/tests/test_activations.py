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
    # Do forward the check that backward updates error correctly
    sigmoid.forward(np.asarray([0]))
    npt.assert_array_equal(sigmoid.backward(np.ones(1)), np.asarray([0.25]))

    sigmoid.forward(np.asarray([0]))
    npt.assert_almost_equal(sigmoid.backward(np.ones(1)), finite_difference_derivative(sigmoid.forward, 0))

    sigmoid.forward(np.asarray([3]))
    npt.assert_almost_equal(sigmoid.backward(np.ones(1)), finite_difference_derivative(sigmoid.forward, 3))

    sigmoid.forward(np.asarray([-3]))
    npt.assert_almost_equal(sigmoid.backward(np.ones(1)), finite_difference_derivative(sigmoid.forward, -3))


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
    # For each case do forward and then check backward changes the error correctly
    tanh.forward(np.zeros(2))
    npt.assert_array_equal(tanh.backward(np.ones(2)), np.ones(2))

    tanh.forward(np.asarray([0]))
    npt.assert_almost_equal(tanh.backward(np.asarray([1])), finite_difference_derivative(tanh.forward, 0))

    tanh.forward(np.asarray([3]))
    npt.assert_almost_equal(tanh.backward(np.asarray([1])), finite_difference_derivative(tanh.forward, 3))

    tanh.forward(np.asarray([-3]))
    npt.assert_almost_equal(tanh.backward(np.asarray([1])), finite_difference_derivative(tanh.forward, -3))


def test_relu():
    relu = ReLU()
    npt.assert_array_equal(relu.forward(np.full((3, ), -1)), np.zeros(3))
    npt.assert_array_equal(relu.forward(np.full((3, 1), 2)), np.full((3, 1), 2))
    x = np.asarray([[1, 2, -3], [-4, 5, -6], [7, 8, 9]])
    y = np.asarray([[1, 2, 0], [0, 5, 0], [7, 8, 9]])
    npt.assert_array_equal(relu.forward(x), y)


def test_relu_derivative():
    relu = ReLU()

    relu.forward(np.full((3, ), -1))
    npt.assert_array_equal(relu.backward(np.ones(3)), np.zeros(3))

    relu.forward(np.full((3, 1), 2))
    npt.assert_array_equal(relu.backward(np.ones((3, 1))), np.ones((3, 1)))

    relu.forward(np.zeros((3, 3)))
    npt.assert_array_equal(relu.backward(np.ones((3, 3))), np.ones((3, 3)))

    x = np.asarray([[0, 2, -3], [-4, 5, -6], [7, 8, 9]])
    d_x = np.asarray([[1, 1, 0], [0, 1, 0], [1, 1, 1]])
    relu.forward(x)
    npt.assert_array_equal(relu.backward(np.ones(x.shape)), d_x)

    relu.forward(np.asarray([3]))
    npt.assert_almost_equal(relu.backward(np.asarray([1])), finite_difference_derivative(relu.forward, 3))
    relu.forward(np.asarray([-3]))
    npt.assert_almost_equal(relu.backward(np.asarray([1])), finite_difference_derivative(relu.forward, -3))
