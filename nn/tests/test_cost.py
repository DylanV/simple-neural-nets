"""Tests for cost functions."""

import numpy as np
import numpy.testing as npt

from nn.cost import MSE, MSE_derivative
from nn.utils import finite_difference_derivative


def test_MSE_cost():
    npt.assert_equal(MSE(np.ones(3), np.zeros(3)), np.full(3, 0.5))
    npt.assert_equal(MSE(np.full(3, 2), np.zeros(3)), np.full(3, 2))


def test_MES_derivative():
    npt.assert_equal(MSE_derivative(np.ones(3), np.zeros(3)), np.full(3, 1))
    npt.assert_equal(MSE_derivative(np.full(3, 2), np.zeros(3)), np.full(3, 2))

    # Check that the derivative is with respect to the output by fixing the target using finite difference
    target = np.random.random((4,4))
    output = np.random.random((4,4))
    def fixed_target_MSE(output):
        return MSE(output, target)
    npt.assert_almost_equal(MSE_derivative(output, target), finite_difference_derivative(fixed_target_MSE, output))
