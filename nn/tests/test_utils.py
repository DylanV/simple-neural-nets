
import numpy.testing as npt

from nn.utils import finite_difference_derivative

def test_finite_difference():
    npt.assert_almost_equal(finite_difference_derivative(lambda x: x, 4), 1)
    npt.assert_almost_equal(finite_difference_derivative(lambda x: x ** 2, 4), 2 * 4)
