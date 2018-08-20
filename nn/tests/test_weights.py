
import numpy as np
import numpy.testing as npt

from nn.weights import initailise_weights

def test_weight_init():
    size = (2000, 500)
    w = initailise_weights(size, method='gauss')
    npt.assert_almost_equal(np.var(w), 1, decimal=2)
    npt.assert_almost_equal(np.mean(w), 0, decimal=2)

    w = initailise_weights(size, method='xavier')
    npt.assert_almost_equal(np.var(w), 1/size[0], decimal=2)
    npt.assert_almost_equal(np.mean(w), 0, decimal=2)

    w = initailise_weights(size, method='xavier-average')
    npt.assert_almost_equal(np.var(w), 2/(size[0] + size[1]), decimal=2)
    npt.assert_almost_equal(np.mean(w), 0, decimal=2)

    w = initailise_weights(size, method='he')
    npt.assert_almost_equal(np.var(w), 2/size[0], decimal=2)
    npt.assert_almost_equal(np.mean(w), 0, decimal=2)
