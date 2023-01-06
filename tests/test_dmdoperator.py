import os
from builtins import range

import matplotlib.pyplot as plt
import numpy as np
from pytest import raises

from pydmd.dmdoperator import DMDOperator
from pydmd.utils import compute_tlsq

# 15 snapshot with 400 data. The matrix is 400x15 and it contains
# the following data: f1 + f2 where
# f1 = lambda x,t: sech(x+3)*(1.*np.exp(1j*2.3*t))
# f2 = lambda x,t: (sech(x)*np.tanh(x))*(2.*np.exp(1j*2.8*t))
sample_data = np.load('tests/test_datasets/input_sample.npy')

def test_constructor():
    operator = DMDOperator(svd_rank=2, exact=True, forward_backward=False,
        rescale_mode='auto', sorted_eigs=False, tikhonov_regularization=None)

    assert operator._svd_rank == 2
    assert operator._exact == True
    assert operator._forward_backward == False
    assert operator._rescale_mode == 'auto'

def test_noncompute_error():
    operator = DMDOperator(svd_rank=2, exact=True, forward_backward=False,
        rescale_mode='auto', sorted_eigs=False, tikhonov_regularization=None)

    with raises(ValueError):
        operator.shape

    with raises(ValueError):
        operator.Lambda

    with raises(ValueError):
        operator.modes

    with raises(ValueError):
        operator.eigenvalues

    with raises(ValueError):
        operator.eigenvectors

    with raises(ValueError):
        operator.as_numpy_array

def test_compute_operator():
    operator = DMDOperator(svd_rank=0, exact=True, forward_backward=False,
        rescale_mode='auto', sorted_eigs=False, tikhonov_regularization=None)
    operator.compute_operator(np.ones((3, 3)), np.ones((3, 3)))

    assert operator.as_numpy_array is not None
    assert operator.eigenvalues is not None
    assert operator.eigenvectors is not None
    assert operator.modes is not None
    assert operator.Lambda is not None

# test that a value of 'auto' in rescale_mode is replaced by the singular
# values of X
def test_rescalemode_auto_singular_values():
    operator = DMDOperator(svd_rank=0, exact=True, forward_backward=False,
        rescale_mode='auto', sorted_eigs=False, tikhonov_regularization=None)
    operator.compute_operator(np.ones((3, 3)), np.ones((3, 3)))
    np.testing.assert_almost_equal(operator._rescale_mode, np.array([3.]),
        decimal=1)

def test_call():
    operator = DMDOperator(svd_rank=2, exact=True, forward_backward=False,
        rescale_mode=None, sorted_eigs=False, tikhonov_regularization=None)

    X = sample_data[:, :-1]
    Y = sample_data[:, 1:]
    X, Y = compute_tlsq(X, Y, 0)

    operator.compute_operator(X,Y)

    expected = np.array([-0.47643628 + 0.87835227j, -0.47270971 + 0.88160808j])

    np.testing.assert_almost_equal(operator(np.ones(2)), expected, decimal=6)

def test_compute_eigenquantities_wrong_rescalemode():
    operator = DMDOperator(svd_rank=0, exact=True, forward_backward=False,
        rescale_mode=4, sorted_eigs=False, tikhonov_regularization=None)
    with raises(ValueError):
        operator.compute_operator(np.ones((3, 3)), np.ones((3, 3)))

    operator = DMDOperator(svd_rank=0, exact=True, forward_backward=False,
        rescale_mode=np.ones((4,)), sorted_eigs=False, tikhonov_regularization=None)
    with raises(ValueError):
        operator.compute_operator(np.ones((3, 3)), np.ones((3, 3)))

def test_plot_operator():
    operator = DMDOperator(svd_rank=2, exact=True, forward_backward=False,
        rescale_mode=None, sorted_eigs=False, tikhonov_regularization=None)

    X = sample_data[:, :-1]
    Y = sample_data[:, 1:]
    X, Y = compute_tlsq(X, Y, 0)

    operator.compute_operator(X, Y)
    operator.plot_operator()
    plt.close()
