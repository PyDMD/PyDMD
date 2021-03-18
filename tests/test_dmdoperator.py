from builtins import range
from unittest import TestCase
import numpy as np
import os

from pydmd.dmdoperator import DMDOperator
from pydmd.utils import compute_tlsq

# 15 snapshot with 400 data. The matrix is 400x15 and it contains
# the following data: f1 + f2 where
# f1 = lambda x,t: sech(x+3)*(1.*np.exp(1j*2.3*t))
# f2 = lambda x,t: (sech(x)*np.tanh(x))*(2.*np.exp(1j*2.8*t))
sample_data = np.load('tests/test_datasets/input_sample.npy')

class TestDmdOperator(TestCase):
    def test_call(self):
        operator = DMDOperator(svd_rank=2, exact=True, forward_backward=False,
            rescale_mode=None)

        X = sample_data[:, :-1]
        Y = sample_data[:, 1:]
        X, Y = compute_tlsq(X, Y, 0)

        operator.compute_operator(X,Y)

        expected = np.array([-0.47643628 + 0.87835227j, -0.47270971 + 0.88160808j])

        np.testing.assert_almost_equal(operator(np.ones(2)), expected, decimal=6)
