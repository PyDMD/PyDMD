from __future__ import division
from past.utils import old_div
from unittest import TestCase
from pydmd import DMDc
import matplotlib.pyplot as plt
import numpy as np

snapshots = np.array([[4, 2, 1, .5, .25], [7, .7, .07, .007, .0007]])

control = np.array([-4, -2, -1, -.5])

b = np.array([[1], [0]])


class TestMrDmd(TestCase):
    def test_atilde_b_known(self):
        dmdc = DMDc(svd_rank=-1)
        dmdc.fit(snapshots, control, b)
        real_atilde = np.array([[1.5, 0], [0, 0.1]])
        np.testing.assert_array_almost_equal(dmdc.atilde, real_atilde)

    def test_reconstruct_b_known(self):
        dmdc = DMDc(svd_rank=-1)
        dmdc.fit(snapshots, control, b)
        np.testing.assert_array_almost_equal(dmdc.reconstructed_data,
                                             snapshots[:, 1:])

    def test_reconstruct_b_unknown(self):
        dmdc = DMDc(svd_rank=-1)
        dmdc.fit(snapshots, control)
        np.testing.assert_array_almost_equal(dmdc.reconstructed_data,
                                             snapshots[:, 1:])
    def test_btilde_b_known(self):
        dmdc = DMDc(svd_rank=-1)
        dmdc.fit(snapshots, control, b)
        np.testing.assert_array_almost_equal(dmdc.btilde, b)

    def test_btilde_b_unknown(self):
        dmdc = DMDc(svd_rank=-1)
        dmdc.fit(snapshots, control)
        expected_btilde = np.array([[-0.05836184, 0.31070992]]).T
        np.testing.assert_array_almost_equal(dmdc.btilde, expected_btilde)
