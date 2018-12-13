from __future__ import division
from past.utils import old_div
from unittest import TestCase
from pydmd import DMDc
import matplotlib.pyplot as plt
import numpy as np
import scipy


def create_system_with_B():
    snapshots = np.array([[4, 2, 1, .5, .25], [7, .7, .07, .007, .0007]])
    u = np.array([-4, -2, -1, -.5])
    B = np.array([[1, 0]]).T
    return {'snapshots': snapshots, 'u': u, 'B': B}


def create_system_without_B():
    n = 5  # dimension snapshots
    m = 15  # number snapshots
    A = scipy.linalg.helmert(n, True)
    B = np.random.rand(n, n) - .5
    x0 = np.array([0.25] * n)
    u = np.random.rand(n, m - 1) - .5
    snapshots = [x0]
    for i in range(m - 1):
        snapshots.append(A.dot(snapshots[i]) + B.dot(u[:, i]))
    snapshots = np.array(snapshots).T
    return {'snapshots': snapshots, 'u': u, 'B': B, 'A': A}


class TestDMDC(TestCase):
    def test_eigs_b_known(self):
        system = create_system_with_B()
        dmdc = DMDc(svd_rank=-1)
        dmdc.fit(system['snapshots'], system['u'], system['B'])
        real_eigs = np.array([0.1, 1.5])
        np.testing.assert_array_almost_equal(dmdc.eigs, real_eigs)

    def test_reconstruct_b_known(self):
        system = create_system_with_B()
        dmdc = DMDc(svd_rank=-1)
        dmdc.fit(system['snapshots'], system['u'], system['B'])
        np.testing.assert_array_almost_equal(dmdc.reconstructed_data(),
                                             system['snapshots'])

    def test_B_b_known(self):
        system = create_system_with_B()
        dmdc = DMDc(svd_rank=-1)
        dmdc.fit(system['snapshots'], system['u'], system['B'])
        np.testing.assert_array_almost_equal(dmdc.B, system['B'])

    def test_reconstruct_b_unknown(self):
        system = create_system_without_B()
        dmdc = DMDc(svd_rank=-1, opt=True)
        dmdc.fit(system['snapshots'], system['u'])
        np.testing.assert_array_almost_equal(
            dmdc.reconstructed_data(), system['snapshots'], decimal=6)

    def test_atilde_b_unknown(self):
        system = create_system_without_B()
        dmdc = DMDc(svd_rank=-1, opt=True)
        dmdc.fit(system['snapshots'], system['u'])
        expected_atilde = dmdc.basis.T.conj().dot(system['A']).dot(dmdc.basis)
        np.testing.assert_array_almost_equal(
            dmdc.atilde, expected_atilde, decimal=1)
