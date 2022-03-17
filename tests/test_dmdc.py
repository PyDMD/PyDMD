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

    def test_eigs_b_unknown(self):
        system = create_system_without_B()
        dmdc = DMDc(svd_rank=3, opt=False, svd_rank_omega=4)
        dmdc.fit(system['snapshots'], system['u'])
        self.assertEqual(dmdc.eigs.shape[0], 3)

    def test_modes_b_unknown(self):
        system = create_system_without_B()
        dmdc = DMDc(svd_rank=3, opt=False, svd_rank_omega=4)
        dmdc.fit(system['snapshots'], system['u'])
        self.assertEqual(dmdc.modes.shape[1], 3)

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

    def test_get_bitmask_default(self):
        system = create_system_with_B()
        dmd = DMDc(svd_rank=-1, opt=True)
        dmd.fit(system['snapshots'], system['u'], system['B'])
        assert np.all(dmd.modes_activation_bitmask == True)

    def test_set_bitmask(self):
        system = create_system_with_B()
        dmd = DMDc(svd_rank=-1, opt=True)
        dmd.fit(system['snapshots'], system['u'], system['B'])

        new_bitmask = np.full(len(dmd.amplitudes), True, dtype=bool)
        new_bitmask[[0]] = False
        dmd.modes_activation_bitmask = new_bitmask

        assert dmd.modes_activation_bitmask[0] == False
        assert np.all(dmd.modes_activation_bitmask[1:] == True)

    def test_not_fitted_get_bitmask_raises(self):
        dmd = DMDc(svd_rank=-1, opt=True)
        with self.assertRaises(RuntimeError):
            print(dmd.modes_activation_bitmask)

    def test_not_fitted_set_bitmask_raises(self):
        dmd = DMDc(svd_rank=-1, opt=True)
        with self.assertRaises(RuntimeError):
            dmd.modes_activation_bitmask = np.full(3, True, dtype=bool)

    def test_raise_wrong_dtype_bitmask(self):
        system = create_system_with_B()
        dmd = DMDc(svd_rank=-1, opt=True)
        dmd.fit(system['snapshots'], system['u'], system['B'])
        with self.assertRaises(RuntimeError):
            dmd.modes_activation_bitmask = np.full(3, 0.1)

    def test_fitted(self):
        system = create_system_with_B()
        dmd = DMDc(svd_rank=-1, opt=True)
        assert not dmd.fitted
        dmd.fit(system['snapshots'], system['u'], system['B'])
        assert dmd.fitted

    def test_bitmask_amplitudes(self):
        system = create_system_with_B()
        dmd = DMDc(svd_rank=-1, opt=True)
        dmd.fit(system['snapshots'], system['u'], system['B'])

        old_n_amplitudes = dmd.amplitudes.shape[0]
        retained_amplitudes = np.delete(dmd.amplitudes, [0,-1])

        new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
        new_bitmask[[0,-1]] = False
        dmd.modes_activation_bitmask = new_bitmask

        assert dmd.amplitudes.shape[0] == old_n_amplitudes - 2
        np.testing.assert_almost_equal(dmd.amplitudes, retained_amplitudes)

    def test_bitmask_eigs(self):
        system = create_system_with_B()
        dmd = DMDc(svd_rank=-1, opt=True)
        dmd.fit(system['snapshots'], system['u'], system['B'])

        old_n_eigs = dmd.eigs.shape[0]
        retained_eigs = np.delete(dmd.eigs, [0,-1])

        new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
        new_bitmask[[0,-1]] = False
        dmd.modes_activation_bitmask = new_bitmask

        assert dmd.eigs.shape[0] == old_n_eigs - 2
        np.testing.assert_almost_equal(dmd.eigs, retained_eigs)

    def test_bitmask_modes(self):
        system = create_system_with_B()
        dmd = DMDc(svd_rank=-1, opt=True)
        dmd.fit(system['snapshots'], system['u'], system['B'])

        old_n_modes = dmd.modes.shape[1]
        retained_modes = np.delete(dmd.modes, [0,-1], axis=1)

        new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
        new_bitmask[[0,-1]] = False
        dmd.modes_activation_bitmask = new_bitmask

        assert dmd.modes.shape[1] == old_n_modes - 2
        np.testing.assert_almost_equal(dmd.modes, retained_modes)

    def test_reconstructed_data(self):
        system = create_system_with_B()
        dmd = DMDc(svd_rank=-1, opt=True)
        dmd.fit(system['snapshots'], system['u'], system['B'])

        new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
        new_bitmask[[0,-1]] = False
        dmd.modes_activation_bitmask = new_bitmask

        dmd.reconstructed_data
        assert True

    def test_getitem_modes(self):
        system = create_system_with_B()
        dmd = DMDc(svd_rank=-1)
        dmd.fit(system['snapshots'], system['u'], system['B'])
        old_n_modes = dmd.modes.shape[1]

        assert dmd[[0,-1]].modes.shape[1] == 2
        np.testing.assert_almost_equal(dmd[[0,-1]].modes, dmd.modes[:,[0,-1]])

        assert dmd.modes.shape[1] == old_n_modes

        assert dmd[1::2].modes.shape[1] == old_n_modes // 2
        np.testing.assert_almost_equal(dmd[1::2].modes, dmd.modes[:,1::2])

        assert dmd.modes.shape[1] == old_n_modes

        assert dmd[1].modes.shape[1] == 1
        np.testing.assert_almost_equal(np.squeeze(dmd[1].modes), dmd.modes[:,1])

        assert dmd.modes.shape[1] == old_n_modes

    def test_getitem_raises(self):
        system = create_system_with_B()
        dmd = DMDc(svd_rank=-1)
        dmd.fit(system['snapshots'], system['u'], system['B'])

        with self.assertRaises(ValueError):
            dmd[[0,1,1,0,1]]
        with self.assertRaises(ValueError):
            dmd[[True, True, False, True]]
        with self.assertRaises(ValueError):
            dmd[1.0]

    # this is a test for the correctness of the amplitudes saved in the Proxy
    # between DMDBase and the modes activation bitmask. if this test fails
    # you probably need to call allocate_proxy once again after you compute
    # the final value of the amplitudes
    def test_correct_amplitudes(self):
        system = create_system_with_B()
        dmd = DMDc(svd_rank=-1)
        dmd.fit(system['snapshots'], system['u'], system['B'])
        np.testing.assert_array_almost_equal(dmd.amplitudes, dmd._b)
