from builtins import range
from unittest import TestCase
from pytest import raises
from pydmd.cdmd import CDMD
import matplotlib.pyplot as plt
import numpy as np
import os

# 15 snapshot with 400 data. The matrix is 400x15 and it contains
# the following data: f1 + f2 where
# f1 = lambda x,t: sech(x+3)*(1.*np.exp(1j*2.3*t))
# f2 = lambda x,t: (sech(x)*np.tanh(x))*(2.*np.exp(1j*2.8*t))
sample_data = np.load('tests/test_datasets/input_sample.npy')


def create_noisy_data():
    mu = 0.
    sigma = 0.  # noise standard deviation
    m = 100  # number of snapshot
    noise = np.random.normal(mu, sigma, m)  # gaussian noise
    A = np.array([[1., 1.], [-1., 2.]])
    A /= np.sqrt(3)
    n = 2
    X = np.zeros((n, m))
    X[:, 0] = np.array([0.5, 1.])
    # evolve the system and perturb the data with noise
    for k in range(1, m):
        X[:, k] = A.dot(X[:, k - 1])
        X[:, k - 1] += noise[k - 1]
    return X


noisy_data = create_noisy_data()


class TestCDmd(TestCase):
    def test_shape(self):
        dmd = CDMD(svd_rank=-1)
        dmd.fit(X=[d for d in sample_data.T])
        assert dmd.modes.shape[1] == sample_data.shape[1] - 1

    def test_truncation_shape(self):
        dmd = CDMD(svd_rank=3)
        dmd.fit(X=sample_data)
        assert dmd.modes.shape[1] == 3

    def test_Atilde_shape(self):
        dmd = CDMD(svd_rank=3)
        dmd.fit(X=sample_data)
        assert dmd.atilde.shape == (dmd.svd_rank, dmd.svd_rank)

    def test_eigs_1(self):
        dmd = CDMD(svd_rank=-1)
        dmd.fit(X=sample_data)
        assert len(dmd.eigs) == 14

    def test_eigs_2(self):
        dmd = CDMD(svd_rank=5)
        dmd.fit(X=sample_data)
        assert len(dmd.eigs) == 5

    def test_eigs_3(self):
        dmd = CDMD(svd_rank=2)
        dmd.fit(X=sample_data)
        expected_eigs = np.array(
            [-0.47386866 + 0.88059553j, -0.80901699 + 0.58778525j])
        np.testing.assert_almost_equal(dmd.eigs, expected_eigs, decimal=6)

    def test_dynamics_1(self):
        dmd = CDMD(svd_rank=5)
        dmd.fit(X=sample_data)
        assert dmd.dynamics.shape == (5, sample_data.shape[1])

    def test_reconstructed_data(self):
        dmd = CDMD()
        dmd.fit(X=sample_data)
        dmd_data = dmd.reconstructed_data
        np.testing.assert_allclose(dmd_data, sample_data)

    def test_original_time(self):
        dmd = CDMD(svd_rank=2)
        dmd.fit(X=sample_data)
        expected_dict = {'dt': 1, 't0': 0, 'tend': 14}
        np.testing.assert_equal(dmd.original_time, expected_dict)

    def test_original_timesteps(self):
        dmd = CDMD()
        dmd.fit(X=sample_data)
        np.testing.assert_allclose(dmd.original_timesteps,
                                   np.arange(sample_data.shape[1]))

    def test_dmd_time_1(self):
        dmd = CDMD(svd_rank=2)
        dmd.fit(X=sample_data)
        expected_dict = {'dt': 1, 't0': 0, 'tend': 14}
        np.testing.assert_equal(dmd.dmd_time, expected_dict)

    def test_dmd_time_2(self):
        dmd = CDMD()
        dmd.fit(X=sample_data)
        dmd.dmd_time['t0'] = 10
        dmd.dmd_time['tend'] = 14
        expected_data = sample_data[:, -5:]
        np.testing.assert_allclose(dmd.reconstructed_data, expected_data)

    def test_dmd_time_3(self):
        dmd = CDMD()
        dmd.fit(X=sample_data)
        dmd.dmd_time['t0'] = 8
        dmd.dmd_time['tend'] = 11
        expected_data = sample_data[:, 8:12]
        np.testing.assert_allclose(dmd.reconstructed_data, expected_data)

    def test_plot_eigs_1(self):
        dmd = CDMD()
        dmd.fit(X=sample_data)
        dmd.plot_eigs(show_axes=True, show_unit_circle=True)
        plt.close()

    def test_plot_eigs_2(self):
        dmd = CDMD()
        dmd.fit(X=sample_data)
        dmd.plot_eigs(show_axes=False, show_unit_circle=False)
        plt.close()

    def test_plot_modes_1(self):
        dmd = CDMD()
        dmd.fit(X=sample_data)
        with self.assertRaises(ValueError):
            dmd.plot_modes_2D()

    def test_plot_modes_2(self):
        dmd = CDMD(svd_rank=-1)
        dmd.fit(X=sample_data)
        dmd.plot_modes_2D((1, 2, 5), x=np.arange(20), y=np.arange(20))
        plt.close()

    def test_plot_modes_3(self):
        dmd = CDMD()
        snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
        dmd.fit(X=snapshots)
        dmd.plot_modes_2D()
        plt.close()

    def test_plot_modes_4(self):
        dmd = CDMD()
        snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
        dmd.fit(X=snapshots)
        dmd.plot_modes_2D(index_mode=1)
        plt.close()

    def test_plot_modes_5(self):
        dmd = CDMD()
        snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
        dmd.fit(X=snapshots)
        dmd.plot_modes_2D(index_mode=1, filename='tmp.png')
        self.addCleanup(os.remove, 'tmp.1.png')

    def test_plot_snapshots_1(self):
        dmd = CDMD()
        dmd.fit(X=sample_data)
        with self.assertRaises(ValueError):
            dmd.plot_snapshots_2D()

    def test_plot_snapshots_2(self):
        dmd = CDMD(svd_rank=-1)
        dmd.fit(X=sample_data)
        dmd.plot_snapshots_2D((1, 2, 5), x=np.arange(20), y=np.arange(20))
        plt.close()

    def test_plot_snapshots_3(self):
        dmd = CDMD()
        snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
        dmd.fit(X=snapshots)
        dmd.plot_snapshots_2D()
        plt.close()

    def test_plot_snapshots_4(self):
        dmd = CDMD()
        snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
        dmd.fit(X=snapshots)
        dmd.plot_snapshots_2D(index_snap=2)
        plt.close()

    def test_plot_snapshots_5(self):
        dmd = CDMD()
        snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
        dmd.fit(X=snapshots)
        dmd.plot_snapshots_2D(index_snap=2, filename='tmp.png')
        self.addCleanup(os.remove, 'tmp.2.png')

    def test_tdmd_plot(self):
        dmd = CDMD(tlsq_rank=3)
        dmd.fit(X=sample_data)
        dmd.plot_eigs(show_axes=False, show_unit_circle=False)
        plt.close()

    def test_cdmd_matrix_uniform(self):
        dmd = CDMD(compression_matrix='uniform')
        dmd.fit(X=sample_data)
        error_norm = np.linalg.norm(dmd.reconstructed_data - sample_data, 1)
        assert error_norm < 1e-10

    def test_cdmd_matrix_sample(self):
        dmd = CDMD(compression_matrix='sample')
        dmd.fit(X=sample_data)
        error_norm = np.linalg.norm(dmd.reconstructed_data - sample_data, 1)
        assert error_norm < 1e-10

    def test_cdmd_matrix_normal(self):
        dmd = CDMD(compression_matrix='normal')
        dmd.fit(X=sample_data)
        error_norm = np.linalg.norm(dmd.reconstructed_data - sample_data, 1)
        assert error_norm < 1e-10

    def test_cdmd_matrix_sparse(self):
        dmd = CDMD(compression_matrix='sparse')
        dmd.fit(X=sample_data)
        error_norm = np.linalg.norm(dmd.reconstructed_data - sample_data, 1)
        assert error_norm < 1e-10

    def test_cdmd_matrix_custom(self):
        matrix = np.random.permutation(
            (sample_data.shape[1] - 3) * sample_data.shape[0]).reshape(
                sample_data.shape[1] - 3, sample_data.shape[0]).astype(float)
        matrix /= float(np.sum(matrix))
        dmd = CDMD(compression_matrix=matrix)
        dmd.fit(X=sample_data)
        error_norm = np.linalg.norm(dmd.reconstructed_data - sample_data, 1)
        assert error_norm < 1e-10

    def test_sorted_eigs_default(self):
        dmd = CDMD(compression_matrix='sparse')
        assert dmd.operator._sorted_eigs == False

    def test_sorted_eigs_param(self):
        dmd = CDMD(compression_matrix='sparse', sorted_eigs='real')
        assert dmd.operator._sorted_eigs == 'real'

    def test_get_bitmask_default(self):
        dmd = CDMD(compression_matrix='normal',)
        dmd.fit(X=sample_data)
        assert np.all(dmd.modes_activation_bitmask == True)

    def test_set_bitmask(self):
        dmd = CDMD(compression_matrix='normal')
        dmd.fit(X=sample_data)

        new_bitmask = np.full(len(dmd.amplitudes), True, dtype=bool)
        new_bitmask[[0]] = False
        dmd.modes_activation_bitmask = new_bitmask

        assert dmd.modes_activation_bitmask[0] == False
        assert np.all(dmd.modes_activation_bitmask[1:] == True)

    def test_not_fitted_get_bitmask_raises(self):
        dmd = CDMD(compression_matrix='normal')
        with self.assertRaises(RuntimeError):
            print(dmd.modes_activation_bitmask)

    def test_not_fitted_set_bitmask_raises(self):
        dmd = CDMD(compression_matrix='normal')
        with self.assertRaises(RuntimeError):
            dmd.modes_activation_bitmask = np.full(3, True, dtype=bool)

    def test_raise_wrong_dtype_bitmask(self):
        dmd = CDMD(compression_matrix='normal')
        dmd.fit(X=sample_data)
        with self.assertRaises(RuntimeError):
            dmd.modes_activation_bitmask = np.full(3, 0.1)

    def test_fitted(self):
        dmd = CDMD(compression_matrix='normal')
        assert not dmd.fitted
        dmd.fit(X=sample_data)
        assert dmd.fitted

    def test_bitmask_amplitudes(self):
        dmd = CDMD(compression_matrix='normal')
        dmd.fit(X=sample_data)

        old_n_amplitudes = dmd.amplitudes.shape[0]
        retained_amplitudes = np.delete(dmd.amplitudes, [0,-1])

        new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
        new_bitmask[[0,-1]] = False
        dmd.modes_activation_bitmask = new_bitmask

        assert dmd.amplitudes.shape[0] == old_n_amplitudes - 2
        np.testing.assert_almost_equal(dmd.amplitudes, retained_amplitudes)

    def test_bitmask_eigs(self):
        dmd = CDMD(compression_matrix='normal')
        dmd.fit(X=sample_data)

        old_n_eigs = dmd.eigs.shape[0]
        retained_eigs = np.delete(dmd.eigs, [0,-1])

        new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
        new_bitmask[[0,-1]] = False
        dmd.modes_activation_bitmask = new_bitmask

        assert dmd.eigs.shape[0] == old_n_eigs - 2
        np.testing.assert_almost_equal(dmd.eigs, retained_eigs)

    def test_bitmask_modes(self):
        dmd = CDMD(compression_matrix='normal')
        dmd.fit(X=sample_data)

        old_n_modes = dmd.modes.shape[1]
        retained_modes = np.delete(dmd.modes, [0,-1], axis=1)

        new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
        new_bitmask[[0,-1]] = False
        dmd.modes_activation_bitmask = new_bitmask

        assert dmd.modes.shape[1] == old_n_modes - 2
        np.testing.assert_almost_equal(dmd.modes, retained_modes)

    def test_getitem_modes(self):
        dmd = CDMD(compression_matrix='normal', svd_rank=10)
        dmd.fit(X=sample_data)
        old_n_modes = dmd.modes.shape[1]

        assert dmd[[0,-1]].modes.shape[1] == 2
        np.testing.assert_almost_equal(dmd[[0,-1]].modes, dmd.modes[:,[0,-1]])

        assert dmd.modes.shape[1] == old_n_modes

        assert dmd[1::2].modes.shape[1] == old_n_modes // 2
        np.testing.assert_almost_equal(dmd[1::2].modes, dmd.modes[:,1::2])

        assert dmd.modes.shape[1] == old_n_modes

        assert dmd[[1,3]].modes.shape[1] == 2
        np.testing.assert_almost_equal(dmd[[1,3]].modes, dmd.modes[:,[1,3]])

        assert dmd.modes.shape[1] == old_n_modes

        assert dmd[2].modes.shape[1] == 1
        np.testing.assert_almost_equal(np.squeeze(dmd[2].modes), dmd.modes[:,2])

        assert dmd.modes.shape[1] == old_n_modes

    def test_getitem_raises(self):
        dmd = CDMD(compression_matrix='normal')
        dmd.fit(X=sample_data)

        with self.assertRaises(ValueError):
            dmd[[0,1,1,0,1]]
        with self.assertRaises(ValueError):
            dmd[[True, True, False, True]]
        with self.assertRaises(ValueError):
            dmd[1.0]

    def test_reconstructed_data(self):
        dmd = CDMD(compression_matrix='normal')
        dmd.fit(X=sample_data)

        new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
        new_bitmask[[0,-1]] = False
        dmd.modes_activation_bitmask = new_bitmask

        dmd.reconstructed_data
        assert True
