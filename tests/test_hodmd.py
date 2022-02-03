from builtins import range
from unittest import TestCase
from pydmd.hodmd import HODMD
import matplotlib.pyplot as plt
import numpy as np
import os
import pytest

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


class TestHODmd(TestCase):
    def test_shape(self):
        dmd = HODMD(svd_rank=-1, d=2, svd_rank_extra=-1)
        dmd.fit(X=sample_data)
        assert dmd.modes.shape[1] == sample_data.shape[1] - 2

    def test_truncation_shape(self):
        dmd = HODMD(svd_rank=3)
        dmd.fit(X=sample_data)
        assert dmd.modes.shape[1] == 3

    def test_rank(self):
        dmd = HODMD(svd_rank=0.9)
        dmd.fit(X=sample_data)
        assert len(dmd.eigs) == 2

    def test_Atilde_shape(self):
        dmd = HODMD(svd_rank=3)
        dmd.fit(X=sample_data)
        assert dmd.atilde.shape == (dmd.svd_rank, dmd.svd_rank)

    def test_d(self):
        single_data = np.sin(np.linspace(0, 10, 100))
        dmd = HODMD(svd_rank=-1, d=50, opt=True, svd_rank_extra=-1)
        dmd.fit(single_data)
        assert np.allclose(dmd.reconstructed_data.flatten(), single_data)
        assert dmd.d == 50

    def test_Atilde_values(self):
        dmd = HODMD(svd_rank=2)
        dmd.fit(X=sample_data)
        exact_atilde = np.array(
            [[-0.70558526 + 0.67815084j, 0.22914898 + 0.20020143j],
             [0.10459069 + 0.09137814j, -0.57730040 + 0.79022994j]])
        np.testing.assert_allclose(exact_atilde, dmd.atilde)

    def test_eigs_1(self):
        dmd = HODMD(svd_rank=-1, svd_rank_extra=-1)
        dmd.fit(X=sample_data)
        assert len(dmd.eigs) == 14

    def test_eigs_2(self):
        dmd = HODMD(svd_rank=5, svd_rank_extra=-1)
        dmd.fit(X=sample_data)
        assert len(dmd.eigs) == 5

    def test_eigs_3(self):
        dmd = HODMD(svd_rank=2)
        dmd.fit(X=sample_data)
        expected_eigs = np.array([
            -8.09016994e-01 + 5.87785252e-01j, -4.73868662e-01 + 8.80595532e-01j
        ])
        np.testing.assert_almost_equal(dmd.eigs, expected_eigs, decimal=6)

    def test_eigs_4(self):
        dmd = HODMD(svd_rank=5, svd_rank_extra=4)
        dmd.fit(X=sample_data)
        assert len(dmd.eigs) == 4

    def test_dynamics_1(self):
        dmd = HODMD(svd_rank=5, svd_rank_extra=-1)
        dmd.fit(X=sample_data)
        assert dmd.dynamics.shape == (5, sample_data.shape[1])

    def test_dynamics_2(self):
        dmd = HODMD(svd_rank=5, svd_rank_extra=4)
        dmd.fit(X=sample_data)
        assert dmd.dynamics.shape == (4, sample_data.shape[1])

    def test_dynamics_opt_1(self):
        dmd = HODMD(svd_rank=5, svd_rank_extra=-1, opt=True)
        dmd.fit(X=sample_data)
        assert dmd.dynamics.shape == (5, sample_data.shape[1])

    def test_dynamics_opt_2(self):
        dmd = HODMD(svd_rank=5, svd_rank_extra=4, opt=True)
        dmd.fit(X=sample_data)
        assert dmd.dynamics.shape == (4, sample_data.shape[1])

    def test_reconstructed_data(self):
        dmd = HODMD(d=2)
        dmd.fit(X=sample_data)
        dmd.reconstructions_of_timeindex(2)
        dmd_data = dmd.reconstructed_data
        np.testing.assert_allclose(dmd_data, sample_data)

    def test_original_time(self):
        dmd = HODMD(svd_rank=2)
        dmd.fit(X=sample_data)
        expected_dict = {'dt': 1, 't0': 0, 'tend': 14}
        np.testing.assert_equal(dmd.original_time, expected_dict)

    def test_original_timesteps(self):
        dmd = HODMD()
        dmd.fit(X=sample_data)
        np.testing.assert_allclose(dmd.original_timesteps,
                                   np.arange(sample_data.shape[1]))

    def test_dmd_time_1(self):
        dmd = HODMD(svd_rank=2)
        dmd.fit(X=sample_data)
        expected_dict = {'dt': 1, 't0': 0, 'tend': 14}
        np.testing.assert_equal(dmd.dmd_time, expected_dict)

    def test_dmd_time_2(self):
        dmd = HODMD()
        dmd.fit(X=sample_data)
        dmd.dmd_time['t0'] = 10
        dmd.dmd_time['tend'] = 14
        expected_data = sample_data[:, -5:]
        np.testing.assert_allclose(dmd.reconstructed_data, expected_data)

    def test_dmd_time_3(self):
        dmd = HODMD()
        dmd.fit(X=sample_data)
        dmd.dmd_time['t0'] = 8
        dmd.dmd_time['tend'] = 11
        expected_data = sample_data[:, 8:12]
        np.testing.assert_allclose(dmd.reconstructed_data, expected_data)

    def test_dmd_time_4(self):
        dmd = HODMD(svd_rank=3)
        dmd.fit(X=sample_data)
        dmd.dmd_time['t0'] = 20
        dmd.dmd_time['tend'] = 20
        expected_data = np.array([[7.29383297e+00 + 0.0j],
                                  [5.69109796e+00 + 2.74068833e+00j],
                                  [           0.0 + 0.0j]])
        np.testing.assert_almost_equal(dmd.dynamics, expected_data, decimal=6)

    def test_dmd_time_5(self):
        x = np.linspace(0, 10, 64)
        y = np.cos(x)*np.sin(np.cos(x)) + np.cos(x*.2)

        dmd = HODMD(svd_rank=-1, exact=True, opt=True, d=30, svd_rank_extra=-1)
        dmd.fit(y)

        dmd.original_time['dt'] = dmd.dmd_time['dt'] = x[1] - x[0]
        dmd.original_time['t0'] = dmd.dmd_time['t0'] = x[0]
        dmd.original_time['tend'] = dmd.dmd_time['tend'] = x[-1]

        # assert that the shape of the output is correct
        assert dmd.reconstructed_data.shape == (1,64)

    def test_plot_eigs_1(self):
        dmd = HODMD()
        dmd.fit(X=sample_data)
        dmd.plot_eigs(show_axes=True, show_unit_circle=True)
        plt.close()

    def test_plot_eigs_2(self):
        dmd = HODMD()
        dmd.fit(X=sample_data)
        dmd.plot_eigs(show_axes=False, show_unit_circle=False)
        plt.close()

    """
    def test_plot_modes_1(self):
        dmd = HODMD()
        dmd.fit(X=sample_data)
        with self.assertRaises(ValueError):
            dmd.plot_modes_2D()

    def test_plot_modes_2(self):
        dmd = HODMD(svd_rank=-1)
        dmd.fit(X=sample_data)
        dmd.plot_modes_2D((1, 2, 5), x=np.arange(1), y=np.arange(15))
        plt.close()

    def test_plot_modes_3(self):
        dmd = HODMD()
        snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
        dmd.fit(X=snapshots)
        dmd.plot_modes_2D()
        plt.close()

    def test_plot_modes_4(self):
        dmd = HODMD()
        snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
        dmd.fit(X=snapshots)
        dmd.plot_modes_2D(index_mode=1)
        plt.close()

    def test_plot_modes_5(self):
        dmd = HODMD()
        snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
        dmd.fit(X=snapshots)
        dmd.plot_modes_2D(index_mode=1, filename='tmp.png')
        self.addCleanup(os.remove, 'tmp.1.png')
    """

    def test_plot_snapshots_1(self):
        dmd = HODMD()
        dmd.fit(X=sample_data)
        with self.assertRaises(ValueError):
            dmd.plot_snapshots_2D()

    def test_plot_snapshots_2(self):
        dmd = HODMD(svd_rank=-1)
        dmd.fit(X=sample_data)
        dmd.plot_snapshots_2D((1, 2, 5), x=np.arange(20), y=np.arange(20))
        plt.close()

    def test_plot_snapshots_3(self):
        dmd = HODMD()
        snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
        dmd.fit(X=snapshots)
        dmd.plot_snapshots_2D()
        plt.close()

    def test_plot_snapshots_4(self):
        dmd = HODMD()
        snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
        dmd.fit(X=snapshots)
        dmd.plot_snapshots_2D(index_snap=2)
        plt.close()

    def test_plot_snapshots_5(self):
        dmd = HODMD()
        snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
        dmd.fit(X=snapshots)
        dmd.plot_snapshots_2D(index_snap=2, filename='tmp.png')
        self.addCleanup(os.remove, 'tmp.2.png')

    def test_tdmd_plot(self):
        dmd = HODMD(tlsq_rank=3)
        dmd.fit(X=sample_data)
        dmd.plot_eigs(show_axes=False, show_unit_circle=False)
        plt.close()

    def test_sorted_eigs_default(self):
        dmd = HODMD()
        assert dmd.operator._sorted_eigs == False

    def test_sorted_eigs_param(self):
        dmd = HODMD(sorted_eigs='real')
        assert dmd.operator._sorted_eigs == 'real'


    def test_reconstruction_method_constructor_error(self):
        with self.assertRaises(ValueError):
            HODMD(reconstruction_method=[1, 2, 3], d=4)

        with self.assertRaises(ValueError):
            HODMD(reconstruction_method=np.array([1, 2, 3]), d=4)

        with self.assertRaises(ValueError):
            HODMD(reconstruction_method=np.array([[1, 2, 3], [3, 4, 5]]), d=3)


    def test_reconstruction_method_default_constructor(self):
        assert HODMD()._reconstruction_method == 'first'

    def test_reconstruction_method_constructor(self):
        assert HODMD(reconstruction_method='mean')._reconstruction_method == 'mean'
        assert HODMD(reconstruction_method=[3])._reconstruction_method == [3]
        assert all(HODMD(reconstruction_method=np.array([1, 2]), d=2)._reconstruction_method == np.array([1, 2]))

    def test_nonan_nomask(self):
        dmd = HODMD(d=3)
        dmd.fit(X=sample_data)
        rec = dmd.reconstructed_data

        assert not isinstance(rec, np.ma.MaskedArray)
        assert not np.nan in rec

    def test_extract_versions_nonan(self):
        dmd = HODMD(d=3)
        dmd.fit(X=sample_data)
        for timeindex in range(sample_data.shape[1]):
            assert not np.nan in dmd.reconstructions_of_timeindex(timeindex)

    def test_rec_method_first(self):
        dmd = HODMD(d=3, reconstruction_method="first")
        dmd.fit(X=sample_data)

        rec = dmd.reconstructed_data
        allrec = dmd.reconstructions_of_timeindex()
        for i in range(rec.shape[1]):
            assert (rec[:,i] == allrec[i, min(i,dmd.d-1)]).all()

    def test_rec_method_mean(self):
        dmd = HODMD(d=3, reconstruction_method='mean')
        dmd.fit(X=sample_data)
        assert (dmd.reconstructed_data.T[2] == np.mean(dmd.reconstructions_of_timeindex(2), axis=0).T).all()

    def test_rec_method_weighted(self):
        dmd = HODMD(d=2, svd_rank_extra=-1,reconstruction_method=[10,20])
        dmd.fit(X=sample_data)
        assert (dmd.reconstructed_data.T[4] == np.average(dmd.reconstructions_of_timeindex(4), axis=0, weights=[10,20]).T).all()
    """
    def test_dynamics_opt_2(self):
        dmd = HODMD(svd_rank=1, opt=True)
        dmd.fit(X=sample_data)
        expected_dynamics = np.array([[
            -5.03688923-6.13804898j,  7.71392231+0.99781981j,
            -6.17317754+4.46645858j, 1.40580999-7.33054163j,
            3.91802381+6.17354485j, -6.93951835-1.77423468j,
            6.14189948-3.39268579j, -2.10431578+6.54348298j,
            -2.89133864-6.08093842j, 6.14488387+2.39737718j,
            -5.99329708+2.41468142j,  2.6548298 -5.74598891j,
            1.96322997+5.8815406j, -5.34888809-2.87815739j,
            5.74815609-1.53732875j
        ]])
        np.testing.assert_allclose(dmd.dynamics, expected_dynamics)
    """

    def test_scalar_func_warning(self):
        x = np.linspace(0, 10, 64)
        arr = np.cos(x) * np.sin(np.cos(x)) + np.cos(x * 0.2)
        # we check that this does not fail
        dmd = HODMD(svd_rank=1, exact=True, opt=True, d=3).fit(arr)

    def test_get_bitmask_default(self):
        dmd = HODMD(svd_rank=-1, d=5)
        dmd.fit(X=sample_data)
        assert np.all(dmd.modes_activation_bitmask == True)

    def test_set_bitmask(self):
        dmd = HODMD(svd_rank=-1, d=5)
        dmd.fit(X=sample_data)

        new_bitmask = np.full(len(dmd.amplitudes), True, dtype=bool)
        new_bitmask[[0]] = False
        dmd.modes_activation_bitmask = new_bitmask

        assert dmd.modes_activation_bitmask[0] == False
        assert np.all(dmd.modes_activation_bitmask[1:] == True)

    def test_not_fitted_get_bitmask_raises(self):
        dmd = HODMD(svd_rank=-1, d=5)
        with self.assertRaises(RuntimeError):
            print(dmd.modes_activation_bitmask)

    def test_not_fitted_set_bitmask_raises(self):
        dmd = HODMD(svd_rank=-1, d=5)
        with self.assertRaises(RuntimeError):
            dmd.modes_activation_bitmask = np.full(3, True, dtype=bool)

    def test_raise_wrong_dtype_bitmask(self):
        dmd = HODMD(svd_rank=-1, d=5)
        dmd.fit(X=sample_data)
        with self.assertRaises(RuntimeError):
            dmd.modes_activation_bitmask = np.full(3, 0.1)

    def test_fitted(self):
        dmd = HODMD(svd_rank=-1, d=5)
        assert not dmd.fitted
        dmd.fit(X=sample_data)
        assert dmd.fitted

    def test_bitmask_amplitudes(self):
        dmd = HODMD(svd_rank=-1, d=5)
        dmd.fit(X=sample_data)

        old_n_amplitudes = dmd.amplitudes.shape[0]
        retained_amplitudes = np.delete(dmd.amplitudes, [0,-1])

        new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
        new_bitmask[[0,-1]] = False
        dmd.modes_activation_bitmask = new_bitmask

        assert dmd.amplitudes.shape[0] == old_n_amplitudes - 2
        np.testing.assert_almost_equal(dmd.amplitudes, retained_amplitudes)

    def test_bitmask_eigs(self):
        dmd = HODMD(svd_rank=-1, d=5)
        dmd.fit(X=sample_data)

        old_n_eigs = dmd.eigs.shape[0]
        retained_eigs = np.delete(dmd.eigs, [0,-1])

        new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
        new_bitmask[[0,-1]] = False
        dmd.modes_activation_bitmask = new_bitmask

        assert dmd.eigs.shape[0] == old_n_eigs - 2
        np.testing.assert_almost_equal(dmd.eigs, retained_eigs)

    def test_bitmask_modes(self):
        dmd = HODMD(svd_rank=-1, d=5)
        dmd.fit(X=sample_data)

        old_n_modes = dmd.modes.shape[1]
        retained_modes = np.delete(dmd.modes, [0,-1], axis=1)

        new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
        new_bitmask[[0,-1]] = False
        dmd.modes_activation_bitmask = new_bitmask

        assert dmd.modes.shape[1] == old_n_modes - 2
        np.testing.assert_almost_equal(dmd.modes, retained_modes)

    def test_reconstructed_data(self):
        dmd = HODMD(svd_rank=-1, d=5)
        dmd.fit(X=sample_data)

        new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
        new_bitmask[[0,-1]] = False
        dmd.modes_activation_bitmask = new_bitmask

        dmd.reconstructed_data
        assert True
