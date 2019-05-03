from builtins import range
from unittest import TestCase
from pydmd.dmd import OptDMD
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


class TestOptDmd(TestCase):
    def test_shape(self):
        optdmd = OptDMD(svd_rank=-1)
        optdmd.fit(X=sample_data)
        assert optdmd.modes.shape[1] == sample_data.shape[1] - 1

    def test_truncation_shape(self):
        optdmd = OptDMD(svd_rank=3)
        optdmd.fit(X=sample_data)
        assert optdmd.modes.shape[1] == 3

    def test_rank(self):
        optdmd = OptDMD(svd_rank=0.9)
        optdmd.fit(X=sample_data)
        assert len(optdmd.eigs) == 2

    def test_Atilde_shape(self):
        optdmd = OptDMD(svd_rank=3)
        optdmd.fit(X=sample_data)
        assert optdmd.atilde.shape == (optdmd.svd_rank, optdmd.svd_rank)

    def test_Atilde_values(self):
        optdmd = OptDMD(svd_rank=2)
        optdmd.fit(X=sample_data)
        exact_atilde = np.array(
            [[-0.70558526 + 0.67815084j, 0.22914898 + 0.20020143j],
             [0.10459069 + 0.09137814j, -0.57730040 + 0.79022994j]])
        np.testing.assert_allclose(exact_atilde, optdmd.atilde)

    def test_eigs_1(self):
        optdmd = OptDMD(svd_rank=-1)
        optdmd.fit(X=sample_data)
        assert len(optdmd.eigs) == 14

    def test_eigs_2(self):
        optdmd = OptDMD(svd_rank=5)
        optdmd.fit(X=sample_data)
        assert len(optdmd.eigs) == 5

    def test_eigs_3(self):
        optdmd = OptDMD(svd_rank=2)
        optdmd.fit(X=sample_data)
        expected_eigs = np.array([
            -8.09016994e-01 + 5.87785252e-01j, -4.73868662e-01 + 8.80595532e-01j
        ])
        np.testing.assert_almost_equal(optdmd.eigs, expected_eigs, decimal=6)

    # def test_dynamics_1(self):
    #     dmd = DMD(svd_rank=5)
    #     dmd.fit(X=sample_data)
    #     assert dmd.dynamics.shape == (5, sample_data.shape[1])
    #
    # def test_dynamics_2(self):
    #     dmd = DMD(svd_rank=1)
    #     dmd.fit(X=sample_data)
    #     expected_dynamics = np.array([[
    #         -2.20639502 - 9.10168802e-16j, 1.55679980 - 1.49626864e+00j,
    #         -0.08375915 + 2.11149018e+00j, -1.37280962 - 1.54663768e+00j,
    #         2.01748787 + 1.60312745e-01j, -1.53222592 + 1.25504678e+00j,
    #         0.23000498 - 1.92462280e+00j, 1.14289644 + 1.51396355e+00j,
    #         -1.83310653 - 2.93174173e-01j, 1.49222925 - 1.03626336e+00j,
    #         -0.35015209 + 1.74312867e+00j, -0.93504202 - 1.46738182e+00j,
    #         1.65485808 + 4.01263449e-01j, -1.43976061 + 8.39117825e-01j,
    #         0.44682540 - 1.56844403e+00j
    #     ]])
    #     np.testing.assert_allclose(dmd.dynamics, expected_dynamics)
    #
    # def test_dynamics_opt_1(self):
    #     dmd = DMD(svd_rank=5, opt=True)
    #     dmd.fit(X=sample_data)
    #     assert dmd.dynamics.shape == (5, sample_data.shape[1])
    #
    # def test_dynamics_opt_2(self):
    #     dmd = DMD(svd_rank=1, opt=True)
    #     dmd.fit(X=sample_data)
    #     expected_dynamics = np.array([[
    #         -4.56004133 - 6.48054238j, 7.61228319 + 1.4801793j,
    #         -6.37489962 + 4.11788355j, 1.70548899 - 7.22866146j,
    #         3.69875496 + 6.25701574j, -6.85298745 - 1.90654427j,
    #         6.12829151 - 3.30212967j, -2.08469012 + 6.48584004j,
    #         -2.92745126 - 5.99004747j, 6.12772217 + 2.24123565j,
    #         -5.84352626 + 2.57413711j, 2.37745273 - 5.77906544j,
    #         2.24158249 + 5.68989493j, -5.44023459 - 2.49457492j,
    #         5.53024740 - 1.92916437j
    #     ]])
    #     np.testing.assert_allclose(dmd.dynamics, expected_dynamics)

    # def test_reconstructed_data(self):
    #     dmd = DMD()
    #     dmd.fit(X=sample_data)
    #     dmd_data = dmd.reconstructed_data
    #     np.testing.assert_allclose(dmd_data, sample_data)

    # def test_original_time(self):
    #     dmd = DMD(svd_rank=2)
    #     dmd.fit(X=sample_data)
    #     expected_dict = {'dt': 1, 't0': 0, 'tend': 14}
    #     np.testing.assert_equal(dmd.original_time, expected_dict)
    #
    # def test_original_timesteps(self):
    #     dmd = DMD()
    #     dmd.fit(X=sample_data)
    #     np.testing.assert_allclose(dmd.original_timesteps,
    #                                np.arange(sample_data.shape[1]))
    #
    # def test_dmd_time_1(self):
    #     dmd = DMD(svd_rank=2)
    #     dmd.fit(X=sample_data)
    #     expected_dict = {'dt': 1, 't0': 0, 'tend': 14}
    #     np.testing.assert_equal(dmd.dmd_time, expected_dict)
    #
    # def test_dmd_time_2(self):
    #     dmd = DMD()
    #     dmd.fit(X=sample_data)
    #     dmd.dmd_time['t0'] = 10
    #     dmd.dmd_time['tend'] = 14
    #     expected_data = sample_data[:, -5:]
    #     np.testing.assert_allclose(dmd.reconstructed_data, expected_data)
    #
    # def test_dmd_time_3(self):
    #     dmd = DMD()
    #     dmd.fit(X=sample_data)
    #     dmd.dmd_time['t0'] = 8
    #     dmd.dmd_time['tend'] = 11
    #     expected_data = sample_data[:, 8:12]
    #     np.testing.assert_allclose(dmd.reconstructed_data, expected_data)
    #
    # def test_dmd_time_4(self):
    #     dmd = DMD(svd_rank=3)
    #     dmd.fit(X=sample_data)
    #     dmd.dmd_time['t0'] = 20
    #     dmd.dmd_time['tend'] = 20
    #     expected_data = np.array([[-7.29383297e+00 - 4.90248179e-14j],
    #                               [-5.69109796e+00 - 2.74068833e+00j],
    #                               [3.38410649e-83 + 3.75677740e-83j]])
    #     np.testing.assert_almost_equal(dmd.dynamics, expected_data, decimal=6)
    #
    # def test_plot_eigs_1(self):
    #     dmd = DMD()
    #     dmd.fit(X=sample_data)
    #     dmd.plot_eigs(show_axes=True, show_unit_circle=True)
    #     plt.close()
    #
    # def test_plot_eigs_2(self):
    #     dmd = DMD()
    #     dmd.fit(X=sample_data)
    #     dmd.plot_eigs(show_axes=False, show_unit_circle=False)
    #     plt.close()
    #
    # def test_plot_modes_1(self):
    #     dmd = DMD()
    #     dmd.fit(X=sample_data)
    #     with self.assertRaises(ValueError):
    #         dmd.plot_modes_2D()
    #
    # def test_plot_modes_2(self):
    #     dmd = DMD(svd_rank=-1)
    #     dmd.fit(X=sample_data)
    #     dmd.plot_modes_2D((1, 2, 5), x=np.arange(20), y=np.arange(20))
    #     plt.close()
    #
    # def test_plot_modes_3(self):
    #     dmd = DMD()
    #     snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
    #     dmd.fit(X=snapshots)
    #     dmd.plot_modes_2D()
    #     plt.close()
    #
    # def test_plot_modes_4(self):
    #     dmd = DMD()
    #     snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
    #     dmd.fit(X=snapshots)
    #     dmd.plot_modes_2D(index_mode=1)
    #     plt.close()
    #
    # def test_plot_modes_5(self):
    #     dmd = DMD()
    #     snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
    #     dmd.fit(X=snapshots)
    #     dmd.plot_modes_2D(index_mode=1, filename='tmp.png')
    #     self.addCleanup(os.remove, 'tmp.1.png')
    #
    # def test_plot_snapshots_1(self):
    #     dmd = DMD()
    #     dmd.fit(X=sample_data)
    #     with self.assertRaises(ValueError):
    #         dmd.plot_snapshots_2D()
    #
    # def test_plot_snapshots_2(self):
    #     dmd = DMD(svd_rank=-1)
    #     dmd.fit(X=sample_data)
    #     dmd.plot_snapshots_2D((1, 2, 5), x=np.arange(20), y=np.arange(20))
    #     plt.close()
    #
    # def test_plot_snapshots_3(self):
    #     dmd = DMD()
    #     snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
    #     dmd.fit(X=snapshots)
    #     dmd.plot_snapshots_2D()
    #     plt.close()
    #
    # def test_plot_snapshots_4(self):
    #     dmd = DMD()
    #     snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
    #     dmd.fit(X=snapshots)
    #     dmd.plot_snapshots_2D(index_snap=2)
    #     plt.close()
    #
    # def test_plot_snapshots_5(self):
    #     dmd = DMD()
    #     snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
    #     dmd.fit(X=snapshots)
    #     dmd.plot_snapshots_2D(index_snap=2, filename='tmp.png')
    #     self.addCleanup(os.remove, 'tmp.2.png')
    #
    # def test_tdmd_plot(self):
    #     dmd = DMD(tlsq_rank=3)
    #     dmd.fit(X=sample_data)
    #     dmd.plot_eigs(show_axes=False, show_unit_circle=False)
    #     plt.close()
