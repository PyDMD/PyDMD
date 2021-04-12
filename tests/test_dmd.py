from builtins import range
from unittest import TestCase
from pydmd.dmd import DMD
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


class TestDmd(TestCase):
    def test_shape(self):
        dmd = DMD(svd_rank=-1)
        dmd.fit(X=sample_data)
        assert dmd.modes.shape[1] == sample_data.shape[1] - 1

    def test_truncation_shape(self):
        dmd = DMD(svd_rank=3)
        dmd.fit(X=sample_data)
        assert dmd.modes.shape[1] == 3

    def test_rank(self):
        dmd = DMD(svd_rank=0.9)
        dmd.fit(X=sample_data)
        assert len(dmd.eigs) == 2

    def test_Atilde_shape(self):
        dmd = DMD(svd_rank=3)
        dmd.fit(X=sample_data)
        assert dmd.atilde.shape == (dmd.svd_rank, dmd.svd_rank)

    def test_Atilde_values(self):
        dmd = DMD(svd_rank=2)
        dmd.fit(X=sample_data)
        exact_atilde = np.array(
            [[-0.70558526 + 0.67815084j, 0.22914898 + 0.20020143j],
             [0.10459069 + 0.09137814j, -0.57730040 + 0.79022994j]])
        np.testing.assert_allclose(exact_atilde, dmd.atilde)

    def test_eigs_1(self):
        dmd = DMD(svd_rank=-1)
        dmd.fit(X=sample_data)
        assert len(dmd.eigs) == 14

    def test_eigs_2(self):
        dmd = DMD(svd_rank=5)
        dmd.fit(X=sample_data)
        assert len(dmd.eigs) == 5

    def test_eigs_3(self):
        dmd = DMD(svd_rank=2)
        dmd.fit(X=sample_data)
        expected_eigs = np.array([
            -8.09016994e-01 + 5.87785252e-01j, -4.73868662e-01 + 8.80595532e-01j
        ])
        np.testing.assert_almost_equal(dmd.eigs, expected_eigs, decimal=6)

    def test_dynamics_1(self):
        dmd = DMD(svd_rank=5)
        dmd.fit(X=sample_data)
        assert dmd.dynamics.shape == (5, sample_data.shape[1])

    def test_dynamics_2(self):
        dmd = DMD(svd_rank=1)
        dmd.fit(X=sample_data)
        expected_dynamics = np.array([[
            -2.20639502 - 9.10168802e-16j, 1.55679980 - 1.49626864e+00j,
            -0.08375915 + 2.11149018e+00j, -1.37280962 - 1.54663768e+00j,
            2.01748787 + 1.60312745e-01j, -1.53222592 + 1.25504678e+00j,
            0.23000498 - 1.92462280e+00j, 1.14289644 + 1.51396355e+00j,
            -1.83310653 - 2.93174173e-01j, 1.49222925 - 1.03626336e+00j,
            -0.35015209 + 1.74312867e+00j, -0.93504202 - 1.46738182e+00j,
            1.65485808 + 4.01263449e-01j, -1.43976061 + 8.39117825e-01j,
            0.44682540 - 1.56844403e+00j
        ]])
        np.testing.assert_allclose(dmd.dynamics, expected_dynamics)

    def test_dynamics_opt_1(self):
        dmd = DMD(svd_rank=5, opt=True)
        dmd.fit(X=sample_data)
        assert dmd.dynamics.shape == (5, sample_data.shape[1])

    def test_dynamics_opt_2(self):
        dmd = DMD(svd_rank=1, opt=True)
        dmd.fit(X=sample_data)
        expected_dynamics = np.array([[
            -4.56004133 - 6.48054238j, 7.61228319 + 1.4801793j,
            -6.37489962 + 4.11788355j, 1.70548899 - 7.22866146j,
            3.69875496 + 6.25701574j, -6.85298745 - 1.90654427j,
            6.12829151 - 3.30212967j, -2.08469012 + 6.48584004j,
            -2.92745126 - 5.99004747j, 6.12772217 + 2.24123565j,
            -5.84352626 + 2.57413711j, 2.37745273 - 5.77906544j,
            2.24158249 + 5.68989493j, -5.44023459 - 2.49457492j,
            5.53024740 - 1.92916437j
        ]])
        np.testing.assert_allclose(dmd.dynamics, expected_dynamics)

    def test_reconstructed_data(self):
        dmd = DMD()
        dmd.fit(X=sample_data)
        dmd_data = dmd.reconstructed_data
        np.testing.assert_allclose(dmd_data, sample_data)

    def test_original_time(self):
        dmd = DMD(svd_rank=2)
        dmd.fit(X=sample_data)
        expected_dict = {'dt': 1, 't0': 0, 'tend': 14}
        np.testing.assert_equal(dmd.original_time, expected_dict)

    def test_original_timesteps(self):
        dmd = DMD()
        dmd.fit(X=sample_data)
        np.testing.assert_allclose(dmd.original_timesteps,
                                   np.arange(sample_data.shape[1]))

    def test_dmd_time_1(self):
        dmd = DMD(svd_rank=2)
        dmd.fit(X=sample_data)
        expected_dict = {'dt': 1, 't0': 0, 'tend': 14}
        np.testing.assert_equal(dmd.dmd_time, expected_dict)

    def test_dmd_time_2(self):
        dmd = DMD()
        dmd.fit(X=sample_data)
        dmd.dmd_time['t0'] = 10
        dmd.dmd_time['tend'] = 14
        expected_data = sample_data[:, -5:]
        np.testing.assert_allclose(dmd.reconstructed_data, expected_data)

    def test_dmd_time_3(self):
        dmd = DMD()
        dmd.fit(X=sample_data)
        dmd.dmd_time['t0'] = 8
        dmd.dmd_time['tend'] = 11
        expected_data = sample_data[:, 8:12]
        np.testing.assert_allclose(dmd.reconstructed_data, expected_data)

    def test_dmd_time_4(self):
        dmd = DMD(svd_rank=3)
        dmd.fit(X=sample_data)
        dmd.dmd_time['t0'] = 20
        dmd.dmd_time['tend'] = 20
        expected_data = np.array([[-7.29383297e+00 - 4.90248179e-14j],
                                  [-5.69109796e+00 - 2.74068833e+00j],
                                  [3.38410649e-83 + 3.75677740e-83j]])
        np.testing.assert_almost_equal(dmd.dynamics, expected_data, decimal=6)

    def test_plot_eigs_1(self):
        dmd = DMD()
        dmd.fit(X=sample_data)
        dmd.plot_eigs(show_axes=True, show_unit_circle=True)
        plt.close()

    def test_plot_eigs_2(self):
        dmd = DMD()
        dmd.fit(X=sample_data)
        dmd.plot_eigs(show_axes=False, show_unit_circle=False)
        plt.close()

    def test_plot_modes_1(self):
        dmd = DMD()
        dmd.fit(X=sample_data)
        with self.assertRaises(ValueError):
            dmd.plot_modes_2D()

    def test_plot_modes_2(self):
        dmd = DMD(svd_rank=-1)
        dmd.fit(X=sample_data)
        dmd.plot_modes_2D((1, 2, 5), x=np.arange(20), y=np.arange(20))
        plt.close()

    def test_plot_modes_3(self):
        dmd = DMD()
        snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
        dmd.fit(X=snapshots)
        dmd.plot_modes_2D()
        plt.close()

    def test_plot_modes_4(self):
        dmd = DMD()
        snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
        dmd.fit(X=snapshots)
        dmd.plot_modes_2D(index_mode=1)
        plt.close()

    def test_plot_modes_5(self):
        dmd = DMD()
        snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
        dmd.fit(X=snapshots)
        dmd.plot_modes_2D(index_mode=1, filename='tmp.png')
        self.addCleanup(os.remove, 'tmp.1.png')

    def test_plot_snapshots_1(self):
        dmd = DMD()
        dmd.fit(X=sample_data)
        with self.assertRaises(ValueError):
            dmd.plot_snapshots_2D()

    def test_plot_snapshots_2(self):
        dmd = DMD(svd_rank=-1)
        dmd.fit(X=sample_data)
        dmd.plot_snapshots_2D((1, 2, 5), x=np.arange(20), y=np.arange(20))
        plt.close()

    def test_plot_snapshots_3(self):
        dmd = DMD()
        snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
        dmd.fit(X=snapshots)
        dmd.plot_snapshots_2D()
        plt.close()

    def test_plot_snapshots_4(self):
        dmd = DMD()
        snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
        dmd.fit(X=snapshots)
        dmd.plot_snapshots_2D(index_snap=2)
        plt.close()

    def test_plot_snapshots_5(self):
        dmd = DMD()
        snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
        dmd.fit(X=snapshots)
        dmd.plot_snapshots_2D(index_snap=2, filename='tmp.png')
        self.addCleanup(os.remove, 'tmp.2.png')

    def test_tdmd_plot(self):
        dmd = DMD(tlsq_rank=3)
        dmd.fit(X=sample_data)
        dmd.plot_eigs(show_axes=False, show_unit_circle=False)
        plt.close()

       # we check that modes are the same vector multiplied by a coefficient
    # when we rescale
    def test_rescale_mode_auto_same_modes(self):
        dmd_no_rescale = DMD(svd_rank=2, opt=True, rescale_mode=None)
        dmd_no_rescale.fit(X=sample_data)

        dmd_auto_rescale = DMD(svd_rank=2, opt=True, rescale_mode='auto')
        dmd_auto_rescale.fit(X=sample_data)

        def normalize(vector):
            return vector / np.linalg.norm(vector)

        dmd_rescale_normalized_modes = np.apply_along_axis(normalize, 0,
            dmd_auto_rescale.modes)
        dmd_no_rescale_normalized_modes = np.apply_along_axis(normalize, 0,
            dmd_no_rescale.modes)

        np.testing.assert_almost_equal(dmd_no_rescale_normalized_modes,
            dmd_rescale_normalized_modes, decimal=3)

    # we check that modes are the same vector multiplied by a coefficient
    # when we rescale
    def test_rescale_mode_custom_same_modes(self):
        dmd_no_rescale = DMD(svd_rank=2, opt=True, rescale_mode=None)
        dmd_no_rescale.fit(X=sample_data)

        dmd_rescale = DMD(svd_rank=2, opt=True, rescale_mode=
            np.linspace(5,10, 2))
        dmd_rescale.fit(X=sample_data)

        def normalize(vector):
            return vector / np.linalg.norm(vector)

        dmd_rescale_normalized_modes = np.apply_along_axis(normalize, 0,
            dmd_rescale.modes)
        dmd_no_rescale_normalized_modes = np.apply_along_axis(normalize, 0,
            dmd_no_rescale.modes)

        np.testing.assert_almost_equal(dmd_no_rescale_normalized_modes,
            dmd_rescale_normalized_modes, decimal=3)

    def test_rescale_mode_same_evolution(self):
        dmd_no_rescale = DMD(svd_rank=5, opt=True, rescale_mode=None)
        dmd_no_rescale.fit(X=sample_data)
        dmd_no_rescale.dmd_time['tend'] *= 2

        dmd_rescale = DMD(svd_rank=5, opt=True, rescale_mode=
            np.linspace(5,10, 5))
        dmd_rescale.fit(X=sample_data)
        dmd_rescale.dmd_time['tend'] *= 2

        np.testing.assert_almost_equal(dmd_rescale.reconstructed_data,
            dmd_no_rescale.reconstructed_data, decimal=6)

    def test_rescale_mode_coefficients_count_check(self):
        dmd_rescale = DMD(svd_rank=5, opt=True, rescale_mode=
            np.linspace(5,10, 6))
        with self.assertRaises(ValueError):
            dmd_rescale.fit(X=sample_data)

    def test_predict(self):
        def f1(x,t):
            return 1./np.cosh(x+3)*np.exp(2.3j*t)

        def f2(x,t):
            return 2./np.cosh(x)*np.tanh(x)*np.exp(2.8j*t)

        x = np.linspace(-2, 2, 4)
        t = np.linspace(0, 4*np.pi, 10)

        xgrid, tgrid = np.meshgrid(x, t)

        X1 = f1(xgrid, tgrid)
        X2 = f2(xgrid, tgrid)
        X = X1 + X2

        dmd = DMD()
        dmd.fit(X.T)

        expected = np.array([
            [ 0.35407111+0.31966903j,  0.0581077 -0.51616519j,
                -0.4936891 +0.36476117j,  0.70397844+0.05332291j,
                -0.56648961-0.50687223j,  0.15372065+0.74444603j,
                0.30751808-0.63550106j, -0.5633934 +0.24365451j,
                0.47550633+0.20903766j, -0.0985528 -0.46673545j],
            [ 0.52924739+0.47782492j,  0.08685642-0.77153733j,
                -0.73794122+0.54522635j,  1.05227097+0.07970435j,
                -0.8467597 -0.7576467j ,  0.22977376+1.11275987j,
                0.4596623 -0.94991449j, -0.84213164+0.3642023j ,
                0.71076254+0.3124588j , -0.14731169-0.69765229j],
            [-0.49897731-0.45049592j, -0.0818887 +0.72740958j,
                0.69573498-0.51404236j, -0.99208678-0.0751457j ,
                0.79832963+0.71431342j, -0.21663195-1.04911604j,
                -0.43337211+0.89558454j,  0.79396628-0.3433719j ,
                -0.67011078-0.29458785j,  0.13888626+0.65775036j],
            [-0.2717424 -0.2453395j , -0.04459648+0.39614632j,
                0.37889637-0.2799468j , -0.54028918-0.04092425j,
                0.43476929+0.38901417j, -0.11797748-0.57134724j,
                -0.23601389+0.48773418j,  0.43239301-0.18699989j,
                - 0.36494147 - 0.16043216j, 0.07563728 + 0.35821j]
        ])

        np.testing.assert_almost_equal(dmd.predict(X.T), expected, decimal=6)

    def test_predict_exact(self):
        dmd = DMD(exact=True)
        expected = np.load('tests/test_datasets/input_sample_predict_exact.npy')

        np.testing.assert_almost_equal(dmd
            .fit(sample_data)
            .predict(sample_data[:,20:40]), expected, decimal=6)

    def test_predict_nexact(self):
        dmd = DMD(exact=False)
        expected = np.load('tests/test_datasets/input_sample_predict_nexact.npy')

        np.testing.assert_almost_equal(dmd
            .fit(sample_data)
            .predict(sample_data[:, 10:30]), expected, decimal=6)


    def test_advanced_snapshot_parameter(self):
        dmd = DMD(svd_rank=0.99)
        dmd.fit(sample_data)

        dmd2 = DMD(svd_rank=0.99, opt=-1)
        dmd2.fit(sample_data)

        np.testing.assert_almost_equal(dmd2.reconstructed_data.real,
            dmd.reconstructed_data.real, decimal=6)


    def test_sorted_eigs_default(self):
        dmd = DMD()
        assert dmd.operator._sorted_eigs == False

    def test_sorted_eigs_set_real(self):
        dmd = DMD(sorted_eigs='real')
        assert dmd.operator._sorted_eigs == 'real'

    def test_sorted_eigs_abs_right_eigs(self):
        dmd = DMD(svd_rank=20, sorted_eigs='abs')
        dmd.fit(sample_data)

        dmd2 = DMD(svd_rank=20)
        dmd2.fit(sample_data)

        assert len(dmd.eigs) == len(dmd2.eigs)
        assert set(dmd.eigs) == set(dmd2.eigs)

        previous = dmd.eigs[0]
        for eig in dmd.eigs[1:]:
            assert abs(previous) <= abs(eig)
            previous = eig

    def test_sorted_eigs_abs_right_eigenvectors(self):
        dmd = DMD(svd_rank=20, sorted_eigs='abs')
        dmd.fit(sample_data)

        dmd2 = DMD(svd_rank=20)
        dmd2.fit(sample_data)

        for idx, eig in enumerate(dmd2.eigs):
            eigenvector = dmd2.operator.eigenvectors.T[idx]
            for idx_new, eig_new in enumerate(dmd.eigs):
                if eig_new == eig:
                    assert all(dmd.operator.eigenvectors.T[idx_new] == eigenvector)
                    break

    def test_sorted_eigs_abs_right_modes(self):
        dmd = DMD(svd_rank=20, sorted_eigs='abs')
        dmd.fit(sample_data)

        dmd2 = DMD(svd_rank=20)
        dmd2.fit(sample_data)

        for idx, eig in enumerate(dmd2.eigs):
            mode = dmd2.modes.T[idx]
            for idx_new, eig_new in enumerate(dmd.eigs):
                if eig_new == eig:
                    np.testing.assert_almost_equal(dmd.modes.T[idx_new], mode,
                        decimal=6)
                    break

    def test_sorted_eigs_real_right_eigs(self):
        dmd = DMD(svd_rank=20, sorted_eigs='real')
        dmd.fit(sample_data)

        dmd2 = DMD(svd_rank=20)
        dmd2.fit(sample_data)

        assert len(dmd.eigs) == len(dmd2.eigs)
        assert set(dmd.eigs) == set(dmd2.eigs)

        previous = complex(dmd.eigs[0])
        for eig in dmd.eigs[1:]:
            x = complex(eig)
            assert x.real > previous.real or (x.real == previous.real and x.imag >= previous.imag)
            previous = x

    def test_sorted_eigs_real_right_eigenvectors(self):
        dmd = DMD(svd_rank=20, sorted_eigs='real')
        dmd.fit(sample_data)

        dmd2 = DMD(svd_rank=20)
        dmd2.fit(sample_data)

        for idx, eig in enumerate(dmd2.eigs):
            eigenvector = dmd2.operator.eigenvectors.T[idx]
            for idx_new, eig_new in enumerate(dmd.eigs):
                if eig_new == eig:
                    assert all(dmd.operator.eigenvectors.T[idx_new] == eigenvector)
                    break

    def test_sorted_eigs_real_right_modes(self):
        dmd = DMD(svd_rank=20, sorted_eigs='real')
        dmd.fit(sample_data)

        dmd2 = DMD(svd_rank=20)
        dmd2.fit(sample_data)

        for idx, eig in enumerate(dmd2.eigs):
            mode = dmd2.modes.T[idx]
            for idx_new, eig_new in enumerate(dmd.eigs):
                if eig_new == eig:
                    np.testing.assert_almost_equal(dmd.modes.T[idx_new], mode,
                        decimal=6)
                    break

    def test_sorted_eigs_dynamics(self):
        dmd = DMD(svd_rank=20, sorted_eigs='abs')
        dmd.fit(sample_data)

        dmd2 = DMD(svd_rank=20)
        dmd2.fit(sample_data)

        for idx, eig in enumerate(dmd2.eigs):
            dynamic = dmd2.dynamics[idx]
            for idx_new, eig_new in enumerate(dmd.eigs):
                if eig_new == eig:
                    np.testing.assert_almost_equal(dmd.dynamics[idx_new],
                        dynamic, decimal=6)
                    break

    def test_sorted_eigs_frequency(self):
        dmd = DMD(svd_rank=20, sorted_eigs='abs')
        dmd.fit(sample_data)

        dmd2 = DMD(svd_rank=20)
        dmd2.fit(sample_data)

        for idx, eig in enumerate(dmd2.eigs):
            frq = dmd2.frequency[idx]
            for idx_new, eig_new in enumerate(dmd.eigs):
                if eig_new == eig:
                    np.testing.assert_almost_equal(dmd.frequency[idx_new],
                        frq, decimal=6)
                    break

    def test_sorted_eigs_amplitudes(self):
        dmd = DMD(svd_rank=20, sorted_eigs='abs')
        dmd.fit(sample_data)

        dmd2 = DMD(svd_rank=20)
        dmd2.fit(sample_data)

        for idx, eig in enumerate(dmd2.eigs):
            amp = dmd2.amplitudes[idx]
            for idx_new, eig_new in enumerate(dmd.eigs):
                if eig_new == eig:
                    np.testing.assert_almost_equal(dmd.amplitudes[idx_new],
                        amp, decimal=6)
                    break
