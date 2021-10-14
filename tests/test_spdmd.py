from unittest import TestCase
from pydmd import SpDMD, DMD
import scipy.io
import numpy as np

data = np.load("tests/test_datasets/heat_90.npy")
gammas = [1.0e-1, 0.5, 2, 5, 10, 20, 40, 50, 100]


class TestSpDmd(TestCase):
    def test_number_nonzero_amplitudes_rho1(self):
        zeros = np.load("tests/test_datasets/zero_amplitudes_rho1.npy")

        for gm, z in zip(gammas, zeros.T):
            assert all(
                SpDMD(
                    svd_rank=30,
                    gamma=gm,
                    release_memory=False,
                )
                .fit(data)
                ._find_zero_amplitudes()
                == z
            )

    def test_number_nonzero_amplitudes_rho1e4(self):
        zeros = np.load("tests/test_datasets/zero_amplitudes_rho1e4.npy")

        for gm, z in zip(gammas, zeros.T):
            assert all(
                SpDMD(
                    svd_rank=30,
                    gamma=gm,
                    rho=1.0e4,
                    release_memory=False,
                )
                .fit(data)
                ._find_zero_amplitudes()
                == z
            )

    def test_rho(self):
        assert SpDMD().rho == 1
        assert SpDMD(rho=10).rho == 10

    def test_maxiter(self):
        assert SpDMD()._max_iterations == 10000
        assert SpDMD(max_iterations=2)._max_iterations == 2

    def test_gamma(self):
        assert SpDMD().gamma == 10
        assert SpDMD(gamma=2).gamma == 2

    def test_exact(self):
        assert SpDMD().exact == True

    def test_abstol(self):
        assert SpDMD()._abs_tol == 1.0e-6
        assert SpDMD(abs_tolerance=1.0e10)._abs_tol == 1.0e10

    def test_reltol(self):
        assert SpDMD()._rel_tol == 1.0e-4
        assert SpDMD(rel_tolerance=1.0e10)._rel_tol == 1.0e10

    def test_verbose(self):
        assert SpDMD()._verbose == True
        assert SpDMD(verbose=False)._verbose == False

    def test_enforce_zero(self):
        assert SpDMD()._enforce_zero == True
        assert SpDMD(enforce_zero=False)._enforce_zero == False

    def test_release_memory(self):
        assert SpDMD()._release_memory == True
        assert SpDMD(release_memory=False)._release_memory == False

    def test_zero_tolerance(self):
        assert SpDMD()._zero_absolute_tolerance == 1.0e-12
        assert SpDMD(
            zero_absolute_tolerance=1.0e-6)._zero_absolute_tolerance == 1.0e-6

    def test_zero_tolerance_no_zeros(self):
        zero_amps = (
            SpDMD(zero_absolute_tolerance=1.0e30, release_memory=False)
            .fit(data)
            ._find_zero_amplitudes()
        )
        assert all(zero_amps)

    def test_zero_tolerance_all_zeros(self):
        zero_amps = (
            SpDMD(zero_absolute_tolerance=0, release_memory=False)
            .fit(data)
            ._find_zero_amplitudes()
        )
        assert all(np.logical_not(zero_amps))

    def test_release_memory_releases(self):
        o = SpDMD(release_memory=True).fit(data)
        assert o._P is None
        assert o._Plow is None
        assert o._q is None

        o = SpDMD(release_memory=False).fit(data)
        assert o._P is not None
        assert o._Plow is not None
        assert o._q is not None

    def test_update_lagrangian(self):
        alpha = np.random.rand(10)
        beta = np.random.rand(10)
        lmbd = np.random.rand(10)

        np.testing.assert_allclose(
            SpDMD(rho=0)._update_lagrangian(alpha, beta, lmbd), lmbd
        )
        np.testing.assert_allclose(
            SpDMD()._update_lagrangian(alpha, alpha, lmbd), lmbd
        )

    def test_update_alpha(self):
        o = SpDMD(release_memory=False).fit(data)

        beta = np.random.rand(len(o.amplitudes))
        lmbd = np.random.rand(len(o.amplitudes))

        uk = beta - lmbd / o.rho

        rhs = np.linalg.solve(o._Plow, o._q + uk * o.rho / 2)
        lhs = o._Plow.conj().T

        np.testing.assert_allclose(lhs.dot(o._update_alpha(beta, lmbd)), rhs)
