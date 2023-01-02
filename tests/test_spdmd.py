import numpy as np
import scipy.io
from pytest import raises

from pydmd import DMD, SpDMD

data = np.load("tests/test_datasets/heat_90.npy")
gammas = [1.0e-1, 0.5, 2, 5, 10, 20, 40, 50, 100]


def test_number_nonzero_amplitudes_rho1():
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

def test_number_nonzero_amplitudes_rho1e4():
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


def test_spdmd_amplitudes_rho1():
    amps = np.load("tests/test_datasets/spdmd_amplitudes_rho1.npy")

    for gm, z in zip(gammas, amps.T):
        np.testing.assert_allclose(
            SpDMD(
                svd_rank=30,
                gamma=gm,
            )
            .fit(data)
            .amplitudes,
            z,
            rtol=2,
            atol=1.e-6
        )

def test_spdmd_amplitudes_rho1e4():
    amps = np.load("tests/test_datasets/spdmd_amplitudes_rho1e4.npy")

    for gm, z in zip(gammas, amps.T):
        np.testing.assert_allclose(
            SpDMD(svd_rank=30, gamma=gm, rho=1.0e4).fit(data).amplitudes,
            z,
            rtol=2,
            atol=1.e-6
        )

def test_rho():
    assert SpDMD().rho == 1
    assert SpDMD(rho=10).rho == 10

def test_maxiter():
    assert SpDMD()._max_iterations == 10000
    assert SpDMD(max_iterations=2)._max_iterations == 2

def test_gamma():
    assert SpDMD().gamma == 10
    assert SpDMD(gamma=2).gamma == 2

def test_exact():
    assert SpDMD().exact == True

def test_abstol():
    assert SpDMD()._abs_tol == 1.0e-6
    assert SpDMD(abs_tolerance=1.0e10)._abs_tol == 1.0e10

def test_reltol():
    assert SpDMD()._rel_tol == 1.0e-4
    assert SpDMD(rel_tolerance=1.0e10)._rel_tol == 1.0e10

def test_verbose():
    assert SpDMD()._verbose == True
    assert SpDMD(verbose=False)._verbose == False

def test_enforce_zero():
    assert SpDMD()._enforce_zero == True
    assert SpDMD(enforce_zero=False)._enforce_zero == False

def test_release_memory():
    assert SpDMD()._release_memory == True
    assert SpDMD(release_memory=False)._release_memory == False

def test_zero_tolerance():
    assert SpDMD()._zero_absolute_tolerance == 1.0e-12
    assert SpDMD(
        zero_absolute_tolerance=1.0e-6)._zero_absolute_tolerance == 1.0e-6

def test_zero_tolerance_no_zeros():
    zero_amps = (
        SpDMD(zero_absolute_tolerance=1.0e30, release_memory=False)
        .fit(data)
        ._find_zero_amplitudes()
    )
    assert all(zero_amps)

def test_zero_tolerance_all_zeros():
    zero_amps = (
        SpDMD(zero_absolute_tolerance=0, release_memory=False)
        .fit(data)
        ._find_zero_amplitudes()
    )
    assert all(np.logical_not(zero_amps))

def test_release_memory_releases():
    o = SpDMD(release_memory=True).fit(data)
    assert o._P is None
    assert o._Plow is None
    assert o._q is None

    o = SpDMD(release_memory=False).fit(data)
    assert o._P is not None
    assert o._Plow is not None
    assert o._q is not None

def test_update_lagrangian():
    alpha = np.random.rand(10)
    beta = np.random.rand(10)
    lmbd = np.random.rand(10)

    np.testing.assert_allclose(
        SpDMD(rho=0)._update_lagrangian(alpha, beta, lmbd), lmbd
    )
    np.testing.assert_allclose(
        SpDMD()._update_lagrangian(alpha, alpha, lmbd), lmbd
    )

def test_update_alpha():
    o = SpDMD(release_memory=False).fit(data)

    beta = np.random.rand(len(o.amplitudes))
    lmbd = np.random.rand(len(o.amplitudes))

    uk = beta - lmbd / o.rho

    rhs = np.linalg.solve(o._Plow, o._q + uk * o.rho / 2)
    lhs = o._Plow.conj().T

    np.testing.assert_allclose(lhs.dot(o._update_alpha(beta, lmbd)), rhs)

def test_get_bitmask_default():
    dmd = SpDMD(release_memory=True, svd_rank=-1)
    dmd.fit(X=data)
    assert np.all(dmd.modes_activation_bitmask == True)

def test_set_bitmask():
    dmd = SpDMD(release_memory=True, svd_rank=-1)
    dmd.fit(X=data)

    new_bitmask = np.full(len(dmd.amplitudes), True, dtype=bool)
    new_bitmask[[0]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.modes_activation_bitmask[0] == False
    assert np.all(dmd.modes_activation_bitmask[1:] == True)

def test_not_fitted_get_bitmask_raises():
    dmd = SpDMD(release_memory=True, svd_rank=-1)
    with raises(RuntimeError):
        print(dmd.modes_activation_bitmask)

def test_not_fitted_set_bitmask_raises():
    dmd = SpDMD(release_memory=True, svd_rank=-1)
    with raises(RuntimeError):
        dmd.modes_activation_bitmask = np.full(3, True, dtype=bool)

def test_raise_wrong_dtype_bitmask():
    dmd = SpDMD(release_memory=True, svd_rank=-1)
    dmd.fit(X=data)
    with raises(RuntimeError):
        dmd.modes_activation_bitmask = np.full(3, 0.1)

def test_fitted():
    dmd = SpDMD(release_memory=True, svd_rank=-1)
    assert not dmd.fitted
    dmd.fit(X=data)
    assert dmd.fitted

def test_bitmask_amplitudes():
    dmd = SpDMD(release_memory=True, svd_rank=-1)
    dmd.fit(X=data)

    old_n_amplitudes = dmd.amplitudes.shape[0]
    retained_amplitudes = np.delete(dmd.amplitudes, [0,-1])

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0,-1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.amplitudes.shape[0] == old_n_amplitudes - 2
    np.testing.assert_almost_equal(dmd.amplitudes, retained_amplitudes)

def test_bitmask_eigs():
    dmd = SpDMD(release_memory=True, svd_rank=-1)
    dmd.fit(X=data)

    old_n_eigs = dmd.eigs.shape[0]
    retained_eigs = np.delete(dmd.eigs, [0,-1])

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0,-1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.eigs.shape[0] == old_n_eigs - 2
    np.testing.assert_almost_equal(dmd.eigs, retained_eigs)

def test_bitmask_modes():
    dmd = SpDMD(release_memory=True, svd_rank=-1)
    dmd.fit(X=data)

    old_n_modes = dmd.modes.shape[1]
    retained_modes = np.delete(dmd.modes, [0,-1], axis=1)

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0,-1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.modes.shape[1] == old_n_modes - 2
    np.testing.assert_almost_equal(dmd.modes, retained_modes)

def test_reconstructed_data():
    dmd = SpDMD(release_memory=True, svd_rank=-1)
    dmd.fit(X=data)

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0,-1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    dmd.reconstructed_data
    assert True

def test_getitem_modes():
    dmd = SpDMD(release_memory=True, svd_rank=-1)
    dmd.fit(X=data)
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

def test_getitem_raises():
    dmd = SpDMD(release_memory=True, svd_rank=-1)
    dmd.fit(X=data)

    with raises(ValueError):
        dmd[[0,1,1,0,1]]
    with raises(ValueError):
        dmd[[True, True, False, True]]
    with raises(ValueError):
        dmd[1.0]

def test_correct_amplitudes():
    dmd = SpDMD(release_memory=True, svd_rank=-1)
    dmd.fit(X=data)
    np.testing.assert_array_almost_equal(dmd.amplitudes, dmd._b)
