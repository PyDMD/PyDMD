import numpy as np
import pytest
import scipy
from pytest import raises

from pydmd import DMDc

from .linalg.utils import assert_allclose, setup_backends

np.random.seed(10)


def create_system_with_B():
    snapshots = np.array([[4, 2, 1, 0.5, 0.25], [7, 0.7, 0.07, 0.007, 0.0007]])
    I = np.array([-4, -2, -1, -0.5])
    B = np.array([[1, 0]]).T
    return {"X": snapshots, "I": I, "B": B}


def create_system_without_B():
    n = 5  # dimension snapshots
    m = 15  # number snapshots
    A = scipy.linalg.helmert(n, True)
    B = np.random.rand(n, n) - 0.5
    x0 = np.array([0.25] * n)
    I = np.random.rand(n, m - 1) - 0.5
    snapshots = [x0]
    for i in range(m - 1):
        snapshots.append(A.dot(snapshots[i]) + B.dot(I[:, i]))
    snapshots = np.array(snapshots).T
    return {"X": snapshots, "I": I, "B": B}


data_backends_with_B = setup_backends(create_system_with_B())
data_backends_without_B = setup_backends(create_system_without_B())


@pytest.mark.parametrize("system", data_backends_with_B)
def test_eigs_b_known(system):
    dmd = DMDc(svd_rank=-1)
    dmd.fit(**system)
    assert_allclose(dmd.eigs, [0.1, 1.5])


@pytest.mark.parametrize("system", data_backends_without_B)
def test_eigs_b_unknown(system):
    dmd = DMDc(svd_rank=3, opt=False, svd_rank_omega=4)
    dmd.fit(**system)
    assert dmd.eigs.shape[0] == 3


@pytest.mark.parametrize("system", data_backends_without_B)
def test_modes_b_unknown(system):
    dmd = DMDc(svd_rank=3, opt=False, svd_rank_omega=4)
    dmd.fit(**system)
    assert dmd.modes.shape[1] == 3


@pytest.mark.parametrize("system", data_backends_with_B)
def test_reconstruct_b_known(system):
    dmd = DMDc(svd_rank=-1)
    dmd.fit(**system)
    assert_allclose(dmd.reconstructed_data(), system["X"])


@pytest.mark.parametrize("system", data_backends_with_B)
def test_B_b_known(system):
    dmd = DMDc(svd_rank=-1)
    dmd.fit(**system)
    assert_allclose(dmd.B, system["B"])


@pytest.mark.parametrize("system", data_backends_without_B)
def test_reconstruct_b_unknown(system):
    dmd = DMDc(svd_rank=-1, opt=True)
    dmd.fit(**system)
    assert_allclose(dmd.reconstructed_data(), system["X"], atol=1.0e-6)


@pytest.mark.parametrize("system", data_backends_with_B)
def test_get_bitmask_default(system):
    dmd = DMDc(svd_rank=-1, opt=True)
    dmd.fit(**system)
    assert dmd.modes_activation_bitmask.all()


@pytest.mark.parametrize("system", data_backends_with_B)
def test_set_bitmask(system):
    dmd = DMDc(svd_rank=-1, opt=True)
    dmd.fit(**system)

    new_bitmask = np.full(len(dmd.amplitudes), True, dtype=bool)
    new_bitmask[[0]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.modes_activation_bitmask[0] == False
    assert (dmd.modes_activation_bitmask[1:] == True).all()


def test_not_fitted_get_bitmask_raises():
    dmd = DMDc(svd_rank=-1, opt=True)
    with raises(RuntimeError):
        print(dmd.modes_activation_bitmask)


def test_not_fitted_set_bitmask_raises():
    dmd = DMDc(svd_rank=-1, opt=True)
    with raises(RuntimeError):
        dmd.modes_activation_bitmask = np.full(3, True, dtype=bool)


@pytest.mark.parametrize("system", data_backends_with_B)
def test_raise_wrong_dtype_bitmask(system):
    dmd = DMDc(svd_rank=-1, opt=True)
    dmd.fit(**system)
    with raises(RuntimeError):
        dmd.modes_activation_bitmask = np.full(3, 0.1)


@pytest.mark.parametrize("system", data_backends_with_B)
def test_fitted(system):
    dmd = DMDc(svd_rank=-1, opt=True)
    assert not dmd.fitted
    dmd.fit(**system)
    assert dmd.fitted


@pytest.mark.parametrize("system", data_backends_with_B)
def test_bitmask_amplitudes(system):
    dmd = DMDc(svd_rank=-1, opt=True)
    dmd.fit(**system)

    ampls = np.array(dmd.amplitudes)
    old_n_amplitudes = ampls.shape[0]
    retained_amplitudes = np.delete(ampls, [0, -1])

    new_bitmask = np.full(ampls.shape[0], True, dtype=bool)
    new_bitmask[[0, -1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.amplitudes.shape[0] == old_n_amplitudes - 2
    assert_allclose(dmd.amplitudes, retained_amplitudes)


@pytest.mark.parametrize("system", data_backends_with_B)
def test_bitmask_eigs(system):
    dmd = DMDc(svd_rank=-1, opt=True)
    dmd.fit(**system)

    eigs = np.array(dmd.eigs)
    old_n_eigs = eigs.shape[0]
    retained_eigs = np.delete(eigs, [0, -1])

    new_bitmask = np.full(eigs.shape[0], True, dtype=bool)
    new_bitmask[[0, -1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.eigs.shape[0] == old_n_eigs - 2
    assert_allclose(dmd.eigs, retained_eigs)


@pytest.mark.parametrize("system", data_backends_with_B)
def test_bitmask_modes(system):
    dmd = DMDc(svd_rank=-1, opt=True)
    dmd.fit(**system)

    modes = np.array(dmd.modes)
    old_n_modes = modes.shape[1]
    retained_modes = np.delete(modes, [0, -1], axis=1)

    new_bitmask = np.full(modes.shape[0], True, dtype=bool)
    new_bitmask[[0, -1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.modes.shape[1] == old_n_modes - 2
    assert_allclose(dmd.modes, retained_modes)


@pytest.mark.parametrize("system", data_backends_with_B)
def test_reconstructed_data(system):
    dmd = DMDc(svd_rank=-1, opt=True)
    dmd.fit(**system)

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0, -1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.reconstructed_data is not None


@pytest.mark.parametrize("system", data_backends_with_B)
def test_getitem_modes(system):
    dmd = DMDc(svd_rank=-1)
    dmd.fit(**system)
    old_n_modes = dmd.modes.shape[1]

    assert dmd[[0, -1]].modes.shape[1] == 2
    assert_allclose(dmd[[0, -1]].modes, dmd.modes[:, [0, -1]])

    assert dmd.modes.shape[1] == old_n_modes

    assert dmd[1::2].modes.shape[1] == old_n_modes // 2
    assert_allclose(dmd[1::2].modes, dmd.modes[:, 1::2])

    assert dmd.modes.shape[1] == old_n_modes

    assert dmd[1].modes.shape[1] == 1
    assert_allclose(np.squeeze(dmd[1].modes), dmd.modes[:, 1])

    assert dmd.modes.shape[1] == old_n_modes


@pytest.mark.parametrize("system", data_backends_with_B)
def test_getitem_raises(system):
    dmd = DMDc(svd_rank=-1)
    dmd.fit(**system)

    with raises(ValueError):
        dmd[[0, 1, 1, 0, 1]]
    with raises(ValueError):
        dmd[[True, True, False, True]]
    with raises(ValueError):
        dmd[1.0]


@pytest.mark.parametrize("system", data_backends_with_B)
def test_correct_amplitudes(system):
    dmd = DMDc(svd_rank=-1)
    dmd.fit(**system)
    assert_allclose(dmd.amplitudes, dmd._b)
