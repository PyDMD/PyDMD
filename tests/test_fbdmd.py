import numpy as np
import pytest
from pytest import raises

from pydmd.fbdmd import FbDMD

from .utils import assert_allclose, setup_backends, noisy_data

data_backends = setup_backends(data=noisy_data())
input_sample_backends = setup_backends(
    data=np.load("tests/test_datasets/input_sample.npy")
)


@pytest.mark.parametrize("X", data_backends)
def test_modes_shape(X):
    dmd = FbDMD(svd_rank=-1)
    dmd.fit(X=X)
    assert dmd.modes.shape[1] == 2


@pytest.mark.parametrize("X", data_backends)
def test_truncation_shape(X):
    dmd = FbDMD(svd_rank=1)
    dmd.fit(X=X)
    assert dmd.modes.shape[1] == 1


@pytest.mark.parametrize("X", data_backends)
def test_eigs_1(X):
    dmd = FbDMD(svd_rank=1)
    dmd.fit(X=X)
    assert len(dmd.eigs) == 1


@pytest.mark.parametrize("X", data_backends)
def test_reconstructed_data(X):
    dmd = FbDMD(exact=True, svd_rank=-1)
    dmd.fit(X=X)
    dmd_data = dmd.reconstructed_data
    dmd_data_correct = np.load("tests/test_datasets/fbdmd_data.npy")
    assert_allclose(dmd_data, dmd_data_correct)


def test_sorted_eigs_default():
    dmd = FbDMD(svd_rank=-1)
    assert dmd.operator._sorted_eigs == False


@pytest.mark.parametrize("X", data_backends)
def test_sorted_eigs_param(X):
    dmd = FbDMD(svd_rank=-1, sorted_eigs="real")
    assert dmd.operator._sorted_eigs == "real"


@pytest.mark.parametrize("X", data_backends)
def test_get_bitmask_default(X):
    dmd = FbDMD(svd_rank=-1)
    dmd.fit(X=X)
    assert dmd.modes_activation_bitmask.all()


@pytest.mark.parametrize("X", data_backends)
def test_set_bitmask(X):
    dmd = FbDMD(svd_rank=-1)
    dmd.fit(X=X)

    new_bitmask = np.full(len(dmd.amplitudes), True, dtype=bool)
    new_bitmask[[0]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.modes_activation_bitmask[0] == False
    assert dmd.modes_activation_bitmask[1:].all()


@pytest.mark.parametrize("X", data_backends)
def test_not_fitted_get_bitmask_raises(X):
    dmd = FbDMD(svd_rank=-1)
    with raises(RuntimeError):
        print(dmd.modes_activation_bitmask)


@pytest.mark.parametrize("X", data_backends)
def test_not_fitted_set_bitmask_raises(X):
    dmd = FbDMD(svd_rank=-1)
    with raises(RuntimeError):
        dmd.modes_activation_bitmask = np.full(3, True, dtype=bool)


@pytest.mark.parametrize("X", data_backends)
def test_raise_wrong_dtype_bitmask(X):
    dmd = FbDMD(svd_rank=-1)
    dmd.fit(X=X)
    with raises(RuntimeError):
        dmd.modes_activation_bitmask = np.full(3, 0.1)


@pytest.mark.parametrize("X", data_backends)
def test_fitted(X):
    dmd = FbDMD(svd_rank=-1)
    assert not dmd.fitted
    dmd.fit(X=X)
    assert dmd.fitted


@pytest.mark.parametrize("X", data_backends)
def test_bitmask_amplitudes(X):
    dmd = FbDMD(svd_rank=-1)
    dmd.fit(X=X)

    old_n_amplitudes = dmd.amplitudes.shape[0]
    retained_amplitudes = np.delete(dmd.amplitudes, [0, -1])

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0, -1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.amplitudes.shape[0] == old_n_amplitudes - 2
    assert_allclose(dmd.amplitudes, retained_amplitudes)


@pytest.mark.parametrize("X", data_backends)
def test_bitmask_eigs(X):
    dmd = FbDMD(svd_rank=-1)
    dmd.fit(X=X)

    old_n_eigs = dmd.eigs.shape[0]
    retained_eigs = np.delete(dmd.eigs, [0, -1])

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0, -1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.eigs.shape[0] == old_n_eigs - 2
    assert_allclose(dmd.eigs, retained_eigs)


@pytest.mark.parametrize("X", data_backends)
def test_bitmask_modes(X):
    dmd = FbDMD(svd_rank=-1)
    dmd.fit(X=X)

    old_n_modes = dmd.modes.shape[1]
    retained_modes = np.delete(dmd.modes, [0, -1], axis=1)

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0, -1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.modes.shape[1] == old_n_modes - 2
    assert_allclose(dmd.modes, retained_modes)


@pytest.mark.parametrize("X", data_backends)
def test_reconstructed_data_with_bitmask(X):
    dmd = FbDMD(svd_rank=-1)
    dmd.fit(X=X)

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0, -1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.reconstructed_data is not None


@pytest.mark.parametrize("X", input_sample_backends)
def test_getitem_modes(X):
    dmd = FbDMD(svd_rank=-1)
    dmd.fit(X=X)
    old_n_modes = dmd.modes.shape[1]

    assert dmd[[0, -1]].modes.shape[1] == 2
    assert_allclose(dmd[[0, -1]].modes, dmd.modes[:, [0, -1]])

    assert dmd.modes.shape[1] == old_n_modes

    assert dmd[1::2].modes.shape[1] == old_n_modes // 2
    assert_allclose(dmd[1::2].modes, dmd.modes[:, 1::2])

    assert dmd.modes.shape[1] == old_n_modes

    assert dmd[[1, 3]].modes.shape[1] == 2
    assert_allclose(dmd[[1, 3]].modes, dmd.modes[:, [1, 3]])

    assert dmd.modes.shape[1] == old_n_modes

    assert dmd[2].modes.shape[1] == 1
    assert_allclose(np.squeeze(dmd[2].modes), dmd.modes[:, 2])

    assert dmd.modes.shape[1] == old_n_modes


@pytest.mark.parametrize("X", input_sample_backends)
def test_getitem_raises(X):
    dmd = FbDMD(svd_rank=-1)
    dmd.fit(X=X)

    with raises(ValueError):
        dmd[[0, 1, 1, 0, 1]]
    with raises(ValueError):
        dmd[[True, True, False, True]]
    with raises(ValueError):
        dmd[1.0]


@pytest.mark.parametrize("X", input_sample_backends)
def test_correct_amplitudes(X):
    dmd = FbDMD(svd_rank=-1)
    dmd.fit(X=X)
    assert_allclose(dmd.amplitudes, dmd._b)
