import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pytest import raises

from pydmd.cdmd import CDMD
from pydmd.linalg import build_linalg_module

from .utils import assert_allclose, data_backends


@pytest.mark.parametrize("X", data_backends)
def test_shape(X):
    dmd = CDMD(svd_rank=-1)
    dmd.fit(X=X)
    assert dmd.modes.shape[1] == X.shape[1] - 1


@pytest.mark.parametrize("X", data_backends)
def test_truncation_shape(X):
    dmd = CDMD(svd_rank=3)
    dmd.fit(X=X)
    assert dmd.modes.shape[1] == 3


@pytest.mark.parametrize("X", data_backends)
def test_Atilde_shape(X):
    dmd = CDMD(svd_rank=3)
    dmd.fit(X=X)
    assert dmd.atilde.shape == (dmd.svd_rank, dmd.svd_rank)


@pytest.mark.parametrize("X", data_backends)
def test_eigs_1(X):
    dmd = CDMD(svd_rank=-1)
    dmd.fit(X=X)
    assert len(dmd.eigs) == 14


@pytest.mark.parametrize("X", data_backends)
def test_eigs_2(X):
    dmd = CDMD(svd_rank=5)
    dmd.fit(X=X)
    assert len(dmd.eigs) == 5


@pytest.mark.parametrize("X", data_backends)
def test_eigs_3(X):
    dmd = CDMD(svd_rank=2)
    dmd.fit(X=X)
    expected_eigs = [-0.47386866 + 0.88059553j, -0.80901699 + 0.58778525j]
    assert_allclose(dmd.eigs, expected_eigs, atol=1.e-6)


@pytest.mark.parametrize("X", data_backends)
def test_dynamics_1(X):
    dmd = CDMD(svd_rank=5)
    dmd.fit(X=X)
    assert dmd.dynamics.shape == (5, X.shape[1])


@pytest.mark.parametrize("X", data_backends)
def test_reconstructed_data(X):
    dmd = CDMD()
    dmd.fit(X=X)
    dmd_data = dmd.reconstructed_data
    assert_allclose(dmd_data, X)


@pytest.mark.parametrize("X", data_backends)
def test_original_time(X):
    dmd = CDMD(svd_rank=2)
    dmd.fit(X=X)
    expected_dict = {"dt": 1, "t0": 0, "tend": 14}
    assert dmd.original_time == expected_dict


@pytest.mark.parametrize("X", data_backends)
def test_original_timesteps(X):
    dmd = CDMD()
    dmd.fit(X=X)
    assert_allclose(dmd.original_timesteps, np.arange(X.shape[1]))


@pytest.mark.parametrize("X", data_backends)
def test_dmd_time_1(X):
    dmd = CDMD(svd_rank=2)
    dmd.fit(X=X)
    expected_dict = {"dt": 1, "t0": 0, "tend": 14}
    assert dmd.dmd_time == expected_dict


@pytest.mark.parametrize("X", data_backends)
def test_dmd_time_2(X):
    dmd = CDMD()
    dmd.fit(X=X)
    dmd.dmd_time["t0"] = 10
    dmd.dmd_time["tend"] = 14
    expected_data = X[:, -5:]
    assert_allclose(dmd.reconstructed_data, expected_data)


@pytest.mark.parametrize("X", data_backends)
def test_dmd_time_3(X):
    dmd = CDMD()
    dmd.fit(X=X)
    dmd.dmd_time["t0"] = 8
    dmd.dmd_time["tend"] = 11
    expected_data = X[:, 8:12]
    assert_allclose(dmd.reconstructed_data, expected_data)


@pytest.mark.parametrize("X", data_backends)
def test_plot_eigs_1(X):
    dmd = CDMD()
    dmd.fit(X=X)
    dmd.plot_eigs(show_axes=True, show_unit_circle=True)
    plt.close()


@pytest.mark.parametrize("X", data_backends)
def test_plot_eigs_2(X):
    dmd = CDMD()
    dmd.fit(X=X)
    dmd.plot_eigs(show_axes=False, show_unit_circle=False)
    plt.close()


@pytest.mark.parametrize("X", data_backends)
def test_plot_modes_1(X):
    dmd = CDMD()
    dmd.fit(X=X)
    with raises(ValueError):
        dmd.plot_modes_2D()


@pytest.mark.parametrize("X", data_backends)
def test_plot_modes_2(X):
    dmd = CDMD(svd_rank=-1)
    dmd.fit(X=X)
    dmd.plot_modes_2D((1, 2, 5), x=np.arange(20), y=np.arange(20))
    plt.close()


@pytest.mark.parametrize("X", data_backends)
def test_plot_modes_3(X):
    dmd = CDMD()
    linalg_module = build_linalg_module(X)
    snapshots = [linalg_module.new_array(snap.reshape(20, 20)) for snap in X.T]
    dmd.fit(X=snapshots)
    dmd.plot_modes_2D()
    plt.close()


@pytest.mark.parametrize("X", data_backends)
def test_plot_modes_4(X):
    dmd = CDMD()
    linalg_module = build_linalg_module(X)
    snapshots = [linalg_module.new_array(snap.reshape(20, 20)) for snap in X.T]
    dmd.fit(X=snapshots)
    dmd.plot_modes_2D(index_mode=1)
    plt.close()


@pytest.mark.parametrize("X", data_backends)
def test_plot_modes_5(X):
    dmd = CDMD()
    linalg_module = build_linalg_module(X)
    snapshots = [linalg_module.new_array(snap.reshape(20, 20)) for snap in X.T]
    dmd.fit(X=snapshots)
    dmd.plot_modes_2D(index_mode=1, filename="tmp.png")
    os.remove("tmp.1.png")


@pytest.mark.parametrize("X", data_backends)
def test_plot_snapshots_1(X):
    dmd = CDMD()
    dmd.fit(X=X)
    with raises(ValueError):
        dmd.plot_snapshots_2D()


@pytest.mark.parametrize("X", data_backends)
def test_plot_snapshots_2(X):
    dmd = CDMD(svd_rank=-1)
    dmd.fit(X=X)
    dmd.plot_snapshots_2D((1, 2, 5), x=np.arange(20), y=np.arange(20))
    plt.close()


@pytest.mark.parametrize("X", data_backends)
def test_plot_snapshots_3(X):
    dmd = CDMD()
    linalg_module = build_linalg_module(X)
    snapshots = [linalg_module.new_array(snap.reshape(20, 20)) for snap in X.T]
    dmd.fit(X=snapshots)
    dmd.plot_snapshots_2D()
    plt.close()


@pytest.mark.parametrize("X", data_backends)
def test_plot_snapshots_4(X):
    dmd = CDMD()
    linalg_module = build_linalg_module(X)
    snapshots = [linalg_module.new_array(snap.reshape(20, 20)) for snap in X.T]
    dmd.fit(X=snapshots)
    dmd.plot_snapshots_2D(index_snap=2)
    plt.close()


@pytest.mark.parametrize("X", data_backends)
def test_plot_snapshots_5(X):
    dmd = CDMD()
    linalg_module = build_linalg_module(X)
    snapshots = [linalg_module.new_array(snap.reshape(20, 20)) for snap in X.T]
    dmd.fit(X=snapshots)
    dmd.plot_snapshots_2D(index_snap=2, filename="tmp.png")
    os.remove("tmp.2.png")


@pytest.mark.parametrize("X", data_backends)
def test_tdmd_plot(X):
    dmd = CDMD(tlsq_rank=3)
    dmd.fit(X=X)
    dmd.plot_eigs(show_axes=False, show_unit_circle=False)
    plt.close()


@pytest.mark.parametrize("X", data_backends)
def test_cdmd_matrix_uniform(X):
    dmd = CDMD(compression_matrix="uniform")
    dmd.fit(X=X)
    error_norm = np.linalg.norm(dmd.reconstructed_data - X, 1)
    assert error_norm < 1e-10


@pytest.mark.parametrize("X", data_backends)
def test_cdmd_matrix_sample(X):
    dmd = CDMD(compression_matrix="sample")
    dmd.fit(X=X)
    error_norm = np.linalg.norm(dmd.reconstructed_data - X, 1)
    assert error_norm < 1e-10


@pytest.mark.parametrize("X", data_backends)
def test_cdmd_matrix_normal(X):
    dmd = CDMD(compression_matrix="normal")
    dmd.fit(X=X)
    error_norm = np.linalg.norm(dmd.reconstructed_data - X, 1)
    assert error_norm < 1e-10


@pytest.mark.parametrize("X", data_backends)
def test_cdmd_matrix_sparse(X):
    dmd = CDMD(compression_matrix="sparse")
    dmd.fit(X=X)
    error_norm = np.linalg.norm(dmd.reconstructed_data - X, 1)
    assert error_norm < 1e-10


@pytest.mark.parametrize("X", data_backends)
def test_cdmd_matrix_custom(X):
    matrix = (
        np.random.permutation(
            (X.shape[1] - 3) * X.shape[0]
        )
        .reshape(X.shape[1] - 3, X.shape[0])
        .astype(float)
    )
    matrix /= float(np.sum(matrix))
    dmd = CDMD(compression_matrix=matrix)
    dmd.fit(X=X)
    error_norm = np.linalg.norm(dmd.reconstructed_data - X, 1)
    assert error_norm < 1e-10


def test_sorted_eigs_default():
    dmd = CDMD(compression_matrix="sparse")
    assert dmd.operator._sorted_eigs == False


def test_sorted_eigs_param():
    dmd = CDMD(compression_matrix="sparse", sorted_eigs="real")
    assert dmd.operator._sorted_eigs == "real"


@pytest.mark.parametrize("X", data_backends)
def test_get_bitmask_default(X):
    dmd = CDMD(
        compression_matrix="normal",
    )
    dmd.fit(X=X)
    assert dmd.modes_activation_bitmask.all()


@pytest.mark.parametrize("X", data_backends)
def test_set_bitmask(X):
    dmd = CDMD(compression_matrix="normal")
    dmd.fit(X=X)

    new_bitmask = np.full(len(dmd.amplitudes), True, dtype=bool)
    new_bitmask[[0]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.modes_activation_bitmask[0] == False
    assert dmd.modes_activation_bitmask[1:].all()


def test_not_fitted_get_bitmask_raises():
    dmd = CDMD(compression_matrix="normal")
    with raises(RuntimeError):
        print(dmd.modes_activation_bitmask)


def test_not_fitted_set_bitmask_raises():
    dmd = CDMD(compression_matrix="normal")
    with raises(RuntimeError):
        dmd.modes_activation_bitmask = np.full(3, True, dtype=bool)


@pytest.mark.parametrize("X", data_backends)
def test_raise_wrong_dtype_bitmask(X):
    dmd = CDMD(compression_matrix="normal")
    dmd.fit(X=X)
    with raises(RuntimeError):
        dmd.modes_activation_bitmask = np.full(3, 0.1)


@pytest.mark.parametrize("X", data_backends)
def test_fitted(X):
    dmd = CDMD(compression_matrix="normal")
    assert not dmd.fitted
    dmd.fit(X=X)
    assert dmd.fitted


@pytest.mark.parametrize("X", data_backends)
def test_bitmask_amplitudes(X):
    dmd = CDMD(compression_matrix="normal", svd_rank=-1)
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
    dmd = CDMD(compression_matrix="normal", svd_rank=-1)
    dmd.fit(X=X)

    old_n_eigs = dmd.eigs.shape[0]
    print(old_n_eigs)
    retained_eigs = np.delete(dmd.eigs, [0, -1])

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0, -1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.eigs.shape[0] == old_n_eigs - 2
    assert_allclose(dmd.eigs, retained_eigs)


@pytest.mark.parametrize("X", data_backends)
def test_bitmask_modes(X):
    dmd = CDMD(compression_matrix="normal", svd_rank=-1)
    dmd.fit(X=X)

    old_n_modes = dmd.modes.shape[1]
    retained_modes = np.delete(dmd.modes, [0, -1], axis=1)

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0, -1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.modes.shape[1] == old_n_modes - 2
    assert_allclose(
        dmd.modes, retained_modes
    )


@pytest.mark.parametrize("X", data_backends)
def test_getitem_modes(X):
    dmd = CDMD(compression_matrix="normal", svd_rank=10)
    dmd.fit(X=X)
    old_n_modes = dmd.modes.shape[1]

    assert dmd[[0, -1]].modes.shape[1] == 2
    assert_allclose(
        dmd[[0, -1]].modes, dmd.modes[:, [0, -1]]
    )

    assert dmd.modes.shape[1] == old_n_modes

    assert dmd[1::2].modes.shape[1] == old_n_modes // 2
    assert_allclose(dmd[1::2].modes, dmd.modes[:, 1::2])

    assert dmd.modes.shape[1] == old_n_modes

    assert dmd[[1, 3]].modes.shape[1] == 2
    assert_allclose(
        dmd[[1, 3]].modes, dmd.modes[:, [1, 3]]
    )

    assert dmd.modes.shape[1] == old_n_modes

    assert dmd[2].modes.shape[1] == 1
    assert_allclose(
        np.squeeze(dmd[2].modes), dmd.modes[:, 2]
    )

    assert dmd.modes.shape[1] == old_n_modes


@pytest.mark.parametrize("X", data_backends)
def test_getitem_raises(X):
    dmd = CDMD(compression_matrix="normal")
    dmd.fit(X=X)

    with raises(ValueError):
        dmd[[0, 1, 1, 0, 1]]
    with raises(ValueError):
        dmd[[True, True, False, True]]
    with raises(ValueError):
        dmd[1.0]


@pytest.mark.parametrize("X", data_backends)
def test_reconstructed_data(X):
    dmd = CDMD(compression_matrix="normal")
    dmd.fit(X=X)

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0, -1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    dmd.reconstructed_data
    assert True


# this is a test for the correctness of the amplitudes saved in the Proxy
# between DMDBase and the modes activation bitmask. if this test fails
# you probably need to call allocate_proxy once again after you compute
# the final value of the amplitudes
@pytest.mark.parametrize("X", data_backends)
def test_correct_amplitudes(X):
    dmd = CDMD(compression_matrix="normal")
    dmd.fit(X=X)
    assert_allclose(dmd.amplitudes, dmd._b)
