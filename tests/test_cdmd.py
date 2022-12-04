from pytest import raises
import pytest

from pydmd.cdmd import CDMD

import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.sparse as sp

# 15 snapshot with 400 data. The matrix is 400x15 and it contains
# the following data: f1 + f2 where
# f1 = lambda x,t: sech(x+3)*(1.*np.exp(1j*2.3*t))
# f2 = lambda x,t: (sech(x)*np.tanh(x))*(2.*np.exp(1j*2.8*t))
sample_data = np.load("tests/test_datasets/input_sample.npy")
data_backends = (
    # NumPy
    sample_data,
    # PyTorch
    torch.from_numpy(sample_data),
    # TODO GPU
    # SciPy.sparse
    # sp.coo_array(sample_data),
    # sp.csr_array(sample_data),
)


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
    expected_eigs = np.array(
        [-0.47386866 + 0.88059553j, -0.80901699 + 0.58778525j]
    )
    np.testing.assert_almost_equal(dmd.eigs, expected_eigs, decimal=6)


@pytest.mark.parametrize("X", data_backends)
def test_dynamics_1(X):
    dmd = CDMD(svd_rank=5)
    dmd.fit(X=X)
    assert dmd.dynamics.shape == (5, sample_data.shape[1])


@pytest.mark.parametrize("X", data_backends)
def test_reconstructed_data(X):
    dmd = CDMD()
    dmd.fit(X=X)
    dmd_data = dmd.reconstructed_data
    np.testing.assert_allclose(dmd_data, sample_data)


@pytest.mark.parametrize("X", data_backends)
def test_original_time(X):
    dmd = CDMD(svd_rank=2)
    dmd.fit(X=X)
    expected_dict = {"dt": 1, "t0": 0, "tend": 14}
    np.testing.assert_equal(dmd.original_time, expected_dict)


@pytest.mark.parametrize("X", data_backends)
def test_original_timesteps(X):
    dmd = CDMD()
    dmd.fit(X=X)
    np.testing.assert_allclose(
        dmd.original_timesteps, np.arange(sample_data.shape[1])
    )


@pytest.mark.parametrize("X", data_backends)
def test_dmd_time_1(X):
    dmd = CDMD(svd_rank=2)
    dmd.fit(X=X)
    expected_dict = {"dt": 1, "t0": 0, "tend": 14}
    np.testing.assert_equal(dmd.dmd_time, expected_dict)


@pytest.mark.parametrize("X", data_backends)
def test_dmd_time_2(X):
    dmd = CDMD()
    dmd.fit(X=X)
    dmd.dmd_time["t0"] = 10
    dmd.dmd_time["tend"] = 14
    expected_data = sample_data[:, -5:]
    np.testing.assert_allclose(dmd.reconstructed_data, expected_data)


@pytest.mark.parametrize("X", data_backends)
def test_dmd_time_3(X):
    dmd = CDMD()
    dmd.fit(X=X)
    dmd.dmd_time["t0"] = 8
    dmd.dmd_time["tend"] = 11
    expected_data = sample_data[:, 8:12]
    np.testing.assert_allclose(dmd.reconstructed_data, expected_data)


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
    snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
    dmd.fit(X=snapshots)
    dmd.plot_modes_2D()
    plt.close()


@pytest.mark.parametrize("X", data_backends)
def test_plot_modes_4(X):
    dmd = CDMD()
    snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
    dmd.fit(X=snapshots)
    dmd.plot_modes_2D(index_mode=1)
    plt.close()


@pytest.mark.parametrize("X", data_backends)
def test_plot_modes_5(X):
    dmd = CDMD()
    snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
    dmd.fit(X=snapshots)
    dmd.plot_modes_2D(index_mode=1, filename="tmp.png")

    import os

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
    snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
    dmd.fit(X=snapshots)
    dmd.plot_snapshots_2D()
    plt.close()


@pytest.mark.parametrize("X", data_backends)
def test_plot_snapshots_4(X):
    dmd = CDMD()
    snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
    dmd.fit(X=snapshots)
    dmd.plot_snapshots_2D(index_snap=2)
    plt.close()


@pytest.mark.parametrize("X", data_backends)
def test_plot_snapshots_5(X):
    dmd = CDMD()
    snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
    dmd.fit(X=snapshots)
    dmd.plot_snapshots_2D(index_snap=2, filename="tmp.png")

    import os

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
    error_norm = np.linalg.norm(dmd.reconstructed_data - sample_data, 1)
    assert error_norm < 1e-10


@pytest.mark.parametrize("X", data_backends)
def test_cdmd_matrix_sample(X):
    dmd = CDMD(compression_matrix="sample")
    dmd.fit(X=X)
    error_norm = np.linalg.norm(dmd.reconstructed_data - sample_data, 1)
    assert error_norm < 1e-10


@pytest.mark.parametrize("X", data_backends)
def test_cdmd_matrix_normal(X):
    dmd = CDMD(compression_matrix="normal")
    dmd.fit(X=X)
    error_norm = np.linalg.norm(dmd.reconstructed_data - sample_data, 1)
    assert error_norm < 1e-10


@pytest.mark.parametrize("X", data_backends)
def test_cdmd_matrix_sparse(X):
    dmd = CDMD(compression_matrix="sparse")
    dmd.fit(X=X)
    error_norm = np.linalg.norm(dmd.reconstructed_data - sample_data, 1)
    assert error_norm < 1e-10


@pytest.mark.parametrize("X", data_backends)
def test_cdmd_matrix_custom(X):
    matrix = (
        np.random.permutation(
            (sample_data.shape[1] - 3) * sample_data.shape[0]
        )
        .reshape(sample_data.shape[1] - 3, sample_data.shape[0])
        .astype(float)
    )
    matrix /= float(np.sum(matrix))
    dmd = CDMD(compression_matrix=matrix)
    dmd.fit(X=X)
    error_norm = np.linalg.norm(dmd.reconstructed_data - sample_data, 1)
    assert error_norm < 1e-10


def test_sorted_eigs_default():
    dmd = CDMD(compression_matrix="sparse")
    assert dmd.operator._sorted_eigs == False


def test_sorted_eigs_param():
    dmd = CDMD(compression_matrix="sparse", sorted_eigs="real")
    assert dmd.operator._sorted_eigs == "real"


def test_get_bitmask_default():
    dmd = CDMD(
        compression_matrix="normal",
    )
    dmd.fit(X=sample_data)
    assert np.all(dmd.modes_activation_bitmask == True)


@pytest.mark.parametrize("X", data_backends)
def test_set_bitmask(X):
    dmd = CDMD(compression_matrix="normal")
    dmd.fit(X=X)

    new_bitmask = np.full(len(dmd.amplitudes), True, dtype=bool)
    new_bitmask[[0]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.modes_activation_bitmask[0] == False
    assert np.all(dmd.modes_activation_bitmask[1:] == True)


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
    dmd.fit(X=sample_data)
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
    np.testing.assert_almost_equal(
        np.array(dmd.amplitudes), np.array(retained_amplitudes)
    )


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
    np.testing.assert_almost_equal(np.array(dmd.eigs), np.array(retained_eigs))


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
    np.testing.assert_almost_equal(
        np.array(dmd.modes), np.array(retained_modes)
    )


@pytest.mark.parametrize("X", data_backends)
def test_getitem_modes(X):
    dmd = CDMD(compression_matrix="normal", svd_rank=10)
    dmd.fit(X=X)
    old_n_modes = dmd.modes.shape[1]

    modes = np.array(dmd.modes)

    assert dmd[[0, -1]].modes.shape[1] == 2
    np.testing.assert_almost_equal(
        np.array(dmd[[0, -1]].modes), modes[:, [0, -1]]
    )

    assert dmd.modes.shape[1] == old_n_modes

    assert dmd[1::2].modes.shape[1] == old_n_modes // 2
    np.testing.assert_almost_equal(np.array(dmd[1::2].modes), modes[:, 1::2])

    assert dmd.modes.shape[1] == old_n_modes

    assert dmd[[1, 3]].modes.shape[1] == 2
    np.testing.assert_almost_equal(
        np.array(dmd[[1, 3]].modes), modes[:, [1, 3]]
    )

    assert dmd.modes.shape[1] == old_n_modes

    assert dmd[2].modes.shape[1] == 1
    np.testing.assert_almost_equal(
        np.squeeze(np.array(dmd[2].modes)), modes[:, 2]
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
    np.testing.assert_array_almost_equal(dmd.amplitudes, dmd._b)
