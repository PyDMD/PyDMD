from builtins import range
from pydmd.fbdmd import FbDMD
import matplotlib.pyplot as plt
import numpy as np
from pytest import raises


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


sample_data = create_noisy_data()


def test_modes_shape():
    dmd = FbDMD(svd_rank=-1)
    dmd.fit(X=sample_data)
    assert dmd.modes.shape[1] == 2

def test_truncation_shape():
    dmd = FbDMD(svd_rank=1)
    dmd.fit(X=sample_data)
    assert dmd.modes.shape[1] == 1

def test_dynamics():
    dmd = FbDMD(svd_rank=-1)
    dmd.fit(X=sample_data)
    assert dmd.dynamics.shape == (2, 100)

def test_eigs_1():
    dmd = FbDMD(svd_rank=-1)
    dmd.fit(X=sample_data)
    assert len(dmd.eigs) == 2

def test_eigs_2():
    dmd = FbDMD(svd_rank=1)
    dmd.fit(X=sample_data)
    assert len(dmd.eigs) == 1

def test_eigs_modulus_1():
    dmd = FbDMD(svd_rank=-1)
    dmd.fit(X=sample_data)
    np.testing.assert_almost_equal(
        np.linalg.norm(dmd.eigs[0]), 1., decimal=6)

def test_eigs_modulus_2():
    dmd = FbDMD(svd_rank=-1, exact=True)
    dmd.fit(X=sample_data)
    np.testing.assert_almost_equal(
        np.linalg.norm(dmd.eigs[1]), 1., decimal=6)

def test_reconstructed_data():
    dmd = FbDMD(exact=True, svd_rank=-1)
    dmd.fit(X=sample_data)
    dmd_data = dmd.reconstructed_data
    dmd_data_correct = np.load('tests/test_datasets/fbdmd_data.npy')
    assert np.allclose(dmd_data, dmd_data_correct)

def test_plot_eigs_1():
    dmd = FbDMD(svd_rank=-1)
    dmd.fit(X=sample_data)
    dmd.plot_eigs(show_axes=True, show_unit_circle=True)
    plt.close()

def test_plot_eigs_2():
    dmd = FbDMD(svd_rank=-1)
    dmd.fit(X=sample_data)
    dmd.plot_eigs(show_axes=False, show_unit_circle=False)
    plt.close()

def test_sorted_eigs_default():
    dmd = FbDMD(svd_rank=-1)
    assert dmd.operator._sorted_eigs == False

def test_sorted_eigs_param():
    dmd = FbDMD(svd_rank=-1, sorted_eigs='real')
    assert dmd.operator._sorted_eigs == 'real'

def test_get_bitmask_default():
    dmd = FbDMD(svd_rank=-1)
    dmd.fit(X=sample_data)
    assert np.all(dmd.modes_activation_bitmask == True)

def test_set_bitmask():
    dmd = FbDMD(svd_rank=-1)
    dmd.fit(X=sample_data)

    new_bitmask = np.full(len(dmd.amplitudes), True, dtype=bool)
    new_bitmask[[0]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.modes_activation_bitmask[0] == False
    assert np.all(dmd.modes_activation_bitmask[1:] == True)

def test_not_fitted_get_bitmask_raises():
    dmd = FbDMD(svd_rank=-1)
    with raises(RuntimeError):
        print(dmd.modes_activation_bitmask)

def test_not_fitted_set_bitmask_raises():
    dmd = FbDMD(svd_rank=-1)
    with raises(RuntimeError):
        dmd.modes_activation_bitmask = np.full(3, True, dtype=bool)

def test_raise_wrong_dtype_bitmask():
    dmd = FbDMD(svd_rank=-1)
    dmd.fit(X=sample_data)
    with raises(RuntimeError):
        dmd.modes_activation_bitmask = np.full(3, 0.1)

def test_fitted():
    dmd = FbDMD(svd_rank=-1)
    assert not dmd.fitted
    dmd.fit(X=sample_data)
    assert dmd.fitted

def test_bitmask_amplitudes():
    dmd = FbDMD(svd_rank=-1)
    dmd.fit(X=sample_data)

    old_n_amplitudes = dmd.amplitudes.shape[0]
    retained_amplitudes = np.delete(dmd.amplitudes, [0,-1])

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0,-1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.amplitudes.shape[0] == old_n_amplitudes - 2
    np.testing.assert_almost_equal(dmd.amplitudes, retained_amplitudes)

def test_bitmask_eigs():
    dmd = FbDMD(svd_rank=-1)
    dmd.fit(X=sample_data)

    old_n_eigs = dmd.eigs.shape[0]
    retained_eigs = np.delete(dmd.eigs, [0,-1])

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0,-1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.eigs.shape[0] == old_n_eigs - 2
    np.testing.assert_almost_equal(dmd.eigs, retained_eigs)

def test_bitmask_modes():
    dmd = FbDMD(svd_rank=-1)
    dmd.fit(X=sample_data)

    old_n_modes = dmd.modes.shape[1]
    retained_modes = np.delete(dmd.modes, [0,-1], axis=1)

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0,-1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.modes.shape[1] == old_n_modes - 2
    np.testing.assert_almost_equal(dmd.modes, retained_modes)

def test_reconstructed_data():
    dmd = FbDMD(svd_rank=-1)
    dmd.fit(X=sample_data)

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0,-1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    dmd.reconstructed_data
    assert True

def test_getitem_modes():
    dmd = FbDMD(svd_rank=-1)
    dmd.fit(X=np.load('tests/test_datasets/input_sample.npy'))
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
    dmd = FbDMD(svd_rank=-1)
    dmd.fit(X=np.load('tests/test_datasets/input_sample.npy'))

    with raises(ValueError):
        dmd[[0,1,1,0,1]]
    with raises(ValueError):
        dmd[[True, True, False, True]]
    with raises(ValueError):
        dmd[1.0]

# this is a test for the correctness of the amplitudes saved in the Proxy
# between DMDBase and the modes activation bitmask. if this test fails
# you probably need to call allocate_proxy once again after you compute
# the final value of the amplitudes
def test_correct_amplitudes():
    dmd = FbDMD(svd_rank=-1)
    dmd.fit(X=np.load('tests/test_datasets/input_sample.npy'))
    np.testing.assert_array_almost_equal(dmd.amplitudes, dmd._b)
