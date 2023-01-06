import numpy as np
import pytest
import torch
from pytest import raises

from pydmd.hodmd import HODMD
from pydmd.linalg import build_linalg_module

from .utils import assert_allclose, setup_backends

data_backends = setup_backends()


@pytest.mark.parametrize("X", data_backends)
def test_shape(X):
    dmd = HODMD(svd_rank=-1, d=2, svd_rank_extra=-1)
    dmd.fit(X=X)
    assert dmd.modes.shape[1] == X.shape[1] - 2

@pytest.mark.parametrize("X", data_backends)
def test_truncation_shape(X):
    dmd = HODMD(svd_rank=3)
    dmd.fit(X=X)
    assert dmd.modes.shape[1] == 3

@pytest.mark.parametrize("X", data_backends)
def test_rank(X):
    dmd = HODMD(svd_rank=0.9)
    dmd.fit(X=X)
    assert len(dmd.eigs) == 2

@pytest.mark.parametrize("X", data_backends)
def test_Atilde_shape(X):
    dmd = HODMD(svd_rank=3)
    dmd.fit(X=X)
    assert dmd.atilde.shape == (dmd.svd_rank, dmd.svd_rank)

@pytest.mark.parametrize("X", data_backends)
def test_d(X):
    single_data = np.sin(np.linspace(0, 10, 100))
    single_data = build_linalg_module(X).new_array(single_data)
    dmd = HODMD(svd_rank=-1, d=50, opt=True, exact=True, svd_rank_extra=-1)
    dmd.fit(single_data)
    assert_allclose(dmd.reconstructed_data.flatten(), single_data, atol=1.e-12)
    assert dmd.d == 50

@pytest.mark.parametrize("X", data_backends)
def test_Atilde_values(X):
    dmd = HODMD(svd_rank=2)
    dmd.fit(X=X)
    exact_atilde = [[-0.70558526 + 0.67815084j, 0.22914898 + 0.20020143j],
                    [0.10459069 + 0.09137814j, -0.57730040 + 0.79022994j]]
    assert_allclose(exact_atilde, dmd.atilde)

@pytest.mark.parametrize("X", data_backends)
def test_eigs_1(X):
    dmd = HODMD(svd_rank=-1, svd_rank_extra=-1)
    dmd.fit(X=X)
    assert len(dmd.eigs) == 14

@pytest.mark.parametrize("X", data_backends)
def test_eigs_2(X):
    dmd = HODMD(svd_rank=5, svd_rank_extra=-1)
    dmd.fit(X=X)
    assert len(dmd.eigs) == 5

@pytest.mark.parametrize("X", data_backends)
def test_eigs_3(X):
    dmd = HODMD(svd_rank=2)
    dmd.fit(X=X)
    expected_eigs = [
        -8.09016994e-01 + 5.87785252e-01j, -4.73868662e-01 + 8.80595532e-01j
    ]
    assert_allclose(dmd.eigs, expected_eigs, atol=1.e-6)

@pytest.mark.parametrize("X", data_backends)
def test_eigs_4(X):
    dmd = HODMD(svd_rank=5, svd_rank_extra=4)
    dmd.fit(X=X)
    assert len(dmd.eigs) == 4

@pytest.mark.parametrize("X", data_backends)
def test_dynamics_1(X):
    dmd = HODMD(svd_rank=5, svd_rank_extra=-1)
    dmd.fit(X=X)
    assert dmd.dynamics.shape == (5, X.shape[1])

@pytest.mark.parametrize("X", data_backends)
def test_dynamics_2(X):
    dmd = HODMD(svd_rank=5, svd_rank_extra=4)
    dmd.fit(X=X)
    assert dmd.dynamics.shape == (4, X.shape[1])

@pytest.mark.parametrize("X", data_backends)
def test_dynamics_opt_1(X):
    dmd = HODMD(svd_rank=5, svd_rank_extra=-1, opt=True)
    dmd.fit(X=X)
    assert dmd.dynamics.shape == (5, X.shape[1])

@pytest.mark.parametrize("X", data_backends)
def test_dynamics_opt_2(X):
    dmd = HODMD(svd_rank=5, svd_rank_extra=4, opt=True)
    dmd.fit(X=X)
    assert dmd.dynamics.shape == (4, X.shape[1])

@pytest.mark.parametrize("X", data_backends)
def test_reconstructed_data(X):
    dmd = HODMD(d=2)
    dmd.fit(X=X)
    dmd.reconstructions_of_timeindex(2)
    dmd_data = dmd.reconstructed_data
    assert_allclose(dmd_data, X)

@pytest.mark.parametrize("X", data_backends)
def test_original_time(X):
    dmd = HODMD(svd_rank=2)
    dmd.fit(X=X)
    expected_dict = {'dt': 1, 't0': 0, 'tend': 14}
    assert dmd.original_time == expected_dict

@pytest.mark.parametrize("X", data_backends)
def test_original_timesteps(X):
    dmd = HODMD()
    dmd.fit(X=X)
    assert_allclose(dmd.original_timesteps,
                                np.arange(X.shape[1]))

@pytest.mark.parametrize("X", data_backends)
def test_dmd_time_1(X):
    dmd = HODMD(svd_rank=2)
    dmd.fit(X=X)
    expected_dict = {'dt': 1, 't0': 0, 'tend': 14}
    assert dmd.dmd_time == expected_dict

@pytest.mark.parametrize("X", data_backends)
def test_dmd_time_2(X):
    dmd = HODMD()
    dmd.fit(X=X)
    dmd.dmd_time['t0'] = 10
    dmd.dmd_time['tend'] = 14
    expected_data = X[:, -5:]
    assert_allclose(dmd.reconstructed_data, expected_data)

@pytest.mark.parametrize("X", data_backends)
def test_dmd_time_3(X):
    dmd = HODMD()
    dmd.fit(X=X)
    dmd.dmd_time['t0'] = 8
    dmd.dmd_time['tend'] = 11
    expected_data = X[:, 8:12]
    assert_allclose(dmd.reconstructed_data, expected_data)

@pytest.mark.parametrize("X", data_backends)
def test_dmd_time_4(X):
    dmd = HODMD(svd_rank=3)
    dmd.fit(X=X)
    dmd.dmd_time['t0'] = 20
    dmd.dmd_time['tend'] = 20
    expected_data = [[7.29383297e+00 + 0.0j],
                    [5.69109796e+00 + 2.74068833e+00j],
                    [           0.0 + 0.0j]]
    assert_allclose(dmd.dynamics, expected_data, atol=1.e-6)

@pytest.mark.parametrize("X", data_backends)
def test_dmd_time_5(X):
    x = np.linspace(0, 10, 64)
    y = np.cos(x)*np.sin(np.cos(x)) + np.cos(x*.2)
    y = build_linalg_module(X).new_array(y)

    dmd = HODMD(svd_rank=-1, exact=True, opt=True, d=30, svd_rank_extra=-1)
    dmd.fit(y)

    dmd.original_time['dt'] = dmd.dmd_time['dt'] = x[1] - x[0]
    dmd.original_time['t0'] = dmd.dmd_time['t0'] = x[0]
    dmd.original_time['tend'] = dmd.dmd_time['tend'] = x[-1]

    # assert that the shape of the output is correct
    assert dmd.reconstructed_data.shape == (1,64)

def test_sorted_eigs_default():
    dmd = HODMD()
    assert dmd.operator._sorted_eigs == False

def test_sorted_eigs_param():
    dmd = HODMD(sorted_eigs='real')
    assert dmd.operator._sorted_eigs == 'real'


def test_reconstruction_method_constructor_error():
    with raises(ValueError):
        HODMD(reconstruction_method=[1, 2, 3], d=4)

    with raises(ValueError):
        HODMD(reconstruction_method=np.array([1, 2, 3]), d=4)

    with raises(ValueError):
        HODMD(reconstruction_method=np.array([[1, 2, 3], [3, 4, 5]]), d=3)


def test_reconstruction_method_default_constructor():
    assert HODMD()._reconstruction_method == 'first'

def test_reconstruction_method_constructor():
    assert HODMD(reconstruction_method='mean')._reconstruction_method == 'mean'
    assert HODMD(reconstruction_method=[3])._reconstruction_method == [3]
    assert_allclose(HODMD(reconstruction_method=np.array([1, 2]), d=2)._reconstruction_method, [1, 2])

@pytest.mark.parametrize("X", data_backends)
def test_nonan_nomask(X):
    dmd = HODMD(d=3)
    dmd.fit(X=X)
    rec = dmd.reconstructed_data

    assert not isinstance(rec, np.ma.MaskedArray)
    assert not np.nan in rec

@pytest.mark.parametrize("X", data_backends)
def test_extract_versions_nonan(X):
    dmd = HODMD(d=3)
    dmd.fit(X=X)
    for timeindex in range(X.shape[1]):
        assert not np.nan in dmd.reconstructions_of_timeindex(timeindex)

@pytest.mark.parametrize("X", data_backends)
def test_rec_method_first(X):
    dmd = HODMD(d=3, reconstruction_method="first")
    dmd.fit(X=X)

    rec = dmd.reconstructed_data
    allrec = dmd.reconstructions_of_timeindex()
    for i in range(rec.shape[1]):
        assert (rec[:,i] == allrec[i, min(i,dmd.d-1)]).all()

@pytest.mark.parametrize("X", data_backends)
def test_rec_method_mean(X):
    dmd = HODMD(d=3, reconstruction_method='mean')
    dmd.fit(X=X)
    assert_allclose(dmd.reconstructed_data[:,2], dmd.reconstructions_of_timeindex(2).mean(axis=0).T)

@pytest.mark.parametrize("X", data_backends)
def test_rec_method_weighted(X):
    dmd = HODMD(d=2, svd_rank_extra=-1,reconstruction_method=[10,20])
    dmd.fit(X=X)
    wavg = np.average(np.array(dmd.reconstructions_of_timeindex(4)), axis=0, weights=[10,20])
    assert_allclose(dmd.reconstructed_data[:,4], wavg.T)

@pytest.mark.parametrize("X", data_backends)
def test_scalar_func_warning(X):
    x = np.linspace(0, 10, 64)
    arr = np.cos(x) * np.sin(np.cos(x)) + np.cos(x * 0.2)
    arr = build_linalg_module(X).new_array(arr)
    # we check that this does not fail
    HODMD(svd_rank=1, exact=True, opt=True, d=3).fit(arr)

@pytest.mark.parametrize("X", data_backends)
def test_get_bitmask_default(X):
    dmd = HODMD(svd_rank=-1, d=5)
    dmd.fit(X=X)
    assert dmd.modes_activation_bitmask.all()

@pytest.mark.parametrize("X", data_backends)
def test_set_bitmask(X):
    dmd = HODMD(svd_rank=-1, d=5)
    dmd.fit(X=X)

    new_bitmask = np.full(len(dmd.amplitudes), True, dtype=bool)
    new_bitmask[[0]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.modes_activation_bitmask[0] == False
    assert dmd.modes_activation_bitmask[1:].all()

def test_not_fitted_get_bitmask_raises():
    dmd = HODMD(svd_rank=-1, d=5)
    with raises(RuntimeError):
        print(dmd.modes_activation_bitmask)

def test_not_fitted_set_bitmask_raises():
    dmd = HODMD(svd_rank=-1, d=5)
    with raises(RuntimeError):
        dmd.modes_activation_bitmask = np.full(3, True, dtype=bool)

@pytest.mark.parametrize("X", data_backends)
def test_raise_wrong_dtype_bitmask(X):
    dmd = HODMD(svd_rank=-1, d=5)
    dmd.fit(X=X)
    with raises(RuntimeError):
        dmd.modes_activation_bitmask = np.full(3, 0.1)

@pytest.mark.parametrize("X", data_backends)
def test_fitted(X):
    dmd = HODMD(svd_rank=-1, d=5)
    assert not dmd.fitted
    dmd.fit(X=X)
    assert dmd.fitted

@pytest.mark.parametrize("X", data_backends)
def test_bitmask_amplitudes(X):
    dmd = HODMD(svd_rank=-1, d=5)
    dmd.fit(X=X)

    old_n_amplitudes = dmd.amplitudes.shape[0]
    retained_amplitudes = np.delete(dmd.amplitudes, [0,-1])

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0,-1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.amplitudes.shape[0] == old_n_amplitudes - 2
    assert_allclose(dmd.amplitudes, retained_amplitudes)

@pytest.mark.parametrize("X", data_backends)
def test_bitmask_eigs(X):
    dmd = HODMD(svd_rank=-1, d=5)
    dmd.fit(X=X)

    old_n_eigs = dmd.eigs.shape[0]
    retained_eigs = np.delete(dmd.eigs, [0,-1])

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0,-1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.eigs.shape[0] == old_n_eigs - 2
    assert_allclose(dmd.eigs, retained_eigs)

@pytest.mark.parametrize("X", data_backends)
def test_bitmask_modes(X):
    dmd = HODMD(svd_rank=-1, d=5)
    dmd.fit(X=X)

    old_n_modes = dmd.modes.shape[1]
    retained_modes = np.delete(dmd.modes, [0,-1], axis=1)

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0,-1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.modes.shape[1] == old_n_modes - 2
    assert_allclose(dmd.modes, retained_modes)

@pytest.mark.parametrize("X", data_backends)
def test_reconstructed_data_with_bitmask(X):
    dmd = HODMD(svd_rank=-1, d=5)
    dmd.fit(X=X)

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0,-1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.reconstructed_data is not None

@pytest.mark.parametrize("X", data_backends)
def test_getitem_modes(X):
    dmd = HODMD(svd_rank=-1, d=5)
    dmd.fit(X=X)
    old_n_modes = dmd.modes.shape[1]

    assert dmd[[0,-1]].modes.shape[1] == 2
    assert_allclose(dmd[[0,-1]].modes, dmd.modes[:,[0,-1]])

    assert dmd.modes.shape[1] == old_n_modes

    assert dmd[1::2].modes.shape[1] == old_n_modes // 2
    assert_allclose(dmd[1::2].modes, dmd.modes[:,1::2])

    assert dmd.modes.shape[1] == old_n_modes

    assert dmd[[1,3]].modes.shape[1] == 2
    assert_allclose(dmd[[1,3]].modes, dmd.modes[:,[1,3]])

    assert dmd.modes.shape[1] == old_n_modes

    assert dmd[2].modes.shape[1] == 1
    assert_allclose(np.squeeze(dmd[2].modes), dmd.modes[:,2])

    assert dmd.modes.shape[1] == old_n_modes

@pytest.mark.parametrize("X", data_backends)
def test_getitem_raises(X):
    dmd = HODMD(svd_rank=-1, d=5)
    dmd.fit(X=X)

    with raises(ValueError):
        dmd[[0,1,1,0,1]]
    with raises(ValueError):
        dmd[[True, True, False, True]]
    with raises(ValueError):
        dmd[1.0]

@pytest.mark.parametrize("X", data_backends)
def test_correct_amplitudes(X):
    dmd = HODMD(svd_rank=-1, d=5)
    dmd.fit(X=X)
    assert_allclose(dmd.amplitudes, dmd._sub_dmd._b)

@pytest.mark.parametrize("X", data_backends)
def test_raises_not_enough_snapshots(X):
    dmd = HODMD(svd_rank=-1, d=5)
    linalg_module = build_linalg_module(X)
    with raises(ValueError,  match="The number of snapshots provided is not enough for d=5."):
        dmd.fit(linalg_module.new_array(np.ones((20,4))))
    dmd.fit(linalg_module.new_array(np.ones((20,5))))