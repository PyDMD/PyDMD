import numpy as np
import pytest
import torch
from pytest import raises

from pydmd import HankelDMD
from pydmd.linalg import build_linalg_module

from .utils import assert_allclose, setup_backends

data_backends = setup_backends()


@pytest.mark.parametrize("X", data_backends)
def test_shape(X):
    dmd = HankelDMD(svd_rank=-1)
    dmd.fit(X=X)
    assert dmd.modes.shape[1] == X.shape[1] - 1

@pytest.mark.parametrize("X", data_backends)
def test_truncation_shape(X):
    dmd = HankelDMD(svd_rank=3)
    dmd.fit(X=X)
    assert dmd.modes.shape[1] == 3

@pytest.mark.parametrize("X", data_backends)
def test_rank(X):
    dmd = HankelDMD(svd_rank=0.9)
    dmd.fit(X=X)
    assert len(dmd.eigs) == 2

@pytest.mark.parametrize("X", data_backends)
def test_Atilde_shape(X):
    dmd = HankelDMD(svd_rank=3)
    dmd.fit(X=X)
    assert dmd.atilde.shape == (dmd.svd_rank, dmd.svd_rank)

@pytest.mark.parametrize("X", data_backends)
def test_d(X):
    single_data = np.sin(np.linspace(0, 10, 100))
    single_data = build_linalg_module(X).new_array(single_data)
    dmd = HankelDMD(svd_rank=-1, d=50, opt=True, exact=True)
    dmd.fit(single_data)
    assert_allclose(dmd.reconstructed_data.flatten(), single_data, atol=1.e-12)
    assert dmd.d == 50

@pytest.mark.parametrize("X", data_backends)
def test_Atilde_values(X):
    dmd = HankelDMD(svd_rank=2)
    dmd.fit(X=X)
    exact_atilde = [
            [-0.70558526 + 0.67815084j, 0.22914898 + 0.20020143j],
            [0.10459069 + 0.09137814j, -0.57730040 + 0.79022994j],
    ]
    assert_allclose(exact_atilde, dmd.atilde)

@pytest.mark.parametrize("X", data_backends)
def test_eigs_1(X):
    dmd = HankelDMD(svd_rank=-1)
    dmd.fit(X=X)
    assert len(dmd.eigs) == 14

@pytest.mark.parametrize("X", data_backends)
def test_eigs_2(X):
    dmd = HankelDMD(svd_rank=5)
    dmd.fit(X=X)
    assert len(dmd.eigs) == 5

@pytest.mark.parametrize("X", data_backends)
def test_eigs_3(X):
    dmd = HankelDMD(svd_rank=2)
    dmd.fit(X=X)
    expected_eigs = [
            -8.09016994e-01 + 5.87785252e-01j,
            -4.73868662e-01 + 8.80595532e-01j,
        ]
    assert_allclose(dmd.eigs, expected_eigs, atol=1.e-6)

@pytest.mark.parametrize("X", data_backends)
def test_dynamics_1(X):
    dmd = HankelDMD(svd_rank=5)
    dmd.fit(X=X)
    assert dmd.dynamics.shape == (5, X.shape[1])

@pytest.mark.parametrize("X", data_backends)
def test_dynamics_2(X):
    dmd = HankelDMD(svd_rank=1)
    dmd.fit(X=X)
    expected_dynamics = [[
                -2.20639502 - 9.10168802e-16j,
                1.55679980 - 1.49626864e00j,
                -0.08375915 + 2.11149018e00j,
                -1.37280962 - 1.54663768e00j,
                2.01748787 + 1.60312745e-01j,
                -1.53222592 + 1.25504678e00j,
                0.23000498 - 1.92462280e00j,
                1.14289644 + 1.51396355e00j,
                -1.83310653 - 2.93174173e-01j,
                1.49222925 - 1.03626336e00j,
                -0.35015209 + 1.74312867e00j,
                -0.93504202 - 1.46738182e00j,
                1.65485808 + 4.01263449e-01j,
                -1.43976061 + 8.39117825e-01j,
                0.44682540 - 1.56844403e00j,
    ]]
    assert_allclose(dmd.dynamics, expected_dynamics)

@pytest.mark.parametrize("X", data_backends)
def test_dynamics_opt_1(X):
    dmd = HankelDMD(svd_rank=5, opt=True)
    dmd.fit(X=X)
    assert dmd.dynamics.shape == (5, X.shape[1])

@pytest.mark.parametrize("X", data_backends)
def test_dynamics_opt_2(X):
    dmd = HankelDMD(svd_rank=1, opt=True, exact=False)
    dmd.fit(X=X)
    expected_dynamics = [[
        -4.609718826226513-6.344781724790875j,
        7.5552686987577165+1.3506997434096375j,
        -6.246864367654589+4.170577993207872j,
        1.5794144248628537-7.179014663490048j,
        3.754043295828462+6.13648812118528j,
        -6.810262177959786-1.7840079278093528j,
        6.015047060133346-3.35961532862783j,
        -1.9658025630719695+6.449604262000736j,
        -2.9867632454837936-5.8838563367460734j,
        6.097558230017521+2.126086276430128j,
        -5.7441543819530265+2.6349291080417103j,
        2.266111252852836-5.7545702519088895j,
        2.303531963541068+5.597105176945707j,
        -5.421019770795679-2.3870927539102658j,
        5.443800581850978-1.9919716610066682j,
    ]]
    assert_allclose(dmd.dynamics, expected_dynamics)

@pytest.mark.parametrize("X", data_backends)
def test_reconstructed_data(X):
    dmd = HankelDMD()
    dmd.fit(X=X)
    dmd_data = dmd.reconstructed_data
    assert_allclose(dmd_data, X)

@pytest.mark.parametrize("X", data_backends)
def test_original_time(X):
    dmd = HankelDMD(svd_rank=2)
    dmd.fit(X=X)
    expected_dict = {"dt": 1, "t0": 0, "tend": 14}
    assert dmd.original_time == expected_dict

@pytest.mark.parametrize("X", data_backends)
def test_original_timesteps(X):
    dmd = HankelDMD()
    dmd.fit(X=X)
    assert_allclose(
        dmd.original_timesteps, np.arange(X.shape[1])
    )

@pytest.mark.parametrize("X", data_backends)
def test_dmd_time_1(X):
    dmd = HankelDMD(svd_rank=2)
    dmd.fit(X=X)
    expected_dict = {"dt": 1, "t0": 0, "tend": 14}
    assert dmd.dmd_time == expected_dict

@pytest.mark.parametrize("X", data_backends)
def test_dmd_time_2(X):
    dmd = HankelDMD()
    dmd.fit(X=X)
    dmd.dmd_time["t0"] = 10
    dmd.dmd_time["tend"] = 14
    expected_data = X[:, -5:]
    assert_allclose(dmd.reconstructed_data, expected_data)

@pytest.mark.parametrize("X", data_backends)
def test_dmd_time_3(X):
    dmd = HankelDMD()
    dmd.fit(X=X)
    dmd.dmd_time["t0"] = 8
    dmd.dmd_time["tend"] = 11
    expected_data = X[:, 8:12]
    assert_allclose(dmd.reconstructed_data, expected_data)

@pytest.mark.parametrize("X", data_backends)
def test_dmd_time_4(X):
    dmd = HankelDMD(svd_rank=3)
    dmd.fit(X=X)
    dmd.dmd_time["t0"] = 20
    dmd.dmd_time["tend"] = 20
    expected_data = [
            [-7.29383297e00 - 4.90248179e-14j],
            [-5.69109796e00 - 2.74068833e00j],
            [3.38410649e-83 + 3.75677740e-83j],
        ]
    assert_allclose(dmd.dynamics, expected_data, atol=1.e-6)

@pytest.mark.parametrize("X", data_backends)
def test_dmd_time_5(X):
    x = np.linspace(0, 10, 64)
    y = np.cos(x) * np.sin(np.cos(x)) + np.cos(x * 0.2)
    y = build_linalg_module(X).new_array(y)

    dmd = HankelDMD(svd_rank=-1, exact=True, opt=True, d=30)
    dmd.fit(y)

    dmd.original_time["dt"] = dmd.dmd_time["dt"] = x[1] - x[0]
    dmd.original_time["t0"] = dmd.dmd_time["t0"] = x[0]
    dmd.original_time["tend"] = dmd.dmd_time["tend"] = x[-1]

    assert dmd.reconstructed_data.shape == (1, 64)

def test_sorted_eigs_default():
    dmd = HankelDMD()
    assert dmd.operator._sorted_eigs == False

def test_sorted_eigs_param():
    dmd = HankelDMD(sorted_eigs="real")
    assert dmd.operator._sorted_eigs == "real"

@pytest.mark.parametrize("X", data_backends)
def test_reconstruction_method_constructor_error(X):
    with raises(ValueError):
        HankelDMD(reconstruction_method=[1, 2, 3], d=4)

    with raises(ValueError):
        HankelDMD(reconstruction_method=np.array([1, 2, 3]), d=4)

    with raises(ValueError):
        HankelDMD(
            reconstruction_method=np.array([[1, 2, 3], [3, 4, 5]]), d=3
        )

def test_reconstruction_method_default_constructor():
    assert HankelDMD()._reconstruction_method == "first"

def test_reconstruction_method_constructor():
    assert (
        HankelDMD(reconstruction_method="mean")._reconstruction_method
        == "mean"
    )
    assert HankelDMD(reconstruction_method=[3])._reconstruction_method == [
        3
    ]
    assert all(
        HankelDMD(
            reconstruction_method=np.array([1, 2]), d=2
        )._reconstruction_method
        == np.array([1, 2])
    )

@pytest.mark.parametrize("X", data_backends)
def test_nonan_nomask(X):
    dmd = HankelDMD(d=3)
    dmd.fit(X=X)
    rec = dmd.reconstructed_data

    assert not isinstance(rec, np.ma.MaskedArray)
    assert not np.nan in rec

@pytest.mark.parametrize("X", data_backends)
def test_extract_versions_nonan(X):
    dmd = HankelDMD(d=3)
    dmd.fit(X=X)
    for timeindex in range(X.shape[1]):
        assert not np.nan in dmd.reconstructions_of_timeindex(timeindex)

@pytest.mark.parametrize("X", data_backends)
def test_rec_method_first(X):
    dmd = HankelDMD(d=3, reconstruction_method="first")
    dmd.fit(X=X)

    rec = dmd.reconstructed_data
    allrec = dmd.reconstructions_of_timeindex()
    for i in range(rec.shape[1]):
        assert (rec[:,i] == allrec[i, min(i,dmd.d-1)]).all()

@pytest.mark.parametrize("X", data_backends)
def test_rec_method_mean(X):
    dmd = HankelDMD(d=3, reconstruction_method="mean")
    dmd.fit(X=X)
    assert_allclose(dmd.reconstructed_data[:,2], dmd.reconstructions_of_timeindex(2).mean(axis=0))

@pytest.mark.parametrize("X", data_backends)
def test_rec_method_weighted(X):
    dmd = HankelDMD(d=2, reconstruction_method=[10, 20])
    dmd.fit(X=X)
    assert_allclose(dmd.reconstructed_data[..., 4], np.average(
            dmd.reconstructions_of_timeindex(4), axis=0, weights=[10, 20]
        )
    )

@pytest.mark.parametrize("X", data_backends)
def test_hankeldmd_timesteps(X):
    x = np.linspace(0, 10, 64)
    
    arr = np.cos(x) * np.sin(np.cos(x)) + np.cos(x * 0.2)
    arr = build_linalg_module(X).new_array(arr)

    dmd = HankelDMD(svd_rank=1, exact=True, opt=True, d=30).fit(arr)
    assert len(dmd.dmd_timesteps) == 64

@pytest.mark.parametrize("X", data_backends)
def test_update_sub_dmd_time(X):
    dmd = HankelDMD()
    x = np.linspace(0, 10, 64)
    
    arr = np.cos(x) * np.sin(np.cos(x)) + np.cos(x * 0.2)
    arr = build_linalg_module(X).new_array(arr)

    dmd = HankelDMD(svd_rank=1, exact=True, opt=True, d=3).fit(arr)

    dmd.dmd_time["tend"] += dmd.dmd_time["dt"] * 20
    dmd._update_sub_dmd_time()

    # assert that the dt for the sub_dmd is always 1
    assert dmd._sub_dmd.dmd_time["dt"] == 1
    assert dmd._sub_dmd.original_time["dt"] == 1

    assert (
        dmd._sub_dmd.dmd_time["tend"]
        == dmd._sub_dmd.original_time["tend"] + 20
    )

@pytest.mark.parametrize("X", data_backends)
def test_hankel_2d(X):
    def fnc(x):
        return np.cos(x) * np.sin(np.cos(x)) + np.cos(x * 0.2)

    x = np.linspace(0, 10, 64)
    snapshots = np.vstack([fnc(x), -fnc(x)])
    snapshots = build_linalg_module(X).new_array(snapshots)

    dmd = HankelDMD(svd_rank=0, exact=True, opt=True, d=30).fit(snapshots)

    dmd.original_time["dt"] = dmd.dmd_time["dt"] = x[1] - x[0]
    dmd.original_time["t0"] = dmd.dmd_time["t0"] = x[0]
    dmd.original_time["tend"] = dmd.dmd_time["tend"] = x[-1]

    dmd.dmd_time["t0"] = x[len(x) // 2]
    dmd.dmd_time["tend"] = x[-1] + dmd.dmd_time["dt"] * 20

    assert len(dmd.dmd_timesteps) == dmd.reconstructed_data.shape[1]

    assert_allclose(
        dmd.reconstructed_data,
        np.vstack([fnc(dmd.dmd_timesteps), -fnc(dmd.dmd_timesteps)]),
        atol=1.e-6
    )

@pytest.mark.parametrize("X", data_backends)
def test_get_bitmask_default(X):
    dmd = HankelDMD(svd_rank=-1, d=5)
    dmd.fit(X=X)
    assert dmd.modes_activation_bitmask.all()

@pytest.mark.parametrize("X", data_backends)
def test_set_bitmask(X):
    dmd = HankelDMD(svd_rank=-1, d=5)
    dmd.fit(X=X)

    new_bitmask = np.full(len(dmd.amplitudes), True, dtype=bool)
    new_bitmask[[0]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.modes_activation_bitmask[0] == False
    assert dmd.modes_activation_bitmask[1:].all()

@pytest.mark.parametrize("X", data_backends)
def test_not_fitted_get_bitmask_raises(X):
    dmd = HankelDMD(svd_rank=-1, d=5)
    with raises(RuntimeError):
        print(dmd.modes_activation_bitmask)

@pytest.mark.parametrize("X", data_backends)
def test_not_fitted_set_bitmask_raises(X):
    dmd = HankelDMD(svd_rank=-1, d=5)
    with raises(RuntimeError):
        dmd.modes_activation_bitmask = np.full(3, True, dtype=bool)

@pytest.mark.parametrize("X", data_backends)
def test_raise_wrong_dtype_bitmask(X):
    dmd = HankelDMD(svd_rank=-1, d=5)
    dmd.fit(X=X)
    with raises(RuntimeError):
        dmd.modes_activation_bitmask = np.full(3, 0.1)

@pytest.mark.parametrize("X", data_backends)
def test_fitted(X):
    dmd = HankelDMD(svd_rank=-1, d=5)
    assert not dmd.fitted
    dmd.fit(X=X)
    assert dmd.fitted

@pytest.mark.parametrize("X", data_backends)
def test_bitmask_amplitudes(X):
    dmd = HankelDMD(svd_rank=-1, d=5)
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
    dmd = HankelDMD(svd_rank=-1, d=5)
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
    dmd = HankelDMD(svd_rank=-1, d=5)
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
    dmd = HankelDMD(svd_rank=-1, d=5)
    dmd.fit(X=X)

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0,-1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    dmd.reconstructed_data
    assert True

@pytest.mark.parametrize("X", data_backends)
def test_getitem_modes(X):
    dmd = HankelDMD(svd_rank=-1, d=5)
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
    dmd = HankelDMD(svd_rank=-1, d=5)
    dmd.fit(X=X)

    with raises(ValueError):
        dmd[[0,1,1,0,1]]
    with raises(ValueError):
        dmd[[True, True, False, True]]
    with raises(ValueError):
        dmd[1.0]

@pytest.mark.parametrize("X", data_backends)
def test_correct_amplitudes(X):
    dmd = HankelDMD(svd_rank=-1, d=5)
    dmd.fit(X=X)
    assert_allclose(dmd.amplitudes, dmd._sub_dmd._b)

def test_raises_not_enough_snapshots():
    dmd = HankelDMD(svd_rank=-1, d=5)
    with raises(ValueError,  match="The number of snapshots provided is not enough for d=5."):
        dmd.fit(np.ones((20,4)))
    dmd.fit(np.ones((20,5)))