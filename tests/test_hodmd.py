from builtins import range

import numpy as np
from pytest import raises

from pydmd.hodmd import HODMD

# 15 snapshot with 400 data. The matrix is 400x15 and it contains
# the following data: f1 + f2 where
# f1 = lambda x,t: sech(x+3)*(1.*np.exp(1j*2.3*t))
# f2 = lambda x,t: (sech(x)*np.tanh(x))*(2.*np.exp(1j*2.8*t))
sample_data = np.load("tests/test_datasets/input_sample.npy")


def create_noisy_data():
    mu = 0.0
    sigma = 0.0  # noise standard deviation
    m = 100  # number of snapshot
    noise = np.random.normal(mu, sigma, m)  # gaussian noise
    A = np.array([[1.0, 1.0], [-1.0, 2.0]])
    A /= np.sqrt(3)
    n = 2
    X = np.zeros((n, m))
    X[:, 0] = np.array([0.5, 1.0])
    # evolve the system and perturb the data with noise
    for k in range(1, m):
        X[:, k] = A.dot(X[:, k - 1])
        X[:, k - 1] += noise[k - 1]
    return X


noisy_data = create_noisy_data()


def test_shape():
    dmd = HODMD(svd_rank=-1, d=2, svd_rank_extra=-1)
    dmd.fit(X=sample_data)
    assert dmd.modes.shape[1] == sample_data.shape[1] - 2


def test_truncation_shape():
    dmd = HODMD(svd_rank=3)
    dmd.fit(X=sample_data)
    assert dmd.modes.shape[1] == 3


def test_rank():
    dmd = HODMD(svd_rank=0.9)
    dmd.fit(X=sample_data)
    assert len(dmd.eigs) == 2


def test_Atilde_shape():
    dmd = HODMD(svd_rank=3)
    dmd.fit(X=sample_data)
    assert dmd.operator.as_numpy_array.shape == (
        dmd.operator._svd_rank,
        dmd.operator._svd_rank,
    )


def test_d():
    single_data = np.sin(np.linspace(0, 10, 100))[None]
    dmd = HODMD(svd_rank=-1, d=50, opt=True, exact=True, svd_rank_extra=-1)
    dmd.fit(single_data)
    assert np.allclose(dmd.reconstructed_data, single_data)
    assert dmd.d == 50


def test_Atilde_values():
    dmd = HODMD(svd_rank=2)
    dmd.fit(X=sample_data)
    exact_atilde = np.array(
        [
            [-0.70558526 + 0.67815084j, 0.22914898 + 0.20020143j],
            [0.10459069 + 0.09137814j, -0.57730040 + 0.79022994j],
        ]
    )
    np.testing.assert_allclose(exact_atilde, dmd.operator.as_numpy_array)


def test_eigs_1():
    dmd = HODMD(svd_rank=-1, svd_rank_extra=-1)
    dmd.fit(X=sample_data)
    assert len(dmd.eigs) == 14


def test_eigs_2():
    dmd = HODMD(svd_rank=5, svd_rank_extra=-1)
    dmd.fit(X=sample_data)
    assert len(dmd.eigs) == 5


def test_eigs_3():
    dmd = HODMD(svd_rank=2)
    dmd.fit(X=sample_data)
    expected_eigs = np.array(
        [-8.09016994e-01 + 5.87785252e-01j, -4.73868662e-01 + 8.80595532e-01j]
    )
    np.testing.assert_almost_equal(dmd.eigs, expected_eigs, decimal=6)


def test_eigs_4():
    dmd = HODMD(svd_rank=5, svd_rank_extra=4)
    dmd.fit(X=sample_data)
    assert len(dmd.eigs) == 4


def test_dynamics_1():
    dmd = HODMD(svd_rank=5, svd_rank_extra=-1)
    dmd.fit(X=sample_data)
    assert dmd.dynamics.shape == (5, sample_data.shape[1])


def test_dynamics_2():
    dmd = HODMD(svd_rank=5, svd_rank_extra=4)
    dmd.fit(X=sample_data)
    assert dmd.dynamics.shape == (4, sample_data.shape[1])


def test_dynamics_opt_1():
    dmd = HODMD(svd_rank=5, svd_rank_extra=-1, opt=True)
    dmd.fit(X=sample_data)
    assert dmd.dynamics.shape == (5, sample_data.shape[1])


def test_dynamics_opt_2():
    dmd = HODMD(svd_rank=5, svd_rank_extra=4, opt=True)
    dmd.fit(X=sample_data)
    assert dmd.dynamics.shape == (4, sample_data.shape[1])


def test_reconstructed_data():
    dmd = HODMD(d=2)
    dmd.fit(X=sample_data)
    dmd.reconstructions_of_timeindex(2)
    dmd_data = dmd.reconstructed_data
    np.testing.assert_allclose(dmd_data, sample_data)


def test_original_time():
    dmd = HODMD(svd_rank=2)
    dmd.fit(X=sample_data)
    expected_dict = {"dt": 1, "t0": 0, "tend": 14}
    np.testing.assert_equal(dmd.original_time, expected_dict)


def test_original_timesteps():
    dmd = HODMD()
    dmd.fit(X=sample_data)
    np.testing.assert_allclose(
        dmd.original_timesteps, np.arange(sample_data.shape[1])
    )


def test_dmd_time_1():
    dmd = HODMD(svd_rank=2)
    dmd.fit(X=sample_data)
    expected_dict = {"dt": 1, "t0": 0, "tend": 14}
    np.testing.assert_equal(dmd.dmd_time, expected_dict)


def test_dmd_time_2():
    dmd = HODMD()
    dmd.fit(X=sample_data)
    dmd.dmd_time["t0"] = 10
    dmd.dmd_time["tend"] = 14
    expected_data = sample_data[:, -5:]
    np.testing.assert_allclose(dmd.reconstructed_data, expected_data)


def test_dmd_time_3():
    dmd = HODMD()
    dmd.fit(X=sample_data)
    dmd.dmd_time["t0"] = 8
    dmd.dmd_time["tend"] = 11
    expected_data = sample_data[:, 8:12]
    np.testing.assert_allclose(dmd.reconstructed_data, expected_data)


def test_dmd_time_4():
    dmd = HODMD(svd_rank=3)
    dmd.fit(X=sample_data)
    dmd.dmd_time["t0"] = 20
    dmd.dmd_time["tend"] = 20
    expected_data = np.array(
        [[7.29383297e00 + 0.0j], [5.69109796e00 + 2.74068833e00j], [0.0 + 0.0j]]
    )
    np.testing.assert_almost_equal(dmd.dynamics, expected_data, decimal=6)


def test_dmd_time_5():
    x = np.linspace(0, 10, 64)[None]
    y = np.cos(x) * np.sin(np.cos(x)) + np.cos(x * 0.2)

    dmd = HODMD(svd_rank=-1, exact=True, opt=True, d=30, svd_rank_extra=-1)
    dmd.fit(y)

    dmd.original_time["dt"] = dmd.dmd_time["dt"] = x[0, 1] - x[0, 0]
    dmd.original_time["t0"] = dmd.dmd_time["t0"] = x[0, 0]
    dmd.original_time["tend"] = dmd.dmd_time["tend"] = x[0, -1]

    # assert that the shape of the output is correct
    assert dmd.reconstructed_data.shape == (1, 64)


def test_sorted_eigs_default():
    dmd = HODMD()
    assert dmd.operator._sorted_eigs == False


def test_sorted_eigs_param():
    dmd = HODMD(sorted_eigs="real")
    assert dmd.operator._sorted_eigs == "real"


def test_reconstruction_method_constructor_error():
    with raises(ValueError):
        HODMD(reconstruction_method=[1, 2, 3], d=4)

    with raises(ValueError):
        HODMD(reconstruction_method=np.array([1, 2, 3]), d=4)

    with raises(ValueError):
        HODMD(reconstruction_method=np.array([[1, 2, 3], [3, 4, 5]]), d=3)


def test_reconstruction_method_default_constructor():
    assert HODMD()._reconstruction_method == "first"


def test_reconstruction_method_constructor():
    assert HODMD(reconstruction_method="mean")._reconstruction_method == "mean"
    assert HODMD(reconstruction_method=[3])._reconstruction_method == [3]
    assert all(
        HODMD(
            reconstruction_method=np.array([1, 2]), d=2
        )._reconstruction_method
        == np.array([1, 2])
    )


def test_nonan_nomask():
    dmd = HODMD(d=3)
    dmd.fit(X=sample_data)
    rec = dmd.reconstructed_data

    assert not isinstance(rec, np.ma.MaskedArray)
    assert not np.nan in rec


def test_extract_versions_nonan():
    dmd = HODMD(d=3)
    dmd.fit(X=sample_data)
    for timeindex in range(sample_data.shape[1]):
        assert not np.nan in dmd.reconstructions_of_timeindex(timeindex)


def test_rec_method_first():
    dmd = HODMD(d=3, reconstruction_method="first")
    dmd.fit(X=sample_data)

    rec = dmd.reconstructed_data
    allrec = dmd.reconstructions_of_timeindex()
    for i in range(rec.shape[1]):
        assert (rec[:, i] == allrec[i, min(i, dmd.d - 1)]).all()


def test_rec_method_mean():
    dmd = HODMD(d=3, reconstruction_method="mean")
    dmd.fit(X=sample_data)
    assert (
        dmd.reconstructed_data.T[2]
        == np.mean(dmd.reconstructions_of_timeindex(2), axis=0).T
    ).all()


def test_rec_method_weighted():
    dmd = HODMD(d=2, svd_rank_extra=-1, reconstruction_method=[10, 20])
    dmd.fit(X=sample_data)
    assert (
        dmd.reconstructed_data.T[4]
        == np.average(
            dmd.reconstructions_of_timeindex(4), axis=0, weights=[10, 20]
        ).T
    ).all()


def test_scalar_func():
    x = np.linspace(0, 10, 64)[None]
    arr = np.cos(x) * np.sin(np.cos(x)) + np.cos(x * 0.2)
    # we check that this does not fail
    dmd = HODMD(svd_rank=1, exact=True, opt=True, d=3).fit(arr)


def test_get_bitmask_default():
    dmd = HODMD(svd_rank=-1, d=5)
    dmd.fit(X=sample_data)
    assert np.all(dmd.modes_activation_bitmask == True)


def test_set_bitmask():
    dmd = HODMD(svd_rank=-1, d=5)
    dmd.fit(X=sample_data)

    new_bitmask = np.full(len(dmd.amplitudes), True, dtype=bool)
    new_bitmask[[0]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.modes_activation_bitmask[0] == False
    assert np.all(dmd.modes_activation_bitmask[1:] == True)


def test_not_fitted_get_bitmask_raises():
    dmd = HODMD(svd_rank=-1, d=5)
    with raises(RuntimeError):
        print(dmd.modes_activation_bitmask)


def test_not_fitted_set_bitmask_raises():
    dmd = HODMD(svd_rank=-1, d=5)
    with raises(RuntimeError):
        dmd.modes_activation_bitmask = np.full(3, True, dtype=bool)


def test_raise_wrong_dtype_bitmask():
    dmd = HODMD(svd_rank=-1, d=5)
    dmd.fit(X=sample_data)
    with raises(RuntimeError):
        dmd.modes_activation_bitmask = np.full(3, 0.1)


def test_fitted():
    dmd = HODMD(svd_rank=-1, d=5)
    assert not dmd.fitted
    dmd.fit(X=sample_data)
    assert dmd.fitted


def test_bitmask_amplitudes():
    dmd = HODMD(svd_rank=-1, d=5)
    dmd.fit(X=sample_data)

    old_n_amplitudes = dmd.amplitudes.shape[0]
    retained_amplitudes = np.delete(dmd.amplitudes, [0, -1])

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0, -1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.amplitudes.shape[0] == old_n_amplitudes - 2
    np.testing.assert_almost_equal(dmd.amplitudes, retained_amplitudes)


def test_bitmask_eigs():
    dmd = HODMD(svd_rank=-1, d=5)
    dmd.fit(X=sample_data)

    old_n_eigs = dmd.eigs.shape[0]
    retained_eigs = np.delete(dmd.eigs, [0, -1])

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0, -1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.eigs.shape[0] == old_n_eigs - 2
    np.testing.assert_almost_equal(dmd.eigs, retained_eigs)


def test_bitmask_modes():
    dmd = HODMD(svd_rank=-1, d=5)
    dmd.fit(X=sample_data)

    old_n_modes = dmd.modes.shape[1]
    retained_modes = np.delete(dmd.modes, [0, -1], axis=1)

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0, -1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.modes.shape[1] == old_n_modes - 2
    np.testing.assert_almost_equal(dmd.modes, retained_modes)


def test_reconstructed_data_with_bitmask():
    dmd = HODMD(svd_rank=-1, d=5)
    dmd.fit(X=sample_data)

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0, -1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    dmd.reconstructed_data
    assert True


def test_getitem_modes():
    dmd = HODMD(svd_rank=-1, d=5)
    dmd.fit(X=sample_data)
    old_n_modes = dmd.modes.shape[1]

    assert dmd[[0, -1]].modes.shape[1] == 2
    np.testing.assert_almost_equal(dmd[[0, -1]].modes, dmd.modes[:, [0, -1]])

    assert dmd.modes.shape[1] == old_n_modes

    assert dmd[1::2].modes.shape[1] == old_n_modes // 2
    np.testing.assert_almost_equal(dmd[1::2].modes, dmd.modes[:, 1::2])

    assert dmd.modes.shape[1] == old_n_modes

    assert dmd[[1, 3]].modes.shape[1] == 2
    np.testing.assert_almost_equal(dmd[[1, 3]].modes, dmd.modes[:, [1, 3]])

    assert dmd.modes.shape[1] == old_n_modes

    assert dmd[2].modes.shape[1] == 1
    np.testing.assert_almost_equal(np.squeeze(dmd[2].modes), dmd.modes[:, 2])

    assert dmd.modes.shape[1] == old_n_modes


def test_getitem_raises():
    dmd = HODMD(svd_rank=-1, d=5)
    dmd.fit(X=sample_data)

    with raises(ValueError):
        dmd[[0, 1, 1, 0, 1]]
    with raises(ValueError):
        dmd[[True, True, False, True]]
    with raises(ValueError):
        dmd[1.0]


def test_correct_amplitudes():
    dmd = HODMD(svd_rank=-1, d=5)
    dmd.fit(X=sample_data)
    np.testing.assert_array_almost_equal(dmd.amplitudes, dmd._sub_dmd._b)


def test_raises_not_enough_snapshots():
    dmd = HODMD(svd_rank=-1, d=5)
    with raises(
        ValueError,
        match="The number of snapshots provided is not enough for d=5.\nExpected at least d.",
    ):
        dmd.fit(np.ones((20, 4)))
    with raises(ValueError, match="Received only one time snapshot."):
        dmd.fit(np.ones((20, 5)))
    dmd.fit(np.ones((20, 6)))
