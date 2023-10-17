from builtins import range

import numpy as np
from pytest import raises

from pydmd import HankelDMD

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
    dmd = HankelDMD(svd_rank=-1)
    dmd.fit(X=sample_data)
    assert dmd.modes.shape[1] == sample_data.shape[1] - 1


def test_truncation_shape():
    dmd = HankelDMD(svd_rank=3)
    dmd.fit(X=sample_data)
    assert dmd.modes.shape[1] == 3


def test_rank():
    dmd = HankelDMD(svd_rank=0.9)
    dmd.fit(X=sample_data)
    assert len(dmd.eigs) == 2


def test_Atilde_shape():
    dmd = HankelDMD(svd_rank=3)
    dmd.fit(X=sample_data)
    assert dmd.operator.as_numpy_array.shape == (
        dmd.operator._svd_rank,
        dmd.operator._svd_rank,
    )


def test_d():
    single_data = np.sin(np.linspace(0, 10, 100))[None]
    dmd = HankelDMD(svd_rank=0, d=50, opt=True)
    dmd.fit(single_data)
    np.testing.assert_array_almost_equal(
        dmd.reconstructed_data.real, single_data, decimal=1
    )  # TODO poor accuracy using projected modes
    dmd = HankelDMD(svd_rank=-1, d=50, opt=True, exact=True)
    dmd.fit(single_data)
    assert np.allclose(dmd.reconstructed_data.flatten().real, single_data)


def test_Atilde_values():
    dmd = HankelDMD(svd_rank=2)
    dmd.fit(X=sample_data)
    exact_atilde = np.array(
        [
            [-0.70558526 + 0.67815084j, 0.22914898 + 0.20020143j],
            [0.10459069 + 0.09137814j, -0.57730040 + 0.79022994j],
        ]
    )
    np.testing.assert_allclose(exact_atilde, dmd.operator.as_numpy_array)


def test_eigs_1():
    dmd = HankelDMD(svd_rank=-1)
    dmd.fit(X=sample_data)
    assert len(dmd.eigs) == 14


def test_eigs_2():
    dmd = HankelDMD(svd_rank=5)
    dmd.fit(X=sample_data)
    assert len(dmd.eigs) == 5


def test_eigs_3():
    dmd = HankelDMD(svd_rank=2)
    dmd.fit(X=sample_data)
    expected_eigs = np.array(
        [
            -8.09016994e-01 + 5.87785252e-01j,
            -4.73868662e-01 + 8.80595532e-01j,
        ]
    )
    np.testing.assert_almost_equal(dmd.eigs, expected_eigs, decimal=6)


def test_dynamics_1():
    dmd = HankelDMD(svd_rank=5)
    dmd.fit(X=sample_data)
    assert dmd.dynamics.shape == (5, sample_data.shape[1])


def test_dynamics_2():
    dmd = HankelDMD(svd_rank=1)
    dmd.fit(X=sample_data)
    expected_dynamics = np.array(
        [
            [
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
            ]
        ]
    )
    np.testing.assert_allclose(dmd.dynamics, expected_dynamics)


def test_dynamics_opt_1():
    dmd = HankelDMD(svd_rank=5, opt=True)
    dmd.fit(X=sample_data)
    assert dmd.dynamics.shape == (5, sample_data.shape[1])


def test_dynamics_opt_2():
    dmd = HankelDMD(svd_rank=1, opt=True, exact=False)
    dmd.fit(X=sample_data)
    expected_dynamics = np.array(
        [
            [
                -4.609718826226513 - 6.344781724790875j,
                7.5552686987577165 + 1.3506997434096375j,
                -6.246864367654589 + 4.170577993207872j,
                1.5794144248628537 - 7.179014663490048j,
                3.754043295828462 + 6.13648812118528j,
                -6.810262177959786 - 1.7840079278093528j,
                6.015047060133346 - 3.35961532862783j,
                -1.9658025630719695 + 6.449604262000736j,
                -2.9867632454837936 - 5.8838563367460734j,
                6.097558230017521 + 2.126086276430128j,
                -5.7441543819530265 + 2.6349291080417103j,
                2.266111252852836 - 5.7545702519088895j,
                2.303531963541068 + 5.597105176945707j,
                -5.421019770795679 - 2.3870927539102658j,
                5.443800581850978 - 1.9919716610066682j,
            ]
        ]
    )
    np.testing.assert_allclose(dmd.dynamics, expected_dynamics)


def test_reconstructed_data():
    dmd = HankelDMD()
    dmd.fit(X=sample_data)
    dmd_data = dmd.reconstructed_data
    np.testing.assert_allclose(dmd_data, sample_data)


def test_original_time():
    dmd = HankelDMD(svd_rank=2)
    dmd.fit(X=sample_data)
    expected_dict = {"dt": 1, "t0": 0, "tend": 14}
    np.testing.assert_equal(dmd.original_time, expected_dict)


def test_original_timesteps():
    dmd = HankelDMD()
    dmd.fit(X=sample_data)
    np.testing.assert_allclose(
        dmd.original_timesteps, np.arange(sample_data.shape[1])
    )


def test_dmd_time_1():
    dmd = HankelDMD(svd_rank=2)
    dmd.fit(X=sample_data)
    expected_dict = {"dt": 1, "t0": 0, "tend": 14}
    np.testing.assert_equal(dmd.dmd_time, expected_dict)


def test_dmd_time_2():
    dmd = HankelDMD()
    dmd.fit(X=sample_data)
    dmd.dmd_time["t0"] = 10
    dmd.dmd_time["tend"] = 14
    expected_data = sample_data[:, -5:]
    np.testing.assert_allclose(dmd.reconstructed_data, expected_data)


def test_dmd_time_3():
    dmd = HankelDMD()
    dmd.fit(X=sample_data)
    dmd.dmd_time["t0"] = 8
    dmd.dmd_time["tend"] = 11
    expected_data = sample_data[:, 8:12]
    np.testing.assert_allclose(dmd.reconstructed_data, expected_data)


def test_dmd_time_4():
    dmd = HankelDMD(svd_rank=3)
    dmd.fit(X=sample_data)
    dmd.dmd_time["t0"] = 20
    dmd.dmd_time["tend"] = 20
    expected_data = np.array(
        [
            [-7.29383297e00 - 4.90248179e-14j],
            [-5.69109796e00 - 2.74068833e00j],
            [3.38410649e-83 + 3.75677740e-83j],
        ]
    )
    np.testing.assert_almost_equal(dmd.dynamics, expected_data, decimal=6)


def test_dmd_time_5():
    x = np.linspace(0, 10, 64)[None]
    y = np.cos(x) * np.sin(np.cos(x)) + np.cos(x * 0.2)

    dmd = HankelDMD(svd_rank=-1, exact=True, opt=True, d=30)
    dmd.fit(y)

    dmd.original_time["dt"] = dmd.dmd_time["dt"] = x[0, 1] - x[0, 0]
    dmd.original_time["t0"] = dmd.dmd_time["t0"] = x[0, 0]
    dmd.original_time["tend"] = dmd.dmd_time["tend"] = x[0, -1]

    assert dmd.reconstructed_data.shape == (1, 64)


def test_sorted_eigs_default():
    dmd = HankelDMD()
    assert dmd.operator._sorted_eigs == False


def test_sorted_eigs_param():
    dmd = HankelDMD(sorted_eigs="real")
    assert dmd.operator._sorted_eigs == "real"


def test_reconstruction_method_constructor_error():
    with raises(ValueError):
        HankelDMD(reconstruction_method=[1, 2, 3], d=4)

    with raises(ValueError):
        HankelDMD(reconstruction_method=np.array([1, 2, 3]), d=4)

    with raises(ValueError):
        HankelDMD(reconstruction_method=np.array([[1, 2, 3], [3, 4, 5]]), d=3)


def test_reconstruction_method_default_constructor():
    assert HankelDMD()._reconstruction_method == "first"


def test_nonan_nomask():
    dmd = HankelDMD(d=3)
    dmd.fit(X=sample_data)
    rec = dmd.reconstructed_data

    assert not isinstance(rec, np.ma.MaskedArray)
    assert not np.nan in rec


def test_hankeldmd_timesteps():
    x = np.linspace(0, 10, 64)[None]
    arr = np.cos(x) * np.sin(np.cos(x)) + np.cos(x * 0.2)
    dmd = HankelDMD(svd_rank=1, exact=True, opt=True, d=30).fit(arr)
    assert len(dmd.dmd_timesteps) == 64


def test_first_occurences():
    x = np.linspace(0, 10, 64)[None]
    arr = np.cos(x) * np.sin(np.cos(x)) + np.cos(x * 0.2)
    dmd = HankelDMD(svd_rank=1, exact=True, opt=True, d=3).fit(arr)
    assert dmd._hankel_first_occurrence(0) == 0
    assert dmd._hankel_first_occurrence(1) == 0
    assert dmd._hankel_first_occurrence(2) == 0
    assert dmd._hankel_first_occurrence(3) == 1
    assert dmd._hankel_first_occurrence(4) == 2
    assert dmd._hankel_first_occurrence(5) == 3

    dmd.dmd_time["tend"] = 100
    assert dmd._hankel_first_occurrence(100) == 98

    # change scale
    dmd.dmd_time["t0"] = dmd.original_time["t0"] = x[0, 0]
    dmd.dmd_time["tend"] = dmd.original_time["tend"] = x[0, -1]
    dmd.dmd_time["dt"] = dmd.original_time["dt"] = x[0, 1] - x[0, 0]

    assert dmd._hankel_first_occurrence(x[0, 0]) == 0
    assert dmd._hankel_first_occurrence(x[0, 1]) == 0
    assert dmd._hankel_first_occurrence(x[0, 2]) == 0
    assert dmd._hankel_first_occurrence(x[0, 3]) == 1
    assert dmd._hankel_first_occurrence(x[0, -1] + dmd.dmd_time["dt"]) == 62

    dmd.dmd_time["t0"] = x[0, x.shape[-1] // 2]
    dmd.dmd_time["tend"] = x[0, -1] + dmd.dmd_time["dt"] * 20

    assert (
        dmd._hankel_first_occurrence(dmd.dmd_time["t0"]) == x.shape[-1] // 2 - 2
    )


def test_update_sub_dmd_time():
    dmd = HankelDMD()
    x = np.linspace(0, 10, 64)[None]
    arr = np.cos(x) * np.sin(np.cos(x)) + np.cos(x * 0.2)
    dmd = HankelDMD(svd_rank=1, exact=True, opt=True, d=3).fit(arr)

    dmd.dmd_time["tend"] += dmd.dmd_time["dt"] * 20
    dmd._update_sub_dmd_time()

    # assert that the dt for the sub_dmd is always 1
    assert dmd._sub_dmd.dmd_time["dt"] == 1
    assert dmd._sub_dmd.original_time["dt"] == 1

    assert (
        dmd._sub_dmd.dmd_time["tend"] == dmd._sub_dmd.original_time["tend"] + 20
    )


def test_hankel_2d():
    def fnc(x):
        return np.cos(x) * np.sin(np.cos(x)) + np.cos(x * 0.2)

    x = np.linspace(0, 10, 64)
    snapshots = np.vstack([fnc(x), -fnc(x)])

    dmd = HankelDMD(svd_rank=0, exact=True, opt=True, d=30).fit(snapshots)

    dmd.original_time["dt"] = dmd.dmd_time["dt"] = x[1] - x[0]
    dmd.original_time["t0"] = dmd.dmd_time["t0"] = x[0]
    dmd.original_time["tend"] = dmd.dmd_time["tend"] = x[-1]

    dmd.dmd_time["t0"] = x[len(x) // 2]
    dmd.dmd_time["tend"] = x[-1] + dmd.dmd_time["dt"] * 20

    assert len(dmd.dmd_timesteps) == dmd.reconstructed_data.shape[1]

    np.testing.assert_allclose(
        dmd.reconstructed_data,
        np.vstack([fnc(dmd.dmd_timesteps), -fnc(dmd.dmd_timesteps)]),
    )


def test_get_bitmask_default():
    dmd = HankelDMD(svd_rank=-1, d=5)
    dmd.fit(X=sample_data)
    assert np.all(dmd.modes_activation_bitmask == True)


def test_set_bitmask():
    dmd = HankelDMD(svd_rank=-1, d=5)
    dmd.fit(X=sample_data)

    new_bitmask = np.full(len(dmd.amplitudes), True, dtype=bool)
    new_bitmask[[0]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.modes_activation_bitmask[0] == False
    assert np.all(dmd.modes_activation_bitmask[1:] == True)


def test_not_fitted_get_bitmask_raises():
    dmd = HankelDMD(svd_rank=-1, d=5)
    with raises(RuntimeError):
        print(dmd.modes_activation_bitmask)


def test_not_fitted_set_bitmask_raises():
    dmd = HankelDMD(svd_rank=-1, d=5)
    with raises(RuntimeError):
        dmd.modes_activation_bitmask = np.full(3, True, dtype=bool)


def test_raise_wrong_dtype_bitmask():
    dmd = HankelDMD(svd_rank=-1, d=5)
    dmd.fit(X=sample_data)
    with raises(RuntimeError):
        dmd.modes_activation_bitmask = np.full(3, 0.1)


def test_fitted():
    dmd = HankelDMD(svd_rank=-1, d=5)
    assert not dmd.fitted
    dmd.fit(X=sample_data)
    assert dmd.fitted


def test_bitmask_amplitudes():
    dmd = HankelDMD(svd_rank=-1, d=5)
    dmd.fit(X=sample_data)

    old_n_amplitudes = dmd.amplitudes.shape[0]
    retained_amplitudes = np.delete(dmd.amplitudes, [0, -1])

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0, -1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.amplitudes.shape[0] == old_n_amplitudes - 2
    np.testing.assert_almost_equal(dmd.amplitudes, retained_amplitudes)


def test_bitmask_eigs():
    dmd = HankelDMD(svd_rank=-1, d=5)
    dmd.fit(X=sample_data)

    old_n_eigs = dmd.eigs.shape[0]
    retained_eigs = np.delete(dmd.eigs, [0, -1])

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0, -1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.eigs.shape[0] == old_n_eigs - 2
    np.testing.assert_almost_equal(dmd.eigs, retained_eigs)


def test_bitmask_modes():
    dmd = HankelDMD(svd_rank=-1, d=5)
    dmd.fit(X=sample_data)

    old_n_modes = dmd.modes.shape[1]
    retained_modes = np.delete(dmd.modes, [0, -1], axis=1)

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0, -1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.modes.shape[1] == old_n_modes - 2
    np.testing.assert_almost_equal(dmd.modes, retained_modes)


def test_reconstructed_data_with_bitmask():
    dmd = HankelDMD(svd_rank=-1, d=5)
    dmd.fit(X=sample_data)

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0, -1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    dmd.reconstructed_data
    assert True


def test_correct_amplitudes():
    dmd = HankelDMD(svd_rank=-1, d=5)
    dmd.fit(X=sample_data)
    np.testing.assert_array_almost_equal(dmd.amplitudes, dmd._sub_dmd._b)


def test_raises_not_enough_snapshots():
    dmd = HankelDMD(svd_rank=-1, d=5)
    with raises(
        ValueError,
        match="The number of snapshots provided is not enough for d=5.\nExpected at least d.",
    ):
        dmd.fit(np.ones((20, 4)))
    with raises(ValueError, match="Received only one time snapshot."):
        dmd.fit(np.ones((20, 5)))
    dmd.fit(np.ones((20, 6)))


def test_hankeldmd_1d():
    dmd = HankelDMD(svd_rank=-1, d=5)
    data = np.pi * np.cos(np.linspace(0, 2, 100))[None]
    dmd.fit(data)
    np.testing.assert_allclose(data, dmd.reconstructed_data)
