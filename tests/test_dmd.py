from builtins import range

import numpy as np
from pytest import raises

from pydmd.dmd import DMD

# 15 snapshot with 400 data. The matrix is 400x15 and it contains
# the following data: f1 + f2 where
# f1 = lambda x,t: sech(x+3)*(1.*np.exp(1j*2.3*t))
# f2 = lambda x,t: (sech(x)*np.tanh(x))*(2.*np.exp(1j*2.8*t))
sample_data = np.load('tests/test_datasets/input_sample.npy')


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


noisy_data = create_noisy_data()


def test_shape():
    dmd = DMD(svd_rank=-1)
    dmd.fit(X=sample_data)
    assert dmd.modes.shape[1] == sample_data.shape[1] - 1

def test_truncation_shape():
    dmd = DMD(svd_rank=3)
    dmd.fit(X=sample_data)
    assert dmd.modes.shape[1] == 3

def test_rank():
    dmd = DMD(svd_rank=0.9)
    dmd.fit(X=sample_data)
    assert len(dmd.eigs) == 2

def test_Atilde_shape():
    dmd = DMD(svd_rank=3)
    dmd.fit(X=sample_data)
    assert dmd.atilde.shape == (dmd.svd_rank, dmd.svd_rank)

def test_Atilde_values():
    dmd = DMD(svd_rank=2)
    dmd.fit(X=sample_data)
    exact_atilde = np.array(
        [[-0.70558526 + 0.67815084j, 0.22914898 + 0.20020143j],
            [0.10459069 + 0.09137814j, -0.57730040 + 0.79022994j]])
    np.testing.assert_allclose(exact_atilde, dmd.atilde)

def test_eigs_1():
    dmd = DMD(svd_rank=-1)
    dmd.fit(X=sample_data)
    assert len(dmd.eigs) == 14

def test_eigs_2():
    dmd = DMD(svd_rank=5)
    dmd.fit(X=sample_data)
    assert len(dmd.eigs) == 5

def test_eigs_3():
    dmd = DMD(svd_rank=2)
    dmd.fit(X=sample_data)
    expected_eigs = np.array([
        -8.09016994e-01 + 5.87785252e-01j, -4.73868662e-01 + 8.80595532e-01j
    ])
    np.testing.assert_almost_equal(dmd.eigs, expected_eigs, decimal=6)

def test_dynamics_1():
    dmd = DMD(svd_rank=5)
    dmd.fit(X=sample_data)
    assert dmd.dynamics.shape == (5, sample_data.shape[1])

def test_dynamics_2():
    dmd = DMD(svd_rank=1)
    dmd.fit(X=sample_data)
    expected_dynamics = np.array([[
        -2.20639502 - 9.10168802e-16j, 1.55679980 - 1.49626864e+00j,
        -0.08375915 + 2.11149018e+00j, -1.37280962 - 1.54663768e+00j,
        2.01748787 + 1.60312745e-01j, -1.53222592 + 1.25504678e+00j,
        0.23000498 - 1.92462280e+00j, 1.14289644 + 1.51396355e+00j,
        -1.83310653 - 2.93174173e-01j, 1.49222925 - 1.03626336e+00j,
        -0.35015209 + 1.74312867e+00j, -0.93504202 - 1.46738182e+00j,
        1.65485808 + 4.01263449e-01j, -1.43976061 + 8.39117825e-01j,
        0.44682540 - 1.56844403e+00j
    ]])
    np.testing.assert_allclose(dmd.dynamics, expected_dynamics)

def test_dynamics_opt_1():
    dmd = DMD(svd_rank=5, opt=True)
    dmd.fit(X=sample_data)
    assert dmd.dynamics.shape == (5, sample_data.shape[1])

def test_dynamics_opt_2():
    dmd = DMD(svd_rank=1, opt=True)
    dmd.fit(X=sample_data)
    expected_dynamics = np.array([[
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
    ]])
    np.testing.assert_allclose(dmd.dynamics, expected_dynamics)

def test_reconstructed_data():
    dmd = DMD()
    dmd.fit(X=sample_data)
    dmd_data = dmd.reconstructed_data
    np.testing.assert_allclose(dmd_data, sample_data)

def test_original_time():
    dmd = DMD(svd_rank=2)
    dmd.fit(X=sample_data)
    expected_dict = {'dt': 1, 't0': 0, 'tend': 14}
    np.testing.assert_equal(dmd.original_time, expected_dict)

def test_original_timesteps():
    dmd = DMD()
    dmd.fit(X=sample_data)
    np.testing.assert_allclose(dmd.original_timesteps,
                                np.arange(sample_data.shape[1]))

def test_dmd_time_1():
    dmd = DMD(svd_rank=2)
    dmd.fit(X=sample_data)
    expected_dict = {'dt': 1, 't0': 0, 'tend': 14}
    np.testing.assert_equal(dmd.dmd_time, expected_dict)

def test_dmd_time_2():
    dmd = DMD()
    dmd.fit(X=sample_data)
    dmd.dmd_time['t0'] = 10
    dmd.dmd_time['tend'] = 14
    expected_data = sample_data[:, -5:]
    np.testing.assert_allclose(dmd.reconstructed_data, expected_data)

def test_dmd_time_3():
    dmd = DMD()
    dmd.fit(X=sample_data)
    dmd.dmd_time['t0'] = 8
    dmd.dmd_time['tend'] = 11
    expected_data = sample_data[:, 8:12]
    np.testing.assert_allclose(dmd.reconstructed_data, expected_data)

def test_dmd_time_4():
    dmd = DMD(svd_rank=3)
    dmd.fit(X=sample_data)
    dmd.dmd_time['t0'] = 20
    dmd.dmd_time['tend'] = 20
    expected_data = np.array([[-7.29383297e+00 - 4.90248179e-14j],
                                [-5.69109796e+00 - 2.74068833e+00j],
                                [3.38410649e-83 + 3.75677740e-83j]])
    np.testing.assert_almost_equal(dmd.dynamics, expected_data, decimal=6)

# we check that modes are the same vector multiplied by a coefficient
# when we rescale
def test_rescale_mode_auto_same_modes():
    dmd_no_rescale = DMD(svd_rank=2, opt=True, rescale_mode=None)
    dmd_no_rescale.fit(X=sample_data)

    dmd_auto_rescale = DMD(svd_rank=2, opt=True, rescale_mode='auto')
    dmd_auto_rescale.fit(X=sample_data)

    def normalize(vector):
        return vector / np.linalg.norm(vector)

    dmd_rescale_normalized_modes = np.apply_along_axis(normalize, 0,
        dmd_auto_rescale.modes)
    dmd_no_rescale_normalized_modes = np.apply_along_axis(normalize, 0,
        dmd_no_rescale.modes)

    np.testing.assert_almost_equal(dmd_no_rescale_normalized_modes,
        dmd_rescale_normalized_modes, decimal=3)

# we check that modes are the same vector multiplied by a coefficient
# when we rescale
def test_rescale_mode_custom_same_modes():
    dmd_no_rescale = DMD(svd_rank=2, opt=True, rescale_mode=None)
    dmd_no_rescale.fit(X=sample_data)

    dmd_rescale = DMD(svd_rank=2, opt=True, rescale_mode=
        np.linspace(5,10, 2))
    dmd_rescale.fit(X=sample_data)

    def normalize(vector):
        return vector / np.linalg.norm(vector)

    dmd_rescale_normalized_modes = np.apply_along_axis(normalize, 0,
        dmd_rescale.modes)
    dmd_no_rescale_normalized_modes = np.apply_along_axis(normalize, 0,
        dmd_no_rescale.modes)

    np.testing.assert_almost_equal(dmd_no_rescale_normalized_modes,
        dmd_rescale_normalized_modes, decimal=3)

def test_rescale_mode_same_evolution():
    dmd_no_rescale = DMD(svd_rank=5, opt=True, exact=True,
                            rescale_mode=None)
    dmd_no_rescale.fit(X=sample_data)
    dmd_no_rescale.dmd_time['tend'] *= 2

    dmd_rescale = DMD(svd_rank=5, opt=True, exact=True, rescale_mode=
        np.linspace(5,10, 5))
    dmd_rescale.fit(X=sample_data)
    dmd_rescale.dmd_time['tend'] *= 2

    np.testing.assert_almost_equal(dmd_rescale.reconstructed_data,
        dmd_no_rescale.reconstructed_data, decimal=6)

def test_rescale_mode_coefficients_count_check():
    dmd_rescale = DMD(svd_rank=5, opt=True, rescale_mode=
        np.linspace(5,10, 6))
    with raises(ValueError):
        dmd_rescale.fit(X=sample_data)

def test_predict():
    def f1(x,t):
        return 1./np.cosh(x+3)*np.exp(2.3j*t)

    def f2(x,t):
        return 2./np.cosh(x)*np.tanh(x)*np.exp(2.8j*t)

    x = np.linspace(-2, 2, 4)
    t = np.linspace(0, 4*np.pi, 10)

    xgrid, tgrid = np.meshgrid(x, t)

    X1 = f1(xgrid, tgrid)
    X2 = f2(xgrid, tgrid)
    X = X1 + X2

    dmd = DMD()
    dmd.fit(X.T)

    expected = np.array([
        [ 0.35407111+0.31966903j,  0.0581077 -0.51616519j,
            -0.4936891 +0.36476117j,  0.70397844+0.05332291j,
            -0.56648961-0.50687223j,  0.15372065+0.74444603j,
            0.30751808-0.63550106j, -0.5633934 +0.24365451j,
            0.47550633+0.20903766j, -0.0985528 -0.46673545j],
        [ 0.52924739+0.47782492j,  0.08685642-0.77153733j,
            -0.73794122+0.54522635j,  1.05227097+0.07970435j,
            -0.8467597 -0.7576467j ,  0.22977376+1.11275987j,
            0.4596623 -0.94991449j, -0.84213164+0.3642023j ,
            0.71076254+0.3124588j , -0.14731169-0.69765229j],
        [-0.49897731-0.45049592j, -0.0818887 +0.72740958j,
            0.69573498-0.51404236j, -0.99208678-0.0751457j ,
            0.79832963+0.71431342j, -0.21663195-1.04911604j,
            -0.43337211+0.89558454j,  0.79396628-0.3433719j ,
            -0.67011078-0.29458785j,  0.13888626+0.65775036j],
        [-0.2717424 -0.2453395j , -0.04459648+0.39614632j,
            0.37889637-0.2799468j , -0.54028918-0.04092425j,
            0.43476929+0.38901417j, -0.11797748-0.57134724j,
            -0.23601389+0.48773418j,  0.43239301-0.18699989j,
            - 0.36494147 - 0.16043216j, 0.07563728 + 0.35821j]
    ])

    np.testing.assert_almost_equal(dmd.predict(X.T), expected, decimal=6)

def test_predict_exact():
    dmd = DMD(exact=True)
    expected = np.load('tests/test_datasets/input_sample_predict_exact.npy')

    np.testing.assert_almost_equal(dmd
        .fit(sample_data)
        .predict(sample_data[:,20:40]), expected, decimal=6)

def test_predict_nexact():
    dmd = DMD(exact=False)
    expected = np.load('tests/test_datasets/input_sample_predict_nexact.npy')

    np.testing.assert_almost_equal(dmd
        .fit(sample_data)
        .predict(sample_data[:, 10:30]), expected, decimal=6)


def test_advanced_snapshot_parameter():
    dmd = DMD(svd_rank=0.99)
    dmd.fit(sample_data)

    dmd2 = DMD(svd_rank=0.99, opt=-1)
    dmd2.fit(sample_data)

    np.testing.assert_almost_equal(dmd2.reconstructed_data.real,
        dmd.reconstructed_data.real, decimal=6)


def test_sorted_eigs_default():
    dmd = DMD()
    assert dmd.operator._sorted_eigs == False

def test_sorted_eigs_set_real():
    dmd = DMD(sorted_eigs='real')
    assert dmd.operator._sorted_eigs == 'real'

def test_sorted_eigs_abs_right_eigs():
    dmd = DMD(svd_rank=20, sorted_eigs='abs')
    dmd.fit(sample_data)

    dmd2 = DMD(svd_rank=20)
    dmd2.fit(sample_data)

    assert len(dmd.eigs) == len(dmd2.eigs)
    assert set(dmd.eigs) == set(dmd2.eigs)

    previous = dmd.eigs[0]
    for eig in dmd.eigs[1:]:
        assert abs(previous) <= abs(eig)
        previous = eig

def test_sorted_eigs_abs_right_eigenvectors():
    dmd = DMD(svd_rank=20, sorted_eigs='abs')
    dmd.fit(sample_data)

    dmd2 = DMD(svd_rank=20)
    dmd2.fit(sample_data)

    for idx, eig in enumerate(dmd2.eigs):
        eigenvector = dmd2.operator.eigenvectors.T[idx]
        for idx_new, eig_new in enumerate(dmd.eigs):
            if eig_new == eig:
                assert all(dmd.operator.eigenvectors.T[idx_new] == eigenvector)
                break

def test_sorted_eigs_abs_right_modes():
    dmd = DMD(svd_rank=20, sorted_eigs='abs')
    dmd.fit(sample_data)

    dmd2 = DMD(svd_rank=20)
    dmd2.fit(sample_data)

    for idx, eig in enumerate(dmd2.eigs):
        mode = dmd2.modes.T[idx]
        for idx_new, eig_new in enumerate(dmd.eigs):
            if eig_new == eig:
                np.testing.assert_almost_equal(dmd.modes.T[idx_new], mode,
                    decimal=6)
                break

def test_sorted_eigs_real_right_eigs():
    dmd = DMD(svd_rank=20, sorted_eigs='real')
    dmd.fit(sample_data)

    dmd2 = DMD(svd_rank=20)
    dmd2.fit(sample_data)

    assert len(dmd.eigs) == len(dmd2.eigs)
    assert set(dmd.eigs) == set(dmd2.eigs)

    previous = complex(dmd.eigs[0])
    for eig in dmd.eigs[1:]:
        x = complex(eig)
        assert x.real > previous.real or (x.real == previous.real and x.imag >= previous.imag)
        previous = x

def test_sorted_eigs_real_right_eigenvectors():
    dmd = DMD(svd_rank=20, sorted_eigs='real')
    dmd.fit(sample_data)

    dmd2 = DMD(svd_rank=20)
    dmd2.fit(sample_data)

    for idx, eig in enumerate(dmd2.eigs):
        eigenvector = dmd2.operator.eigenvectors.T[idx]
        for idx_new, eig_new in enumerate(dmd.eigs):
            if eig_new == eig:
                assert all(dmd.operator.eigenvectors.T[idx_new] == eigenvector)
                break

def test_sorted_eigs_real_right_modes():
    dmd = DMD(svd_rank=20, sorted_eigs='real')
    dmd.fit(sample_data)

    dmd2 = DMD(svd_rank=20)
    dmd2.fit(sample_data)

    for idx, eig in enumerate(dmd2.eigs):
        mode = dmd2.modes.T[idx]
        for idx_new, eig_new in enumerate(dmd.eigs):
            if eig_new == eig:
                np.testing.assert_almost_equal(dmd.modes.T[idx_new], mode,
                    decimal=6)
                break

def test_sorted_eigs_dynamics():
    dmd = DMD(svd_rank=20, sorted_eigs='abs')
    dmd.fit(sample_data)

    dmd2 = DMD(svd_rank=20)
    dmd2.fit(sample_data)

    for idx, eig in enumerate(dmd2.eigs):
        dynamic = dmd2.dynamics[idx]
        for idx_new, eig_new in enumerate(dmd.eigs):
            if eig_new == eig:
                np.testing.assert_almost_equal(dmd.dynamics[idx_new],
                    dynamic, decimal=6)
                break

def test_sorted_eigs_frequency():
    dmd = DMD(svd_rank=20, sorted_eigs='abs')
    dmd.fit(sample_data)

    dmd2 = DMD(svd_rank=20)
    dmd2.fit(sample_data)

    for idx, eig in enumerate(dmd2.eigs):
        frq = dmd2.frequency[idx]
        for idx_new, eig_new in enumerate(dmd.eigs):
            if eig_new == eig:
                np.testing.assert_almost_equal(dmd.frequency[idx_new],
                    frq, decimal=6)
                break

def test_sorted_eigs_amplitudes():
    dmd = DMD(svd_rank=20, sorted_eigs='abs')
    dmd.fit(sample_data)

    dmd2 = DMD(svd_rank=20)
    dmd2.fit(sample_data)

    for idx, eig in enumerate(dmd2.eigs):
        amp = dmd2.amplitudes[idx]
        for idx_new, eig_new in enumerate(dmd.eigs):
            if eig_new == eig:
                np.testing.assert_almost_equal(dmd.amplitudes[idx_new],
                    amp, decimal=6)
                break

def test_save():
    dmd = DMD(svd_rank=-1)
    dmd.fit(X=sample_data)
    dmd.save('pydmd.test')

def test_load():
    dmd = DMD(svd_rank=-1)
    dmd.fit(X=sample_data)
    dmd.save('pydmd.test2')
    loaded_dmd = DMD.load('pydmd.test2')
    np.testing.assert_array_equal(dmd.reconstructed_data,
                                    loaded_dmd.reconstructed_data)

def test_load():
    dmd = DMD(svd_rank=-1)
    dmd.fit(X=sample_data)
    dmd.save('pydmd.test2')
    loaded_dmd = DMD.load('pydmd.test2')
    assert isinstance(loaded_dmd, DMD)

def test_get_bitmask_default():
    dmd = DMD(svd_rank=10)
    dmd.fit(X=sample_data)
    assert np.all(dmd.modes_activation_bitmask == True)

def test_set_bitmask():
    dmd = DMD(svd_rank=3)
    dmd.fit(X=sample_data)

    new_bitmask = np.full(len(dmd.amplitudes), True, dtype=bool)
    new_bitmask[[0]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.modes_activation_bitmask[0] == False
    assert np.all(dmd.modes_activation_bitmask[1:] == True)

def test_not_fitted_get_bitmask_raises():
    dmd = DMD(svd_rank=3)
    with raises(RuntimeError):
        print(dmd.modes_activation_bitmask)

def test_not_fitted_set_bitmask_raises():
    dmd = DMD(svd_rank=3)
    with raises(RuntimeError):
        dmd.modes_activation_bitmask = np.full(3, True, dtype=bool)

def test_raise_wrong_dtype_bitmask():
    dmd = DMD(svd_rank=3)
    dmd.fit(X=sample_data)
    with raises(RuntimeError):
        dmd.modes_activation_bitmask = np.full(3, 0.1)

def test_fitted():
    dmd = DMD(svd_rank=3)
    assert not dmd.fitted
    dmd.fit(X=sample_data)
    assert dmd.fitted

def test_bitmask_amplitudes():
    dmd = DMD(svd_rank=10)
    dmd.fit(X=sample_data)

    old_n_amplitudes = dmd.amplitudes.shape[0]
    retained_amplitudes = np.delete(dmd.amplitudes, [0,-1])

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0,-1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.amplitudes.shape[0] == old_n_amplitudes - 2
    np.testing.assert_almost_equal(dmd.amplitudes, retained_amplitudes)

def test_bitmask_eigs():
    dmd = DMD(svd_rank=10)
    dmd.fit(X=sample_data)

    old_n_eigs = dmd.eigs.shape[0]
    retained_eigs = np.delete(dmd.eigs, [0,-1])

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0,-1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.eigs.shape[0] == old_n_eigs - 2
    np.testing.assert_almost_equal(dmd.eigs, retained_eigs)

def test_bitmask_modes():
    dmd = DMD(svd_rank=10)
    dmd.fit(X=sample_data)

    old_n_modes = dmd.modes.shape[1]
    retained_modes = np.delete(dmd.modes, [0,-1], axis=1)

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0,-1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    assert dmd.modes.shape[1] == old_n_modes - 2
    np.testing.assert_almost_equal(dmd.modes, retained_modes)

def test_reconstructed_data():
    dmd = DMD(svd_rank=10)
    dmd.fit(X=sample_data)

    new_bitmask = np.full(dmd.amplitudes.shape[0], True, dtype=bool)
    new_bitmask[[0,-1]] = False
    dmd.modes_activation_bitmask = new_bitmask

    dmd.reconstructed_data
    assert True

def test_getitem_modes():
    dmd = DMD(svd_rank=-1)
    dmd.fit(X=sample_data)
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
    dmd = DMD(svd_rank=-1)
    dmd.fit(X=sample_data)

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
    dmd = DMD(svd_rank=-1)
    dmd.fit(X=sample_data)
    np.testing.assert_array_almost_equal(dmd.amplitudes, dmd._b)
