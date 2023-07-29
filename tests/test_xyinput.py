import numpy as np
from pytest import raises
from scipy.integrate import solve_ivp

from pydmd.dmd import DMD
from pydmd.bopdmd import BOPDMD


def simulate_z(t):
    """
    Given a time vector t = t1, t2, ..., evaluates and returns the snapshots
    z(t1), z(t2), ... as columns of the matrix Z via explicit Runge-Kutta.
    Simulates data z given by the system of ODEs
        z' = Az
    where A = [1 -2; 1 -1] and z_0 = [1, 0.1].
    """

    def ode_sys(t, z):
        z1, z2 = z
        return [z1 - 2 * z2, z1 - z2]

    sol = solve_ivp(ode_sys, [t[0], t[-1]], [1.0, 0.1], t_eval=t)

    return sol.y


def differentiate(X, dt):
    """
    Method for performing 2nd order centered finite difference. Assumes the
    input matrix X is 2-D, with uniformly-sampled snapshots filling each
    column. Requires dt, which is the time step between each snapshot.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("Please ensure that input data is a 2D array.")
    X_prime = np.empty(X.shape)
    X_prime[:, 1:-1] = (X[:, 2:] - X[:, :-2]) / (2 * dt)
    X_prime[:, 0] = (-3 * X[:, 0] + 4 * X[:, 1] - X[:, 2]) / (2 * dt)
    X_prime[:, -1] = (3 * X[:, -1] - 4 * X[:, -2] + X[:, -3]) / (2 * dt)

    return X_prime


# Simulate ODE system data.
dt = 0.01
t = np.arange(2000) * dt
Z = simulate_z(t)
Z_dot = differentiate(Z, dt)

# Additionally re-obtain DMD testing data.
# 15 snapshot with 400 data. The matrix is 400x15 and it contains
# the following data: f1 + f2 where
# f1 = lambda x,t: sech(x+3)*(1.*np.exp(1j*2.3*t))
# f2 = lambda x,t: (sech(x)*np.tanh(x))*(2.*np.exp(1j*2.8*t))
sample_data = np.load("tests/test_datasets/input_sample.npy")
sample_data_1 = sample_data[:, :-1]
sample_data_2 = sample_data[:, 1:]


def assert_equal_models(dmd, dmd_true, rtol=1e-07, atol=0):
    """
    Helper method for comparing two DMD models. Ensures that the eigs, modes,
    amplitudes, and Atilde operators agree up to the given tolerance values.
    """
    # Compare eigenvalues.
    np.testing.assert_allclose(dmd.eigs, dmd_true.eigs, rtol=rtol, atol=atol)
    # Compare modes.
    np.testing.assert_allclose(dmd.modes, dmd_true.modes, rtol=rtol, atol=atol)
    # Compare amplitudes.
    np.testing.assert_allclose(
        dmd.amplitudes, dmd_true.amplitudes, rtol=rtol, atol=atol
    )
    # Compare Atilde matrices.
    np.testing.assert_allclose(
        dmd.operator.as_numpy_array,
        dmd_true.operator.as_numpy_array,
        rtol=rtol,
        atol=atol,
    )


def test_shape_error_1():
    # Checks that an error is thrown when X and Y
    # don't contain the same number of snapshots.
    dmd = DMD()
    with raises(ValueError):
        dmd.fit(X=sample_data, Y=sample_data[:, 1:])


def test_shape_error_2():
    # Checks that an error is thrown when the
    # snapshots in X and Y aren't the same size.
    dmd = DMD()
    with raises(ValueError):
        dmd.fit(X=sample_data, Y=sample_data[1:])


def test_time_dicts():
    # Checks that the default time dictionaries contain 0, 1, ..., m-1
    # when data matrices X, Y containing m snapshots each are given.
    dmd = DMD()
    dmd.fit(X=sample_data_1, Y=sample_data_2)
    expected_dict = {"dt": 1, "t0": 0, "tend": 13}
    np.testing.assert_equal(dmd.original_time, expected_dict)
    np.testing.assert_equal(dmd.dmd_time, expected_dict)


def test_equal_models_default():
    # Checks that a DMD model given X=data, Y=None is qualitatively the same as
    # a DMD model given X=data[:, :-1], Y=data[:, 1:] using default parameters.
    dmd = DMD(svd_rank=2)
    dmd.fit(X=sample_data)

    dmd_xy = DMD(svd_rank=2)
    dmd_xy.fit(X=sample_data_1, Y=sample_data_2)

    assert_equal_models(dmd_xy, dmd)


def test_equal_models_exact():
    # Checks that a DMD model given X=data, Y=None is qualitatively the same as
    # a DMD model given X=data[:, :-1], Y=data[:, 1:] using exact=True.
    dmd = DMD(svd_rank=2, exact=True)
    dmd.fit(X=sample_data)

    dmd_xy = DMD(svd_rank=2, exact=True)
    dmd_xy.fit(X=sample_data_1, Y=sample_data_2)

    assert_equal_models(dmd_xy, dmd)


def test_equal_models_opt():
    # Checks that a DMD model given X=data, Y=None is almost qualitatively the
    # same as a DMD model given X=data[:, :-1], Y=data[:, 1:] using opt=True.
    dmd = DMD(svd_rank=2, opt=True)
    dmd.fit(X=sample_data)

    dmd_xy = DMD(svd_rank=2, opt=True)
    dmd_xy.fit(X=sample_data_1, Y=sample_data_2)

    assert_equal_models(dmd_xy, dmd, rtol=0.05)


def test_equal_models_opt_exact():
    # Checks that a DMD model given X=data, Y=None is qualitatively the same as
    # a model given X=data[:, :-1], Y=data[:, 1:] using opt=True, exact=True.
    dmd = DMD(svd_rank=2, opt=True, exact=True)
    dmd.fit(X=sample_data)

    dmd_xy = DMD(svd_rank=2, opt=True, exact=True)
    dmd_xy.fit(X=sample_data_1, Y=sample_data_2)

    assert_equal_models(dmd_xy, dmd)


def test_time_shifted_model():
    # Checks that a DMD model fitted with X=data, Y=shifted(data) and missing
    # data is able to recover the diagnostics obtained using all of the data.
    uneven_indices = np.delete(np.arange(len(t)), np.arange(1000)[1::2])
    uneven_indices = uneven_indices[:-1]

    Z_uneven_1 = Z[:, uneven_indices]
    Z_uneven_2 = Z[:, uneven_indices + 1]

    dmd = DMD(svd_rank=2)
    dmd.fit(X=Z)

    dmd_xy = DMD(svd_rank=2)
    dmd_xy.fit(X=Z_uneven_1, Y=Z_uneven_2)

    assert_equal_models(dmd_xy, dmd, rtol=0.01)


def test_time_derivative_model():
    # Checks that a DMD model given X=data, Y=derivative(data) yields
    # the same operator as a BOPDMD model given the same data set.
    dmd = DMD(svd_rank=2)
    dmd.fit(X=Z, Y=Z_dot)

    bopdmd = BOPDMD(svd_rank=2)
    bopdmd.fit(Z, t)

    np.testing.assert_allclose(
        dmd.operator.as_numpy_array,
        bopdmd.operator.as_numpy_array,
        rtol=0.01,
    )
