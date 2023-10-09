import numpy as np
from scipy.integrate import solve_ivp
from numpy.testing import assert_allclose, assert_equal

from pydmd.lando import LANDO

# Chaotic Lorenz parameters:
sigma, rho, beta = 10, 28, 8 / 3

# Lorenz system fixed point:
x_bar = [-np.sqrt(beta * (rho - 1)), -np.sqrt(beta * (rho - 1)), rho - 1]

# Settings to replicate odeint defaults.
solve_ivp_opts = {}
solve_ivp_opts["rtol"] = 1e-12
solve_ivp_opts["atol"] = 1e-12
solve_ivp_opts["method"] = "LSODA"


def relative_error(mat, mat_true):
    """
    Computes and returns the relative error between two matrices.
    """
    return np.linalg.norm(mat - mat_true) / np.linalg.norm(mat_true)


def differentiate(X, dt):
    """
    Method for performing 2nd order finite difference. Assumes the input
    matrix X is 2-D, with uniformly-sampled snapshots filling each column.
    Requires dt, which is the time step between each snapshot.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("Please ensure that input data is a 2D array.")
    X_prime = np.empty(X.shape)
    X_prime[:, 1:-1] = (X[:, 2:] - X[:, :-2]) / (2 * dt)
    X_prime[:, 0] = (-3 * X[:, 0] + 4 * X[:, 1] - X[:, 2]) / (2 * dt)
    X_prime[:, -1] = (3 * X[:, -1] - 4 * X[:, -2] + X[:, -3]) / (2 * dt)

    return X_prime


def generate_lorenz_data(t_eval, x0=(-8, 8, 27)):
    """
    Given a time vector t_eval = t1, t2, ..., evaluates and
    returns the snapshots of the Lorenz system as columns of
    the matrix X for the initial condition x0.
    """

    def lorenz_system(t, state):
        x, y, z = state
        x_dot = sigma * (y - x)
        y_dot = (x * (rho - z)) - y
        z_dot = (x * y) - (beta * z)
        return [x_dot, y_dot, z_dot]

    sol = solve_ivp(
        lorenz_system,
        [t_eval[0], t_eval[-1]],
        x0,
        t_eval=t_eval,
        **solve_ivp_opts,
    )

    return sol.y


# Generate Lorenz system data.
dt = 0.001
t = np.arange(0, 10, dt)
X = generate_lorenz_data(t)
Y = differentiate(X, dt)

# Set the LANDO learning parameters used by most test models.
lando_params = {}
lando_params["svd_rank"] = 3
lando_params["kernel_metric"] = "poly"
lando_params["kernel_params"] = {"degree": 3, "coef0": 1.0, "gamma": 1.0}
lando_params["x_rescale"] = 1.0 / np.abs(X).max(axis=1)
lando_params["dict_tol"] = 1e-6


def test_fitted():
    """
    Test that the partially fitted and fitted attributes are updated correctly.
    """
    lando = LANDO(**lando_params)
    assert not lando.partially_fitted
    assert not lando.fitted

    lando.fit(X, Y)
    assert lando.partially_fitted
    assert not lando.fitted

    lando.analyze_fixed_point(x_bar)
    assert lando.partially_fitted
    assert lando.fitted


def test_shapes():
    """
    Test that the shapes of the sparse dictionary and the sparse dictionary
    weights are as expected.
    """
    lando = LANDO(**lando_params)
    lando.fit(X, Y)
    assert X.shape[0] == lando.sparse_dictionary.shape[0]
    assert X.shape[-1] > lando.sparse_dictionary.shape[-1]
    assert lando.operator.weights.shape == lando.sparse_dictionary.shape


def test_f():
    """
    Test that the computed function f() is accurate.
    """
    lando = LANDO(**lando_params)
    lando.fit(X, Y)
    assert relative_error(lando.f(X), Y) < 1e-5


def test_bias():
    """
    Test that the computed bias term is accurate.
    """
    lando = LANDO(**lando_params)
    lando.fit(X, Y)
    lando.analyze_fixed_point(x_bar)
    assert lando.bias.shape == (3, 1)
    assert np.linalg.norm(lando.bias) < 1e-3


def test_linear():
    """
    Test that the computed linear operator is accurate.
    """
    lando = LANDO(**lando_params)
    lando.fit(X, Y)
    lando.analyze_fixed_point(x_bar, compute_A=True)

    A_true = np.array(
        [
            [-sigma, sigma, 0.0],
            [1.0, -1.0, np.sqrt(beta * (rho - 1))],
            [-np.sqrt(beta * (rho - 1)), -np.sqrt(beta * (rho - 1)), -beta],
        ]
    )
    eigs_true, modes_true = np.linalg.eig(A_true)
    modes_rescale = np.divide(lando.modes[0], modes_true[0])
    modes_true_rescaled = np.multiply(modes_true, modes_rescale)

    assert relative_error(lando.linear, A_true) < 1e-4
    assert_allclose(lando.eigs, eigs_true, rtol=1e-4)
    assert_allclose(lando.modes, modes_true_rescaled, rtol=1e-3)


def test_nonlinear():
    """
    Test that the nonlinear operator has the correct shape.
    """
    lando = LANDO(**lando_params)
    lando.fit(X, Y)
    lando.analyze_fixed_point(x_bar)
    assert lando.nonlinear(X).shape == X.shape


def test_predict_1():
    """
    Test that predict() can be used to reconstruct the system.
    """
    lando = LANDO(**lando_params)
    lando.fit(X, Y)
    lando_recon = lando.predict(
        x0=(-8, 8, 27),
        tend=len(t),
        continuous=True,
        dt=dt,
        solve_ivp_opts=solve_ivp_opts,
    )
    assert relative_error(lando_recon, X) < 0.01


def test_predict_2():
    """
    Test that predict() can be used to predict the system.
    """
    t_long = np.arange(0, 12, dt)
    X_long = generate_lorenz_data(t_long)

    lando = LANDO(**lando_params)
    lando.fit(X, Y)

    lando_predict = lando.predict(
        x0=(-8, 8, 27),
        tend=len(t_long),
        continuous=True,
        dt=dt,
        solve_ivp_opts=solve_ivp_opts,
    )
    assert relative_error(lando_predict, X_long) < 0.05


def test_predict_3():
    """
    Test that predict() can accurately produce data for new initial conditions.
    """
    new_init = (10, 14, 10)
    X2 = generate_lorenz_data(t, x0=new_init)

    lando = LANDO(**lando_params)
    lando.fit(X, Y)

    lando_predict = lando.predict(
        x0=new_init,
        tend=len(t),
        continuous=True,
        dt=dt,
        solve_ivp_opts=solve_ivp_opts,
    )
    assert relative_error(lando_predict, X2) < 0.01


def test_predict_4():
    """
    Test that predict() works for discrete time.
    """
    lando = LANDO(**lando_params)
    lando.fit(X)
    lando_predict = lando.predict(
        x0=(-8, 8, 27),
        tend=len(t),
        continuous=False,
    )
    assert relative_error(lando_predict, X) < 0.05


def test_online_1():
    """
    Test that a LANDO model fitted with the online option yields the
    same results as a LANDO model fitted without the online option.
    """
    lando = LANDO(**lando_params)
    lando.fit(X, Y)

    lando_online = LANDO(online=True, **lando_params)
    lando_online.fit(X, Y)

    assert relative_error(lando_online.f(X), lando.f(X)) < 1e-6


def test_online_2():
    """
    Test that a LANDO model fitted with the online option and an update yields
    the same results as a LANDO model fitted without the online option.
    """
    lando = LANDO(**lando_params)
    lando.fit(X, Y)

    batch_split = int(0.8 * len(t))
    lando_online = LANDO(online=True, **lando_params)
    lando_online.fit(X[:, :batch_split], Y[:, :batch_split])
    lando_online.update(X[:, batch_split:], Y[:, batch_split:])

    assert relative_error(lando_online.f(X), lando.f(X)) < 1e-6


def test_online_3():
    """
    Test that a LANDO model fitted with the online option and an update
    properly updates fixed point analysis results after the update.
    """
    lando = LANDO(**lando_params)
    lando.fit(X, Y)
    lando.analyze_fixed_point(x_bar, compute_A=True)

    batch_split = int(0.8 * len(t))
    lando_online = LANDO(online=True, **lando_params)
    lando_online.fit(X[:, :batch_split], Y[:, :batch_split])
    lando_online.analyze_fixed_point(x_bar, compute_A=True)
    lando_online.update(X[:, batch_split:], Y[:, batch_split:])

    assert_equal(lando_online.fixed_point, lando.fixed_point)
    assert np.linalg.norm(lando_online.bias) < 1e-3
    assert relative_error(lando_online.linear, lando.linear) < 1e-5
    assert relative_error(lando_online.nonlinear(X), lando.nonlinear(X)) < 1e-5
