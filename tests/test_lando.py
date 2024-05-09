import numpy as np
from pytest import raises, warns
from scipy.integrate import solve_ivp
from numpy.testing import assert_allclose, assert_equal

from pydmd.lando import LANDO

# Chaotic Lorenz parameters:
sigma, rho, beta = 10, 28, 8 / 3

# Lorenz system fixed point:
x_bar = [-np.sqrt(beta * (rho - 1)), -np.sqrt(beta * (rho - 1)), rho - 1]

# True linear operator at the fixed point:
A_true = np.array(
    [
        [-sigma, sigma, 0.0],
        [1.0, -1.0, np.sqrt(beta * (rho - 1))],
        [-np.sqrt(beta * (rho - 1)), -np.sqrt(beta * (rho - 1)), -beta],
    ]
)

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


def differentiate(X_data, time_step):
    """
    Method for performing 2nd order finite difference. Assumes the input
    matrix X is 2-D, with uniformly-sampled snapshots filling each column.
    Requires the time step between each snapshot.
    """
    if not isinstance(X_data, np.ndarray) or X_data.ndim != 2:
        raise ValueError("Please ensure that input data is a 2D array.")
    X_prime = np.empty(X_data.shape)
    X_prime[:, 1:-1] = (X_data[:, 2:] - X_data[:, :-2]) / (2 * time_step)
    X_prime[:, 0] = (-3 * X_data[:, 0] + 4 * X_data[:, 1] - X_data[:, 2]) / (
        2 * time_step
    )
    X_prime[:, -1] = (3 * X_data[:, -1] - 4 * X_data[:, -2] + X_data[:, -3]) / (
        2 * time_step
    )
    return X_prime


def generate_lorenz_data(t_eval, x0=(-8, 8, 27)):
    """
    Given a time vector t_eval = t1, t2, ..., evaluates and
    returns the snapshots of the Lorenz system as columns of
    the matrix X for the initial condition x0.
    """

    def lorenz_system(state_t, state):
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


def dummy_kernel(Xk, Yk):
    """
    Externally-defined linear kernel function.
    """
    return Xk.T.dot(Yk)


def dummy_grad(Xk, yk):
    """
    Externally-defined linear kernel function gradient.
    """
    return Xk.T


# Generate Lorenz system data.
dt = 0.001
t = np.arange(0, 10, dt)
X = generate_lorenz_data(t)
Y = differentiate(X, dt)

# Generate short version of the data.
X_short = X[:, :2000]
Y_short = Y[:, :2000]

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

    lando.fit(X_short, Y_short)
    assert lando.partially_fitted
    assert not lando.fitted

    lando.analyze_fixed_point(x_bar)
    assert lando.partially_fitted
    assert lando.fitted


def test_sparse_dictionary():
    """
    Test that the shapes of the sparse dictionary and that the sparse
    dictionary weights are as expected.
    """
    lando = LANDO(**lando_params)

    with raises(ValueError):
        _ = lando.sparse_dictionary

    lando.fit(X_short, Y_short)
    assert X_short.shape[0] == lando.sparse_dictionary.shape[0]
    assert X_short.shape[-1] > lando.sparse_dictionary.shape[-1]
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

    eigs_true, modes_true = np.linalg.eig(A_true)
    modes_rescale = np.divide(lando.modes[0], modes_true[0])
    modes_true_rescaled = np.multiply(modes_true, modes_rescale)

    assert relative_error(lando.linear, A_true) < 1e-4
    assert_allclose(lando.eigs, eigs_true, rtol=1e-4)
    assert_allclose(lando.modes, modes_true_rescaled, rtol=1e-3)


def test_nonlinear():
    """
    Test that the nonlinear operator is accurate.
    """
    lando = LANDO(**lando_params)
    lando.fit(X, Y)
    lando.analyze_fixed_point(x_bar)
    assert lando.nonlinear(X).shape == X.shape

    X_centered = X - np.array(x_bar)[..., None]
    N_est = lando.nonlinear(X_centered)
    N_true = Y - A_true.dot(X_centered)
    assert relative_error(N_est, N_true) < 1e-4


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
    assert relative_error(lando_predict, X_long) < 0.1


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
    assert relative_error(lando_predict, X) < 0.1


def test_online_1():
    """
    Test that a LANDO model fitted with the online option yields the
    same results as a LANDO model fitted without the online option.
    """
    lando = LANDO(**lando_params)
    lando.fit(X, Y)

    lando_online = LANDO(online=True, **lando_params)
    lando_online.fit(X, Y)

    assert relative_error(lando_online.f(X), lando.f(X)) < 1e-3


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

    assert relative_error(lando_online.f(X), lando.f(X)) < 1e-3


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
    assert relative_error(lando_online.linear, lando.linear) < 1e-3
    assert relative_error(lando_online.nonlinear(X), lando.nonlinear(X)) < 1e-3


def test_default_kernel():
    """
    Test that there are no errors when calling the default linear kernel.
    """
    lando = LANDO()
    lando.fit(X_short, Y_short)
    lando.analyze_fixed_point(x_bar)


def test_rbf_kernel():
    """
    Test that there are no errors when calling the default RBF kernel.
    """
    lando = LANDO(kernel_metric="rbf")
    lando.fit(X_short, Y_short)
    lando.analyze_fixed_point(x_bar)


def test_custom_kernel():
    """
    Test that there are no errors when using custom kernel functions.
    """
    lando = LANDO(kernel_function=dummy_kernel, kernel_gradient=dummy_grad)
    lando.fit(X_short, Y_short)
    lando.analyze_fixed_point(x_bar)


def test_custom_kernel_error():
    """
    Test that an error occurs if a user attempts a fixed point analysis with a
    custom kernel function but without a kernel gradient function.
    """
    lando = LANDO(kernel_function=dummy_kernel)
    lando.fit(X_short, Y_short)

    with raises(ValueError):
        lando.analyze_fixed_point(x_bar)


def test_kernel_inputs():
    """
    Tests various errors caught by the test_kernel_inputs function.
    """
    # Error should be thrown if an invalid kernel metric is given.
    with raises(ValueError):
        _ = LANDO(kernel_metric="blah")

    # Error should be thrown if kernel_params isn't a dict.
    with raises(TypeError):
        _ = LANDO(kernel_metric="poly", kernel_params=3)

    # Error should be thrown if kernel_params contains invalid entries.
    with raises(ValueError):
        _ = LANDO(kernel_metric="poly", kernel_params={"blah": 3})


def test_kernel_functions_1():
    """
    Tests various errors caught by the test_kernel_functions function.
    Tests for errors related to invalid inputs and combinations.
    """
    # Warning should arise if a kernel function is given without a gradient.
    with warns():
        _ = LANDO(kernel_function=dummy_kernel)

    # Error should be thrown if a gradient is given without a kernel function.
    with raises(ValueError):
        _ = LANDO(kernel_gradient=dummy_grad)

    # Error should be thrown if kernel_function isn't a function.
    with raises(TypeError):
        _ = LANDO(kernel_function=0, kernel_gradient=dummy_grad)

    # Error should be thrown if kernel_gradient isn't a function.
    with raises(TypeError):
        _ = LANDO(kernel_function=dummy_kernel, kernel_gradient=0)


def test_kernel_functions_2():
    """
    Tests various errors caught by the test_kernel_functions function.
    Tests for errors related to invalid function inputs.
    """

    # Define functions that malfunction when called:
    def bad_kernel_1(Xk, Yk):
        return dummy_kernel(Xk.T, Yk)

    def bad_grad_1(Xk, yk):
        return dummy_grad(Xk.T, yk)

    # Define functions that yield incorrect dimensions:
    def bad_kernel_2(Xk, Yk):
        return dummy_kernel(Xk, Yk).T

    def bad_grad_2(Xk, yk):
        return dummy_grad(Xk, yk).T

    with raises(ValueError):
        _ = LANDO(kernel_function=bad_kernel_1, kernel_gradient=dummy_grad)

    with raises(ValueError):
        _ = LANDO(kernel_function=bad_kernel_2, kernel_gradient=dummy_grad)

    with raises(ValueError):
        _ = LANDO(kernel_function=dummy_kernel, kernel_gradient=bad_grad_1)

    with raises(ValueError):
        _ = LANDO(kernel_function=dummy_kernel, kernel_gradient=bad_grad_2)


def test_supported_kernels():
    """
    Test a call to supported kernels.
    """
    lando = LANDO()
    print(lando.supported_kernels)


def test_errors_f():
    """
    Test that expected errors are thrown when calling f.
    """
    lando = LANDO()

    # Error should be thrown if f is called prior to fitting.
    with raises(ValueError):
        _ = lando.f(X_short)

    lando.fit(X_short, Y_short)

    # Error should be thrown if f is given data with the wrong dimension.
    with raises(ValueError):
        _ = lando.f(X_short[:-1])


def test_errors_predict():
    """
    Test that expected errors are thrown when calling predict.
    """
    lando = LANDO()

    # Error should be thrown if called prior to fitting.
    with raises(ValueError):
        _ = lando.predict(x0=(-8, 8, 27), tend=len(t))

    lando.fit(X_short, Y_short)

    # Error should be thrown if data is the wrong dimension.
    with raises(ValueError):
        _ = lando.predict(x0=(-8, 8), tend=len(t))


def test_errors_fixed_point():
    """
    Test that expected errors are thrown when calling analyze_fixed_point.
    """
    lando = LANDO()

    # Error should be thrown if called prior to fitting.
    with raises(ValueError):
        lando.analyze_fixed_point(x_bar)

    lando.fit(X_short, Y_short)

    # Error should be thrown if fixed point is the wrong dimension.
    with raises(ValueError):
        lando.analyze_fixed_point(x_bar[:-1])


def test_errors_update():
    """
    Test that expected errors are thrown when calling update.
    """
    lando = LANDO()

    # Error should be thrown if called prior to fitting.
    with raises(ValueError):
        lando.update(X_short, Y_short)

    lando.fit(X_short, Y_short)

    # Error should be thrown if data is the wrong dimension.
    with raises(ValueError):
        lando.update(X_short, Y_short[:, :-1])


def test_errors_get():
    """
    Test that errors are thrown if the following are attempted to be retrieved
    prior to fully fitting: fixed_point, bias, linear, nonlinear.
    """
    lando = LANDO()

    # Errors should be thrown if called prior to fitting:
    with raises(ValueError):
        _ = lando.fixed_point

    with raises(ValueError):
        _ = lando.bias

    with raises(ValueError):
        _ = lando.linear

    with raises(ValueError):
        _ = lando.nonlinear(X_short)

    lando.fit(X_short, Y_short)

    # Errors should still be thrown after a call to just fit:
    with raises(ValueError):
        _ = lando.fixed_point

    with raises(ValueError):
        _ = lando.bias

    with raises(ValueError):
        _ = lando.linear

    with raises(ValueError):
        _ = lando.nonlinear(X_short)


def test_reconstructed_data():
    """
    Test reconstructed data shape and output.
    """
    lando = LANDO()
    lando.fit(X_short, Y_short)
    lando.analyze_fixed_point(x_bar)

    # Calling for reconstructed data should yield a warning.
    with warns():
        X_recon = lando.reconstructed_data

    assert X_recon.shape == X_short.shape
