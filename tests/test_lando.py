import numpy as np
from scipy.integrate import solve_ivp

from pydmd.lando import LANDO

# Chaotic Lorenz parameters:
sigma, rho, beta = 10, 28, 8/3

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

    # Settings to replicate odeint defaults.
    solve_ivp_opts = {}
    solve_ivp_opts["rtol"] = 1e-12
    solve_ivp_opts["atol"] = 1e-12
    solve_ivp_opts["method"] = "LSODA"

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

# Lorenz system fixed point.
x_bar = [-np.sqrt(beta * (rho - 1)), -np.sqrt(beta * (rho - 1)), rho - 1]

def test_fit_errors():
    """
    Test that
    """
    lando = LANDO(
        svd_rank=3,
        kernel_metric="poly",
        kernel_params={"degree":3, "coef0":1.0, "gamma":1.0},
        x_rescale=1/np.abs(X).max(axis=1),
        dict_tol=1e-6,
    )
    # with raises(ValueError):


def test_f():
    """
    Test that the computed function f() is accurate.
    """
    raise NotImplementedError()

def test_bias():
    """
    Test that the computed bias term is accurate.
    """
    raise NotImplementedError()
    
def test_linear():
    """
    Test that the computed linear operator is accurate.
    """
    raise NotImplementedError()

def test_eigs():
    """
    Test that the eigenvalues of the linear operator are accurate.
    """
    raise NotImplementedError()

def test_predict_1():
    """
    Test that predict() can be used to reconstruct and forecast the system.
    """
    raise NotImplementedError()

def test_predict_2():
    """
    Test that predict() can accurately produce data for new initial conditions.
    """
    raise NotImplementedError()

def test_online():
    """
    Test that a LANDO model fitted with the online option yields the
    same results as a LANDO model fitted without the online option.
    """
    raise NotImplementedError()