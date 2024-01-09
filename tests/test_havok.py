import numpy as np
from pytest import raises
from scipy.integrate import solve_ivp

from pydmd import HAVOK


def generate_lorenz_data(t_eval):
    """
    Given a time vector t_eval = t1, t2, ..., evaluates and returns
    the snapshots of the Lorenz system as columns of the matrix X.
    """
    def lorenz_system(t, state):
        sigma, rho, beta = 10, 28, 8/3 # chaotic parameters
        x, y, z = state
        x_dot = sigma * (y - x)
        y_dot = (x * (rho - z)) - y
        z_dot = (x * y) - (beta * z)
        return [x_dot, y_dot, z_dot]

    # Set integrator keywords to replicate the odeint defaults
    integrator_keywords = {}
    integrator_keywords["rtol"] = 1e-12
    integrator_keywords["atol"] = 1e-12
    integrator_keywords["method"] = "LSODA"

    sol = solve_ivp(
        lorenz_system,
        [t_eval[0], t_eval[-1]],
        [-8, 8, 27],
        t_eval=t_eval,
        **integrator_keywords,
    )

    return sol.y


# Generate Lorenz system data.
dt = 0.001 # time step
m = 100000  # number of data samples
t = np.arange(m) * dt
X = generate_lorenz_data(t)
x = X[0]


def test_shape():
    """
    Using the default HAVOK parameters, checks that the shapes of
    linear_embeddings, forcing_input, A, and B are accurate.
    """
    havok = HAVOK()
    havok.fit(x, t)
    assert havok.linear_dynamics.shape == (len(t) - havok.delays + 1, havok.r - 1)
    assert havok.forcing.shape == (len(t) - havok.delays + 1, 1)
    assert havok.A.shape == (havok.r - 1, havok.r - 1)
    assert havok.B.shape == (havok.r - 1, 1)


def test_error_fitted():
    """
    Ensure that attempting to get the attributes linear_embeddings,
    forcing_input, A, B, or r results in a RuntimeError if fit()
    has not yet been called.
    """
    havok = HAVOK()
    with raises(ValueError):
        _ = havok.linear_dynamics
    with raises(ValueError):
        _ = havok.forcing
    with raises(ValueError):
        _ = havok.A
    with raises(ValueError):
        _ = havok.B
    with raises(ValueError):
        _ = havok.r


def test_error_small_r():
    """
    Ensure that a runtime error is thrown if r is too small.
    """
    havok = HAVOK(delays=1)
    with raises(RuntimeError):
        havok.fit(x, t)


def test_r():
    """
    Ensure the accuracy of the r property in the following situations:
    """
    # If no svd truncation, r is the min of the dimensions of the hankel matrix
    havok = HAVOK(svd_rank=-1)
    havok.fit(x, t)
    assert havok.r == min(havok.delays, len(t) - havok.delays + 1)

    # Test the above case, but for a larger d value
    havok = HAVOK(svd_rank=-1, delays=500)
    havok.fit(x, t)
    assert havok.r == min(havok.delays, len(t) - havok.delays + 1)

    # Test the above case, but for an even larger d value
    havok = HAVOK(svd_rank=-1, delays=len(t) - 20)
    havok.fit(x, t)
    assert havok.r == min(havok.delays, len(t) - havok.delays + 1)


def test_reconstruction():
    """
    Test the accuracy of the HAVOK reconstruction.
    """
    havok = HAVOK(svd_rank=15, delays=100)
    havok.fit(x, t)
    error = x - havok.reconstructed_data.real
    assert np.linalg.norm(error) / np.linalg.norm(x) < 0.2


def test_predict():
    """
    Test the accuracy of the HAVOK prediction.
    """


def test_time():
    """
    Test that 
    """
