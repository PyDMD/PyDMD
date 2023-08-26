import numpy as np
from pytest import raises
from scipy.integrate import odeint
import torch

from pydmd import HAVOK


def lorenz_system(t, state, sigma, rho, beta):
    """
    Defines the system of differential equations y'(t) = f(t, y, params)
    """
    x, y, z = state
    x_dot = sigma * (y - x)
    y_dot = (x * (rho - z)) - y
    z_dot = (x * y) - (beta * z)
    return np.array((x_dot, y_dot, z_dot))


def generate_lorenz_data(t):
    """
    Given a time vector t = t1, t2, ..., evaluates and returns the snapshots
    of the Lorenz system as columns of the matrix X via explicit Runge-Kutta.
    """
    # Chaotic Lorenz parameters
    sigma, rho, beta = 10, 28, 8 / 3

    # Initial condition
    initial = np.array((-8, 8, 27))

    # Generate Lorenz data
    X = np.empty(len(t))
    X[0] = initial[0]

    return odeint(
        lorenz_system, initial, t, args=(sigma, rho, beta), tfirst=True
    )[:, 0]


# Generate chaotic Lorenz System data
dt = 0.01
t = np.arange(0, 100, dt)
lorenz_x = generate_lorenz_data(t)


def test_shape():
    """
    Using the default HAVOK parameters, checks that the shapes of
    linear_embeddings, forcing_input, A, and B are accurate.
    """
    havok = HAVOK()
    havok.fit(lorenz_x, dt)
    assert havok.linear_embeddings.shape == (len(t) - havok._d + 1, havok.r - 1)
    assert havok.forcing_input.shape == (len(t) - havok._d + 1,)
    assert havok.A.shape == (havok.r - 1, havok.r - 1)
    assert havok.B.shape == (havok.r - 1, 1)


def test_error_fitted():
    """
    Ensure that attempting to get the attributes linear_embeddings,
    forcing_input, A, B, or r results in a RuntimeError if fit()
    has not yet been called.
    """
    havok = HAVOK()
    with raises(RuntimeError):
        _ = havok.linear_embeddings
    with raises(RuntimeError):
        _ = havok.forcing_input
    with raises(RuntimeError):
        _ = havok.A
    with raises(RuntimeError):
        _ = havok.B
    with raises(RuntimeError):
        _ = havok.r


def test_error_1d():
    """
    Ensure that the fit function will reject data that isn't one-dimensional.
    """
    havok = HAVOK()
    with raises(ValueError):
        havok.fit(np.zeros((2, 100)), dt)


def test_error_small_r():
    """
    Ensure that a runtime error is thrown if r is too small.
    """
    havok = HAVOK(d=1)
    with raises(RuntimeError):
        havok.fit(lorenz_x, dt)


def test_r():
    """
    Ensure the accuracy of the r property in the following situations:
    """
    # If no svd truncation, r is the min of the dimensions of the hankel matrix
    havok = HAVOK(svd_rank=-1)
    havok.fit(lorenz_x, dt)
    assert havok.r == min(havok._d, len(t) - havok._d + 1)

    # Test the above case, but for a larger d value
    havok = HAVOK(svd_rank=-1, d=500)
    havok.fit(lorenz_x, dt)
    assert havok.r == min(havok._d, len(t) - havok._d + 1)

    # Test the above case, but for an even larger d value
    havok = HAVOK(svd_rank=-1, d=len(t) - 20)
    havok.fit(lorenz_x, dt)
    assert havok.r == min(havok._d, len(t) - havok._d + 1)

    # If given a positive integer svd truncation, r should equal svd_rank
    havok = HAVOK(svd_rank=3)
    havok.fit(lorenz_x, dt)
    assert havok.r == havok.operator._svd_rank


def test_reconstruction():
    """
    Test the accuracy of the HAVOK reconstruction. Note that the parameters
    used here have been successful in reconstructing the Lorenz System.
    """
    havok = HAVOK(svd_rank=15, d=100)
    havok.fit(lorenz_x, dt)
    error = lorenz_x - havok.reconstructed_data.real
    error_norm = np.linalg.norm(error) / np.linalg.norm(lorenz_x)
    assert error_norm < 0.6


def test_rejects_torch():
    havok = HAVOK(svd_rank=15, d=100)
    with raises(
        ValueError, match="PyTorch not supported with this DMD variant"
    ):
        havok.fit(torch.ones(1000), dt)
