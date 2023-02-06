import numpy as np
from scipy.integrate import ode
from pydmd.bopdmd import BOPDMD

def f(t, y):
    """
    y'(t) = f(t, y)
    """
    z1, z2 = y
    z1_prime = z1 - 2 * z2
    z2_prime = z1 - z2
    return np.array((z1_prime, z2_prime))

def simulate_z(t):
    """
    Given a time vector t = t1, t2, ..., evaluates and returns the snapshots
    z(t1), z(t2), ... as columns of the matrix Z via explicit Runge-Kutta.

    Simulates data z given by the system of ODEs
        z' = Az
    where A = [1 -2; 1 -1] and z_0 = [1, 0.1].
    """
    z_0 = np.array((1.0, 0.1))
    Z = np.empty((2, len(t)))
    Z[:, 0] = z_0
    r = ode(f).set_integrator("dopri5")
    r.set_initial_value(z_0, t[0])
    for i, t_i in enumerate(t):
        if i == 0:
            continue
        r.integrate(t_i)
        Z[:, i] = r.y
    return Z

def sort_imag(x):
    """
    Helper method that sorts the entries of x by imaginary component, and then
    by real component in order to break ties.
    """
    x_real_imag_swapped = x.imag + 1j * x.real
    sorted_inds = np.argsort(x_real_imag_swapped)
    return x[sorted_inds]

# Simulate data.
t = np.arange(2000) * 0.01
t_long = np.arange(4000) * 0.01
t_uneven = np.delete(t, np.arange(1000)[1::2])
Z = simulate_z(t)
Z_long = simulate_z(t_long)
Z_uneven = np.delete(Z, np.arange(1000)[1::2], axis=1)

# Define the true eigenvalues of the system.
expected_eigs = np.array((-1j, 1j))

# Define the true system operator.
expected_A = np.array(((1, -2), (1, -1)))

def test_truncation_shape():
    """
    Tests that, when given a positive integer rank truncation, the shape of the
    modes, eigenvalues, amplitudes, Atilde operator, and A matrix are accurate.
    """
    bopdmd = BOPDMD(svd_rank=2, compute_A=True)
    bopdmd.fit(Z, t)
    assert bopdmd.modes.shape[1] == 2
    assert len(bopdmd.eigs) == 2
    assert len(bopdmd.amplitudes) == 2
    assert bopdmd.atilde.shape == (2, 2)
    assert bopdmd.A.shape == (2, 2)

def test_eigs():
    """
    Tests that the computed eigenvalues are accurate for the following cases:
    - standard optimized dmd, default parameters, even dataset
    - standard optimized dmd, default parameters, uneven dataset
    - standard optimized dmd, rank truncated
    - standard optimized dmd, fit full data
    - optimized dmd with bagging
    """
    bopdmd = BOPDMD()
    bopdmd.fit(Z, t)
    np.testing.assert_allclose(sort_imag(bopdmd.eigs), expected_eigs)

    bopdmd = BOPDMD()
    bopdmd.fit(Z_uneven, t_uneven)
    np.testing.assert_allclose(sort_imag(bopdmd.eigs), expected_eigs)

    bopdmd = BOPDMD(svd_rank=2)
    bopdmd.fit(Z, t)
    np.testing.assert_allclose(sort_imag(bopdmd.eigs), expected_eigs)

    bopdmd = BOPDMD(use_proj=False)
    bopdmd.fit(Z, t)
    np.testing.assert_allclose(sort_imag(bopdmd.eigs), expected_eigs)

    bopdmd = BOPDMD(num_trials=100, trial_size=0.2)
    bopdmd.fit(Z, t)
    np.testing.assert_allclose(sort_imag(bopdmd.eigs), expected_eigs)

def test_A():
    """
    Tests that the computed A matrix is accurate for the following cases:
    - standard optimized dmd, default parameters, even dataset
    - standard optimized dmd, default parameters, uneven dataset
    - standard optimized dmd, rank truncated
    - standard optimized dmd, fit full data
    - optimized dmd with bagging
    """
    bopdmd = BOPDMD(compute_A=True)
    bopdmd.fit(Z, t)
    np.testing.assert_allclose(bopdmd.A, expected_A)

    bopdmd = BOPDMD(compute_A=True)
    bopdmd.fit(Z_uneven, t_uneven)
    np.testing.assert_allclose(bopdmd.A, expected_A)

    bopdmd = BOPDMD(svd_rank=2, compute_A=True)
    bopdmd.fit(Z, t)
    np.testing.assert_allclose(bopdmd.A, expected_A)

    bopdmd = BOPDMD(compute_A=True, use_proj=False)
    bopdmd.fit(Z, t)
    np.testing.assert_allclose(bopdmd.A, expected_A)

    bopdmd = BOPDMD(compute_A=True, num_trials=100, trial_size=0.2)
    bopdmd.fit(Z, t)
    np.testing.assert_allclose(bopdmd.A, expected_A)

def test_reconstruction():
    """
    Tests the accuracy of the reconstructed data for the default parameters.
    Tests for both standard optimized dmd and BOP-DMD.
    """
    bopdmd = BOPDMD()
    bopdmd.fit(Z, t)
    np.testing.assert_allclose(bopdmd.reconstructed_data, Z, rtol=1e-5)

    bopdmd = BOPDMD(num_trials=100, trial_size=0.2)
    bopdmd.fit(Z, t)
    np.testing.assert_allclose(bopdmd.reconstructed_data, Z, rtol=1e-5)

def test_forecast():
    """
    Tests that the BOP-DMD model generalizes to data not used for training.
    Tests the following cases:
    - Generalizing to long dataset after training on short, even dataset
        without bagging.
    - Generalizing to long dataset after training on short, uneven dataset
        without bagging.
    - Generalizing to long dataset after training on short, even dataset
        with bagging.
    """
    bopdmd = BOPDMD()
    bopdmd.fit(Z, t)
    np.testing.assert_allclose(bopdmd.forecast(t_long), Z_long, rtol=1e-4)

    bopdmd = BOPDMD()
    bopdmd.fit(Z_uneven, t_uneven)
    np.testing.assert_allclose(bopdmd.forecast(t_long), Z_long, rtol=1e-4)

    bopdmd = BOPDMD(num_trials=100, trial_size=0.2)
    bopdmd.fit(Z, t)
    np.testing.assert_allclose(bopdmd.forecast(t_long)[0], Z_long, rtol=1e-4)
