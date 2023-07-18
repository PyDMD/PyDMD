import numpy as np
from scipy.integrate import ode
from pytest import raises
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
    - standard optimized dmd, even dataset
    - standard optimized dmd, uneven dataset
    - standard optimized dmd, fit full data
    - optimized dmd with bagging
    """
    bopdmd = BOPDMD(svd_rank=2)
    bopdmd.fit(Z, t)
    np.testing.assert_allclose(sort_imag(bopdmd.eigs), expected_eigs)

    bopdmd = BOPDMD(svd_rank=2)
    bopdmd.fit(Z_uneven, t_uneven)
    np.testing.assert_allclose(sort_imag(bopdmd.eigs), expected_eigs)

    bopdmd = BOPDMD(svd_rank=2, use_proj=False)
    bopdmd.fit(Z, t)
    np.testing.assert_allclose(sort_imag(bopdmd.eigs), expected_eigs)

    bopdmd = BOPDMD(svd_rank=2, num_trials=100, trial_size=0.2)
    bopdmd.fit(Z, t)
    np.testing.assert_allclose(sort_imag(bopdmd.eigs), expected_eigs)


def test_A():
    """
    Tests that the computed A matrix is accurate for the following cases:
    - standard optimized dmd, even dataset
    - standard optimized dmd, uneven dataset
    - standard optimized dmd, fit full data
    - optimized dmd with bagging
    """
    bopdmd = BOPDMD(svd_rank=2, compute_A=True)
    bopdmd.fit(Z, t)
    np.testing.assert_allclose(bopdmd.A, expected_A)

    bopdmd = BOPDMD(svd_rank=2, compute_A=True)
    bopdmd.fit(Z_uneven, t_uneven)
    np.testing.assert_allclose(bopdmd.A, expected_A)

    bopdmd = BOPDMD(svd_rank=2, compute_A=True, use_proj=False)
    bopdmd.fit(Z, t)
    np.testing.assert_allclose(bopdmd.A, expected_A)

    bopdmd = BOPDMD(svd_rank=2, compute_A=True, num_trials=100, trial_size=0.2)
    bopdmd.fit(Z, t)
    np.testing.assert_allclose(bopdmd.A, expected_A)


def test_reconstruction():
    """
    Tests the accuracy of the reconstructed data.
    Tests for both standard optimized dmd and BOP-DMD.
    """
    bopdmd = BOPDMD(svd_rank=2)
    bopdmd.fit(Z, t)
    np.testing.assert_allclose(bopdmd.reconstructed_data, Z, rtol=1e-5)

    bopdmd = BOPDMD(svd_rank=2, num_trials=100, trial_size=0.2)
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
    bopdmd = BOPDMD(svd_rank=2)
    bopdmd.fit(Z, t)
    np.testing.assert_allclose(bopdmd.forecast(t_long), Z_long, rtol=1e-3)

    bopdmd = BOPDMD(svd_rank=2)
    bopdmd.fit(Z_uneven, t_uneven)
    np.testing.assert_allclose(bopdmd.forecast(t_long), Z_long, rtol=1e-3)

    bopdmd = BOPDMD(svd_rank=2, num_trials=100, trial_size=0.2)
    bopdmd.fit(Z, t)
    np.testing.assert_allclose(bopdmd.forecast(t_long)[0], Z_long, rtol=1e-3)


def test_compute_A():
    """
    Tests that the BOPDMD module appropriately calculates or doesn't calculate
    A depending on the compute_A flag. Also tests that atilde, the dmd modes,
    and the dmd eigenvalues are not effected by the compute_A flag.
    """
    bopdmd_with_A = BOPDMD(svd_rank=2, compute_A=True)
    bopdmd_no_A = BOPDMD(svd_rank=2, compute_A=False)
    bopdmd_with_A.fit(Z, t)
    bopdmd_no_A.fit(Z, t)

    np.testing.assert_allclose(bopdmd_with_A.A, expected_A)
    with raises(ValueError):
        print(bopdmd_no_A.A)

    np.testing.assert_array_equal(bopdmd_with_A.atilde, bopdmd_no_A.atilde)
    np.testing.assert_array_equal(bopdmd_with_A.modes, bopdmd_no_A.modes)
    np.testing.assert_array_equal(bopdmd_with_A.eigs, bopdmd_no_A.eigs)


def test_eig_constraints_errors():
    """
    Tests that the BOPDMD module correctly throws an error upon initialization
    in each of the following cases:
    - eig_constraints is a string rather than a set of strings
    - eig_constraints contains an invalid constraint argument
        (either alone or along with another constraint argument)
    - eig_constraints contains the invalid combination "stable"+"imag"
        (either alone or along with the extra argument "conjugate_pairs")
    """
    with raises(ValueError):
        bopdmd = BOPDMD(eig_constraints="stable")

    with raises(ValueError):
        bopdmd = BOPDMD(eig_constraints={"stablee"})

    with raises(ValueError):
        bopdmd = BOPDMD(eig_constraints={"stablee", "imag"})

    with raises(ValueError):
        bopdmd = BOPDMD(eig_constraints={"stable", "imag"})

    with raises(ValueError):
        bopdmd = BOPDMD(eig_constraints={"stable", "imag", "conjugate_pairs"})


def test_eig_constraints():
    """
    Tests that the BOPDMD module correctly enforces that eigenvalues...
    - lie in the left half plane when eig_constraints contains "stable".
    - lie on the imaginary axis when eig_constraints contains "imag".
    - are present with their complex conjugate when eig_constraints
        contains "conjugate_pairs".
    Eigenvalue constraint combinations are also tested.
    """
    bopdmd = BOPDMD(svd_rank=2, eig_constraints={"stable"})
    bopdmd.fit(Z, t)
    assert np.all(bopdmd.eigs.real <= 0.0)

    bopdmd = BOPDMD(svd_rank=2, eig_constraints={"imag"})
    bopdmd.fit(Z, t)
    assert np.all(bopdmd.eigs.real == 0.0)

    bopdmd = BOPDMD(svd_rank=2, eig_constraints={"conjugate_pairs"})
    bopdmd.fit(Z, t)
    assert bopdmd.eigs[0].real == bopdmd.eigs[1].real
    assert bopdmd.eigs[0].imag == -bopdmd.eigs[1].imag

    bopdmd = BOPDMD(svd_rank=2, eig_constraints={"stable", "conjugate_pairs"})
    bopdmd.fit(Z, t)
    assert np.all(bopdmd.eigs.real <= 0.0)
    assert bopdmd.eigs[0].real == bopdmd.eigs[1].real
    assert bopdmd.eigs[0].imag == -bopdmd.eigs[1].imag

    bopdmd = BOPDMD(svd_rank=2, eig_constraints={"imag", "conjugate_pairs"})
    bopdmd.fit(Z, t)
    assert np.all(bopdmd.eigs.real == 0.0)
    assert bopdmd.eigs[0].imag == -bopdmd.eigs[1].imag
