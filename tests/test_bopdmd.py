import numpy as np
import matplotlib.pyplot as plt
from pytest import raises, warns
from scipy.integrate import solve_ivp

from pydmd.bopdmd import BOPDMD
from pydmd.bopdmd import BOPDMDOperator


def simulate_z(t_eval):
    """
    Given a time vector t_eval = t1, t2, ..., evaluates and returns
    the snapshots z(t1), z(t2), ... as columns of the matrix Z.
    Simulates data z given by the system of ODEs
        z' = Az
    where A = [1 -2; 1 -1] and z_0 = [1, 0.1].
    """

    def ode_sys(zt, z):
        z1, z2 = z
        return [z1 - 2 * z2, z1 - z2]

    # Set integrator keywords to replicate odeint defaults.
    integrator_keywords = {}
    integrator_keywords["rtol"] = 1e-12
    integrator_keywords["method"] = "LSODA"
    integrator_keywords["atol"] = 1e-12

    sol = solve_ivp(
        ode_sys,
        [t_eval[0], t_eval[-1]],
        [1.0, 0.1],
        t_eval=t_eval,
        **integrator_keywords,
    )

    return sol.y


def sort_imag(x):
    """
    Helper method that sorts the entries of x by imaginary component, and then
    by real component in order to break ties.
    """
    x_real_imag_swapped = x.imag + 1j * x.real
    sorted_inds = np.argsort(x_real_imag_swapped)
    return x[sorted_inds]


def compute_error(actual, truth):
    """
    Helper method that computes relative error.
    """
    return np.linalg.norm(truth - actual) / np.linalg.norm(truth)


# Simulate data.
t = np.arange(4000) * 0.01
t_long = np.arange(6000) * 0.01
t_uneven = np.delete(t, np.arange(1000)[1::2])
Z = simulate_z(t)
Z_long = simulate_z(t_long)
Z_uneven = np.delete(Z, np.arange(1000)[1::2], axis=1)
rng = np.random.default_rng(seed=1234)
Z_noisy = Z + 0.01 * rng.standard_normal(Z.shape)

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

    bopdmd = BOPDMD(svd_rank=2, num_trials=10, trial_size=0.8)
    bopdmd.fit(Z, t)
    np.testing.assert_allclose(sort_imag(bopdmd.eigs), expected_eigs)


def test_A():
    """
    Tests that the computed A matrix is accurate for the following cases:
    - standard optimized dmd, even dataset
    - standard optimized dmd, uneven dataset
    - standard optimized dmd, noisy dataset
    - standard optimized dmd, fit full data
    - optimized dmd with bagging
    """
    bopdmd = BOPDMD(svd_rank=2, compute_A=True)
    bopdmd.fit(Z, t)
    assert compute_error(bopdmd.A, expected_A) < 1e-3

    bopdmd = BOPDMD(svd_rank=2, compute_A=True)
    bopdmd.fit(Z_uneven, t_uneven)
    assert compute_error(bopdmd.A, expected_A) < 1e-3

    bopdmd = BOPDMD(svd_rank=2, compute_A=True)
    bopdmd.fit(Z_noisy, t)
    assert compute_error(bopdmd.A, expected_A) < 1e-3

    bopdmd = BOPDMD(svd_rank=2, compute_A=True, use_proj=False)
    bopdmd.fit(Z, t)
    assert compute_error(bopdmd.A, expected_A) < 1e-3

    bopdmd = BOPDMD(svd_rank=2, compute_A=True, num_trials=10, trial_size=0.8)
    bopdmd.fit(Z, t)
    assert compute_error(bopdmd.A, expected_A) < 1e-3


def test_reconstruction():
    """
    Tests the accuracy of the reconstructed data.
    Tests for both standard optimized dmd and BOP-DMD.
    """
    bopdmd = BOPDMD(svd_rank=2)
    bopdmd.fit(Z, t)
    np.testing.assert_allclose(bopdmd.reconstructed_data, Z, rtol=1e-5)

    bopdmd = BOPDMD(svd_rank=2, num_trials=10, trial_size=0.8)
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
    assert compute_error(bopdmd.forecast(t_long), Z_long) < 1e-2

    bopdmd = BOPDMD(svd_rank=2)
    bopdmd.fit(Z_uneven, t_uneven)
    assert compute_error(bopdmd.forecast(t_long), Z_long) < 1e-2

    bopdmd = BOPDMD(svd_rank=2, num_trials=10, trial_size=0.8)
    bopdmd.fit(Z, t)
    assert compute_error(bopdmd.forecast(t_long)[0], Z_long) < 1e-2


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
    with raises(TypeError):
        BOPDMD(eig_constraints="stable")

    with raises(ValueError):
        BOPDMD(eig_constraints={"stablee"})

    with raises(ValueError):
        BOPDMD(eig_constraints={"stablee", "imag"})

    with raises(ValueError):
        BOPDMD(eig_constraints={"stable", "imag"})

    with raises(ValueError):
        BOPDMD(eig_constraints={"stable", "limited"})

    with raises(ValueError):
        BOPDMD(eig_constraints={"imag", "limited"})

    with raises(ValueError):
        BOPDMD(eig_constraints={"stable", "imag", "conjugate_pairs"})


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

    bopdmd = BOPDMD(
        svd_rank=2,
        eig_constraints={"limited", "conjugate_pairs"},
        real_eig_limit=0.001,
    )
    bopdmd.fit(Z, t)
    assert np.all(np.abs(bopdmd.eigs.real) < 0.001)
    assert bopdmd.eigs[0].imag == -bopdmd.eigs[1].imag


def test_eig_constraints_errors_2():
    """
    Tests that the BOPDMD module correctly throws an error upon initialization
    whenever eig_constraints is a function and...
    - eig_constraints is incompatible with general (n,) numpy.ndarray inputs
    - eig_constraints doesn't return a single numpy.ndarray
    - eig_constraints takes multiple arguments
    - eig_constraints changes the shape of the input array
    """

    # Function that assumes the input is length 3.
    def bad_func_1(x):
        return np.multiply(x, np.arange(3))

    # Function that assumes the input array is at least 2-dimensional.
    def bad_func_2(x):
        return x[0, :] + x[1, :]

    # Function that doesn't return an array.
    def bad_func_3(x):
        return len(x)

    # Function that returns 2 arrays instead of 1.
    def bad_func_4(x):
        return x, 2 * x

    # Function that accepts more than 1 input.
    def bad_func_5(x, y):
        return x + y

    # Function that returns an array with a different shape.
    def bad_func_6(x):
        return x[:-1]

    # Function that returns an array with an extra dimension.
    def bad_func_7(x):
        return x[:, None]

    with raises(ValueError):
        BOPDMD(eig_constraints=bad_func_1)

    with raises(ValueError):
        BOPDMD(eig_constraints=bad_func_2)

    with raises(ValueError):
        BOPDMD(eig_constraints=bad_func_3)

    with raises(ValueError):
        BOPDMD(eig_constraints=bad_func_4)

    with raises(ValueError):
        BOPDMD(eig_constraints=bad_func_5)

    with raises(ValueError):
        BOPDMD(eig_constraints=bad_func_6)

    with raises(ValueError):
        BOPDMD(eig_constraints=bad_func_7)

    # See that bad_func_1 is fine if the svd_rank is 3.
    BOPDMD(svd_rank=3, eig_constraints=bad_func_1)


def test_eig_constraints_2():
    """
    Tests that if the eig_constraints function discards all real parts,
    the functionality is the same as setting eig_constraints={"imag"}.
    """

    def make_imag(x):
        return 1j * x.imag

    # Providing a custom function performs a slightly different branch of the
    # variable projection solution, so we must test to be almost equal not
    # exactly equal.
    bopdmd1 = BOPDMD(svd_rank=2, eig_constraints={"imag"}).fit(Z, t)
    bopdmd2 = BOPDMD(svd_rank=2, eig_constraints=make_imag).fit(Z, t)
    np.testing.assert_almost_equal(bopdmd1.eigs, bopdmd2.eigs, decimal=10)


def test_eig_constraints_conjugate():
    """
    Tests if the "conjugate_pairs" constraint is able to pair eigenvalues
    that do not exactly adhere to well-separated conjugate pairs. All the
    tested arrays are real states that the BOPDMD solver arrived at even
    though the system was describable by conjugate pairs.
    """

    # This example has an ambiguous pairing.
    eigs_1 = np.array(
        [
            0.00658707 + 1.06858631j,
            0.00362803 - 1.10256819j,
            -0.00633559 + 0.32695117j,
            -0.00062934 - 0.31402938j,
            0.0040683 + 0.70560261j,
            -0.00163795 - 0.7185244j,
            0.00393976 + 0.6847774j,
            -0.00098072 - 0.65079552j,
        ]
    )

    # This example lacks a conjugate pair but clearly has two eigenvalues
    # with the same imaginary component magnitude.
    eigs_2 = np.array(
        [
            1j * 0.84244753,
            1j * 0.08135547,
            1j * 0.10386969,
            1j * -1.02767269,
        ]
    )

    # This example has the imaginary component of a pair equal to exactly zero.
    eigs_3 = np.array(
        [
            0.01060103 + 2.37774049j,
            0.01060103 - 2.37774049j,
            0.2533088 + 0.0j,
            -0.3247746 + 0.0j,
        ]
    )

    # An odd svd rank example.
    eigs_4 = np.array(
        [
            7.17854760e-05 + 1.14433634j,
            7.17854760e-05 - 1.14433634j,
            3.46250187e-06 + 0.88297678j,
            3.46250187e-06 - 0.88297678j,
            -1.89823212e-06 + 0.64454973j,
            -1.89823212e-06 - 0.64454973j,
            0.03e-06 + 0j,
            7.09378586e-08 + 0.20859477j,
            7.09378586e-08 - 0.20859477j,
            -3.40460840e-06 + 0.42177977j,
            -3.40460840e-06 - 0.42177977j,
        ]
    )

    # This example has nearly degenerate conjugate pairs.
    eigs_5 = np.array(
        [
            0.01054178 - 0.19798906j,
            0.01053877 + 0.19798656j,
            0.00768596 + -0.41628322j,
            0.00765037 + 0.41628555j,
            0.23208416 + 0.41926024j,
            0.23389110 - 0.41705963j,
        ]
    )

    # This example comes from the solver going into a local minima, far away
    # from conjugate pairs, requiring a strong nudge back towards conjugate
    # pairs.
    eigs_6 = np.array(
        [
            -0.01767645 + 0.21779537j,
            -0.01767645 - 0.21779537j,
            0.00961734 + 0.41684479j,
            -0.00961734 - 0.41684479j,
            -0.13112859 - 0.04273738j,
            -0.15072534 - 0.17359778j,
            0.31147709 - 0.59712213j,
            0.29188035 + 0.81345729j,
        ]
    )

    eig_constraints = {"conjugate_pairs"}
    BDO = BOPDMDOperator(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        eig_constraints,
        None,
        None,
        None,
        None,
        None,
    )

    eig_list = [eigs_1, eigs_2, eigs_3, eigs_4, eigs_5, eigs_6]

    for eigs in eig_list:
        # We have to make a copy because the function changes the contents of
        # the input vector.
        new_eigs = BDO._push_eigenvalues(np.copy(eigs))

        # We should have only n // 2 unique imaginary eigenvalues or n // 2 +
        # 1 for the case of the odd rank with a DC mode.
        np.testing.assert_array_equal(
            len(np.unique(np.abs(new_eigs.imag))),
            np.ceil(len(eigs) / 2).astype(int),
        )


def test_plot_mode_uq():
    """
    Test that a basic call to plot_mode_uq is successful.
    """
    bopdmd = BOPDMD(svd_rank=2, num_trials=10, trial_size=0.8)
    bopdmd.fit(Z, t)
    bopdmd.plot_mode_uq()
    plt.close()


def test_plot_eig_uq():
    """
    Test that a basic call to plot_eig_uq is successful.
    """
    bopdmd = BOPDMD(svd_rank=2, num_trials=10, trial_size=0.8)
    bopdmd.fit(Z, t)
    bopdmd.plot_eig_uq()
    plt.close()


def test_plot_error():
    """
    Test that UQ plotters fail if bagging wasn't used.
    """
    bopdmd = BOPDMD(svd_rank=2, num_trials=0)
    bopdmd.fit(Z, t)

    with raises(ValueError):
        bopdmd.plot_mode_uq()

    with raises(ValueError):
        bopdmd.plot_eig_uq()


def test_varpro_opts_warn():
    """
    Test that errors or warnings are correctly thrown if invalid
    or poorly-chosen variable projection parameters are given.
    The `tol` parameter is specifically tested.
    """
    with raises(TypeError):
        bopdmd = BOPDMD(varpro_opts_dict={"tol": None})
        bopdmd.fit(Z, t)

    with warns():
        bopdmd = BOPDMD(varpro_opts_dict={"tol": -1.0})
        bopdmd.fit(Z, t)

    with warns():
        bopdmd = BOPDMD(varpro_opts_dict={"tol": np.inf})
        bopdmd.fit(Z, t)


def test_varpro_opts_print():
    """
    Test that variable projection parameters can be printed after fitting.
    """
    bopdmd = BOPDMD(svd_rank=2)

    with raises(ValueError):
        bopdmd.print_varpro_opts()

    bopdmd.fit(Z, t)
    bopdmd.print_varpro_opts()


def test_verbose_outputs_1():
    """
    Test variable projection verbosity for optimized DMD.
    """
    bopdmd = BOPDMD(
        svd_rank=2,
        num_trials=0,
        varpro_opts_dict={"verbose": True},
    )
    bopdmd.fit(Z, t)


def test_verbose_outputs_2():
    """
    Test variable projection verbosity for BOP-DMD.
    """
    bopdmd = BOPDMD(
        svd_rank=2,
        num_trials=10,
        trial_size=0.8,
        varpro_opts_dict={"verbose": True},
    )
    bopdmd.fit(Z, t)


def test_verbose_outputs_3():
    """
    Test variable projection verbosity for BOP-DMD without bad bags.
    """
    bopdmd = BOPDMD(
        svd_rank=2,
        num_trials=10,
        trial_size=0.8,
        varpro_opts_dict={"verbose": True, "tol": 1.0},
        remove_bad_bags=True,
    )
    bopdmd.fit(Z, t)


def test_bag_int():
    """
    Test that trial_size can be a valid integer value.
    """
    bopdmd = BOPDMD(svd_rank=2, num_trials=10, trial_size=3200)
    bopdmd.fit(Z, t)


def test_bag_getters():
    """
    Test calls to the num_trials and trial_size parameters.
    """
    bopdmd = BOPDMD(svd_rank=2, num_trials=0, trial_size=0.8)
    assert bopdmd.num_trials == 0
    assert bopdmd.trial_size == 0.8

    bopdmd = BOPDMD(svd_rank=2, num_trials=100, trial_size=3200)
    assert bopdmd.num_trials == 100
    assert bopdmd.trial_size == 3200


def test_bag_error():
    """
    Test that errors are thrown if invalid bagging parameters are given.
    """
    # Error should raise if trial_size isn't a positive integer...
    with raises(ValueError):
        bopdmd = BOPDMD(svd_rank=2, num_trials=10, trial_size=-1)
        bopdmd.fit(Z, t)

    # ...or if it isn't a float in the range (0.0, 1.0).
    with raises(ValueError):
        bopdmd = BOPDMD(svd_rank=2, num_trials=10, trial_size=2.0)
        bopdmd.fit(Z, t)

    # Error should raise if the requested trial size is too big...
    with raises(ValueError):
        bopdmd = BOPDMD(svd_rank=2, num_trials=10, trial_size=4001)
        bopdmd.fit(Z, t)

    # ...or if the requested trial size is too small.
    with raises(ValueError):
        bopdmd = BOPDMD(svd_rank=2, num_trials=10, trial_size=1e-6)
        bopdmd.fit(Z, t)


def test_mode_prox():
    """
    Test that the mode_prox function is applied as expected.
    """

    def dummy_prox(X):
        return X + 1.0

    # Test that the function is applied in the use_proj=False case.
    bopdmd = BOPDMD(svd_rank=2, use_proj=False, mode_prox=dummy_prox)
    bopdmd.fit(Z, t)

    # Test that the function is applied in the use_proj=True case.
    bopdmd = BOPDMD(svd_rank=2, use_proj=True, mode_prox=dummy_prox)
    bopdmd.fit(Z, t)

    # Compare use_proj=True results to the no prox case.
    bopdmd_noprox = BOPDMD(svd_rank=2, use_proj=True)
    bopdmd_noprox.fit(Z, t)
    np.testing.assert_allclose(bopdmd.modes, dummy_prox(bopdmd_noprox.modes))


def test_init_alpha_initializer():
    """
    Test that the eigenvalues are accurately initialized by default.
    """
    bopdmd = BOPDMD(svd_rank=2)

    # Initial eigs shouldn't be defined yet.
    with raises(ValueError):
        _ = bopdmd.init_alpha

    # After fitting, the initial eigs used should be fairly accurate.
    bopdmd.fit(Z, t)
    np.testing.assert_allclose(
        sort_imag(bopdmd.init_alpha),
        expected_eigs,
        rtol=0.01,
    )


def test_proj_basis_initializer():
    """
    Test that the projection basis is accurately initialized by default.
    """
    bopdmd = BOPDMD(svd_rank=2)

    # Projection basis shouldn't be defined yet.
    with raises(ValueError):
        _ = bopdmd.proj_basis

    # After fitting, the projection basis used should be accurate.
    bopdmd.fit(Z, t)
    np.testing.assert_array_equal(bopdmd.proj_basis, np.linalg.svd(Z)[0])


def test_svd_rank():
    """
    Test svd_rank getter and setter.
    """
    bopdmd = BOPDMD(svd_rank=2)
    assert bopdmd.svd_rank == 2

    bopdmd.svd_rank = 0
    assert bopdmd.svd_rank == 0


def test_init_alpha():
    """
    Test init_alpha getter and setter.
    """
    dummy_eigs = 2.0 * expected_eigs
    bopdmd = BOPDMD(init_alpha=dummy_eigs)
    np.testing.assert_array_equal(bopdmd.init_alpha, dummy_eigs)

    bopdmd.init_alpha = 2.0 * dummy_eigs
    np.testing.assert_array_equal(bopdmd.init_alpha, 2.0 * dummy_eigs)


def test_proj_basis():
    """
    Test proj_basis getter and setter.
    """
    dummy_basis = 2.0 * np.linalg.svd(Z)[0]
    bopdmd = BOPDMD(proj_basis=dummy_basis)
    np.testing.assert_array_equal(bopdmd.proj_basis, dummy_basis)

    bopdmd.proj_basis = 2.0 * dummy_basis
    np.testing.assert_array_equal(bopdmd.proj_basis, 2.0 * dummy_basis)


def test_default_initializers():
    """
    Test that default initialization does not impact custom inputs.
    """
    # Define the dummy init_alpha and proj_basis to NOT be the defaults.
    dummy_eigs = 2.0 * expected_eigs
    dummy_basis = 2.0 * np.linalg.svd(Z)[0]
    bopdmd = BOPDMD(
        svd_rank=2,
        init_alpha=dummy_eigs,
        proj_basis=dummy_basis,
    )
    bopdmd.fit(Z, t)
    np.testing.assert_array_equal(bopdmd.init_alpha, dummy_eigs)
    np.testing.assert_array_equal(bopdmd.proj_basis, dummy_basis)


def test_std_shape():
    """
    Test the shapes of the standard deviation attributes.
    """
    bopdmd = BOPDMD(svd_rank=2, num_trials=10, trial_size=0.8)
    bopdmd.fit(Z, t)

    assert bopdmd.eigenvalues_std.shape == bopdmd.eigs.shape
    assert bopdmd.modes_std.shape == bopdmd.modes.shape
    assert bopdmd.amplitudes_std.shape == bopdmd.amplitudes.shape


def test_std_nobags():
    """
    Test that std attributes are simply None if no bags were used.
    """
    bopdmd = BOPDMD(svd_rank=2, num_trials=0)
    bopdmd.fit(Z, t)

    assert bopdmd.eigenvalues_std is None
    assert bopdmd.modes_std is None
    assert bopdmd.amplitudes_std is None


def test_getter_errors():
    """
    Test that error occurs if the following properties are called before fit:
    time, atilde, A, eigenvalues_std, modes_std, amplitudes_std.
    """
    bopdmd = BOPDMD(compute_A=True)

    with raises(ValueError):
        _ = bopdmd.time

    with raises(ValueError):
        _ = bopdmd.atilde

    with raises(ValueError):
        _ = bopdmd.A

    with raises(ValueError):
        _ = bopdmd.eigenvalues_std

    with raises(ValueError):
        _ = bopdmd.modes_std

    with raises(ValueError):
        _ = bopdmd.amplitudes_std


def test_fit_econ():
    """
    Test that the fit_econ method gives the same results as the fit method.
    """
    svd_rank = 2
    bopdmd = BOPDMD(svd_rank=svd_rank, use_proj=True)
    bopdmd.fit(Z, t)
    forecast = bopdmd.forecast(t_long)

    U, s, V = np.linalg.svd(Z)
    U = U[:, :svd_rank]
    s = s[:svd_rank]
    V = V[:svd_rank, :]
    bopdmd_econ = BOPDMD(svd_rank=svd_rank, use_proj=True, proj_basis=U)
    bopdmd_econ.fit_econ(s, V, t)
    forecast_econ = bopdmd_econ.forecast(t_long)

    np.testing.assert_allclose(bopdmd_econ.eigs, bopdmd.eigs)
    np.testing.assert_allclose(bopdmd_econ.modes, bopdmd.modes)
    np.testing.assert_allclose(bopdmd_econ.amplitudes, bopdmd.amplitudes)
    np.testing.assert_allclose(forecast_econ.real, forecast.real)


def test_fit_econ_with_bagging():
    """
    Test that the fit_econ method gives the same results as the fit method
    when bagging is used.
    """
    svd_rank = 2
    num_trials = 5
    bopdmd = BOPDMD(svd_rank=svd_rank, use_proj=True, num_trials=num_trials)
    np.random.seed(42)
    bopdmd.fit(Z, t)
    forecast, _ = bopdmd.forecast(t_long)

    U, s, V = np.linalg.svd(Z)
    U = U[:, :svd_rank]
    s = s[:svd_rank]
    V = V[:svd_rank, :]
    bopdmd_econ = BOPDMD(
        svd_rank=svd_rank, use_proj=True, proj_basis=U, num_trials=num_trials
    )
    np.random.seed(42)
    bopdmd_econ.fit_econ(s, V, t)
    forecast_econ, _ = bopdmd_econ.forecast(t_long)

    np.testing.assert_allclose(bopdmd_econ.eigs, bopdmd.eigs)
    np.testing.assert_allclose(bopdmd_econ.modes, bopdmd.modes)
    np.testing.assert_allclose(bopdmd_econ.amplitudes, bopdmd.amplitudes)
    np.testing.assert_allclose(forecast_econ.real, forecast.real)
