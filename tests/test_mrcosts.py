import numpy as np
from scipy.integrate import solve_ivp
import scipy
from pytest import raises
import glob
import os

from pydmd.mrcosts import mrCOSTS


def build_multiscale_process():
    """ """

    def rhs_FNM(t, x, tau, a, b, Iext):
        # FitzHugh-Nagumo Model
        v = x[0]
        w = x[1]
        vdot = v - (v**3) / 3 - w + Iext
        wdot = (1 / tau) * (v + a - b * w)
        dx = np.array([vdot, wdot])

        return dx

    def rhs_UFD(t, y, eta, epsilon, tau):
        # Unforced Duffing Oscillator
        p = y[0]
        q = y[1]
        pdot = q
        qdot = (1 / tau) * (-2 * eta * q - p - epsilon * p**3)
        dy = np.array([pdot, qdot])

        return dy

    T = 64

    x0 = np.array([-1.110, -0.125])
    tau1 = 2
    a = 0.7
    b = 0.8
    Iext = 0.65

    y0 = np.array([0, 1])
    eta = 0
    epsilon = 1
    tau2 = 0.2

    # RK4 integration of the mixed system
    dt = 0.0001 * 8
    t_solution = np.arange(0, T, dt)

    # Solve the FitzHugh-Nagumo Model
    solution_fn = solve_ivp(
        rhs_FNM, [0, T], x0, t_eval=t_solution, args=(tau1, a, b, Iext)
    )

    # Solve the Unforced Duffing Oscillator Model
    solution_ufd = solve_ivp(
        rhs_UFD, [0, T], y0, t_eval=t_solution, args=(eta, epsilon, tau2)
    )

    seed = 1
    num_space_dims = 4

    uv_tiled = np.hstack(
        [
            np.tile(solution_fn.y.T, num_space_dims),
            np.tile(solution_ufd.y.T, num_space_dims),
        ]
    )

    # Subsample after solving the pdes
    substep = 100
    uv_tiled = uv_tiled[0::substep, :]
    t_solution = t_solution[0::substep]

    # Dimension of space to map into
    n_space_dims = np.shape(uv_tiled)[1]
    n_time = np.shape(uv_tiled)[0]

    # Orthonormalized linear mixing matrix
    Q = scipy.stats.ortho_group.rvs(n_space_dims, random_state=seed)
    Q = Q[0:n_space_dims, :]
    x = uv_tiled @ Q

    # COSTS expects time by space, so we transpose x.
    data_original = x.T

    # For the scale separation we want to compare to the actual slow and fast
    # components.
    slow_modes = (
        uv_tiled[:, 0 : n_space_dims // 2] @ Q[0 : n_space_dims // 2, :]
    )
    fast_modes = uv_tiled[:, n_space_dims // 2 :] @ Q[n_space_dims // 2 :, :]

    # Add a transient wave packet
    recon_filter_sd = len(t_solution) * 0.25

    recon_filter = np.exp(
        -((np.arange(n_time) - (n_time + 1) / 2) ** 2) / recon_filter_sd**2
    )
    recon_filter[recon_filter < 0.0001] = 0
    f_transient = 10
    x_transient = (
        0.5
        * np.sin(f_transient * t_solution.flatten())
        * np.sin(0.25 * t_solution.flatten())
        * recon_filter
    )

    # Add the transient feature to the data
    data_transient = data_original + np.atleast_2d(x_transient)

    return (
        t_solution,
        data_transient,
        slow_modes.T,
        fast_modes.T,
        x_transient.T,
    )


# Simulate data.
(
    time,
    data,
    low_frequency,
    high_frequency,
    transient,
) = build_multiscale_process()
# Define the true eigenvalues of the system.
expected_frequency_bands = np.array((0.4, 1.0))
expected_n_components = 2

# Define the expected error in the reconstructions.
expected_global_error = 0.053
expected_lf_error = 0.12
expected_hf_error = 0.19
expected_transient_error = 0.3

# Fit mrCOSTS for testing
window_lengths = [15, 60]
step_sizes = [1, 12]
svd_ranks = [4] * len(window_lengths)
suppress_growth = True
transform_method = "square_frequencies"
n_components_array = [2] * len(window_lengths)
global_svd_array = [True] * len(window_lengths)

mrc = mrCOSTS(
    svd_rank_array=svd_ranks,
    window_length_array=window_lengths,
    step_size_array=step_sizes,
    global_svd_array=global_svd_array,
    transform_method=transform_method,
    n_components_array=n_components_array,
)
mrc.fit(data, np.atleast_2d(time))
mrc.multi_res_interp()

# Global clustering
n_components_range = np.arange(2, 8)
scores, n_optimal = mrc.global_cluster_hyperparameter_sweep(
    n_components_range, transform_method="log10", verbose=False
)

cluster_centroids, omega_classes, omega_array = mrc.global_cluster_omega(
    transform_method="log10"
)


def test_frequency_band_centroids():
    """
    Tests that the identified frequency bands are near the expected values.
    """

    np.testing.assert_allclose(
        mrc.cluster_centroids, expected_frequency_bands, atol=0.1
    )
    np.testing.assert_equal(np.unique(mrc.omega_classes_interpolated), [0, 1])
    assert mrc._n_components_global == 2

    assert len(mrc.ragged_omega_classes) == len(window_lengths)
    om_cl = mrc.ragged_omega_classes
    np.testing.assert_equal(np.unique(om_cl[0]), [-1, 1])
    np.testing.assert_equal(np.unique(om_cl[1]), [-1, 0])

    assert mrc.transform_method == transform_method

    # Verify these fields are empty prior to clustering.
    mrc_no_clustering = mrCOSTS()
    assert mrc_no_clustering._n_components_global is None
    assert mrc_no_clustering.cluster_centroids is None
    assert mrc_no_clustering.transform_method is None


def test_reconstructions():
    """
    Tests the accuracy of the reconstructed data.
    """
    xr_sep = mrc.global_scale_reconstruction()
    xr_hf = xr_sep.sum(axis=0)[0, :, :].squeeze()
    xr_transient = xr_sep.sum(axis=0)[1, :, :].squeeze()
    xr_lf = mrc.get_background()
    xr_global = mrc.global_reconstruction()

    mrd = mrc.costs_array[0]
    re_global = mrd.relative_error(xr_global, data)
    re_lf = mrd.relative_error(xr_lf, low_frequency)
    re_hf = mrd.relative_error(xr_hf, high_frequency)

    n_time_steps, n_data_vars = mrc._data_shape(data)

    error_array = np.zeros(n_data_vars)
    for n in range(n_data_vars):
        error_array[n] = mrc.costs_array[0].relative_error(
            xr_transient[n, :], xr_transient
        )
    re_transient = error_array.mean()

    np.testing.assert_allclose(re_global, expected_global_error, atol=0.01)
    np.testing.assert_allclose(re_lf, expected_lf_error, atol=0.01)
    np.testing.assert_allclose(re_hf, expected_hf_error, atol=0.01)
    # There is a strong random component to this test, necessitating very loose
    # tolerances.
    np.testing.assert_allclose(
        re_transient, expected_transient_error, atol=0.15
    )


def test_omega_transforms():
    """
    Tests that the COSTS module correctly transforms the eigenvalues yielding...
    - absolute values for "absolute"
    - squared eigenvalues for "square_frequencies"
    - 1 / absolute eigenvalues for "period"
    - log10 of the eignevalues for "log10"
    Eigenvalue constraint combinations are also tested.
    """

    list_transform_methods = [
        "absolute",
        "square_frequencies",
        "period",
        "log10",
    ]

    omega_to_transform = mrc.ragged_omega_array[1]

    for method in list_transform_methods:
        mrc_transformer = mrCOSTS()

        omega_transformed = mrc_transformer.transform_omega(
            omega_to_transform, transform_method=method
        )
        if method != "log10":
            assert np.all(omega_transformed >= 0.0)
        elif method == "log10":
            assert np.allclose(
                np.log10(np.max(np.abs(omega_to_transform.imag))),
                np.max(omega_transformed),
            )
        assert np.all(np.isfinite(omega_transformed))

    with raises(ValueError):
        mrc_transformer = mrCOSTS()
        mrc_transformer.transform_omega(
            omega_to_transform, transform_method="bad"
        )


def test_netcdf():
    """ """
    mrc.to_netcdf("tests")
    file_list = glob.glob("*tests*.nc")
    mrc_from_file = mrCOSTS()
    mrc_from_file.from_netcdf(file_list)

    for n in range(mrc.n_decompositions):
        assert np.allclose(
            mrc.ragged_modes_array[n], mrc_from_file.ragged_modes_array[n]
        )
    assert np.allclose(mrc.cluster_centroids, mrc.cluster_centroids)


def test_plot_local_reconstructions():
    mrc.plot_local_reconstructions(0, data=data)

    with raises(ValueError):
        mrc.plot_local_reconstructions(0)

    with raises(ValueError):
        mrc.plot_local_reconstructions(0, data=data.T)


def test_plot_local_error():
    mrc.plot_local_error(0, data=data)

    with raises(ValueError):
        mrc.plot_local_error(0)

    with raises(ValueError):
        mrc.plot_local_error(0, data=data.T)


def test_plot_local_scale_separation():
    _ = mrc.plot_local_scale_separation(0, data=data)

    with raises(ValueError):
        _ = mrc.plot_local_scale_separation(0)

    with raises(ValueError):
        _ = mrc.plot_local_scale_separation(0, data=data.T)


def test_plot_local_time_series():
    _ = mrc.plot_local_time_series(0, 0, data=data)

    with raises(ValueError):
        _ = mrc.plot_local_time_series(0, 0)

    with raises(ValueError):
        _ = mrc.plot_local_time_series(0, 0, data=data.T)


def tear_down():
    os.remove("*tests*")
