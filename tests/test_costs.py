import numpy as np
from scipy.integrate import solve_ivp
import scipy
from pytest import raises

from pydmd.costs import COSTS


def overlapping_oscillators():
    """
    Given a time vector t_eval = t1, t2, ..., evaluates and returns
    the snapshots z(t1), z(t2), ... as columns of the matrix Z.
    Simulates data z given by the system of ODEs
        z' = Az
    where A = [1 -2; 1 -1] and z_0 = [1, 0.1].
    """

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

    # Orthonormalized linear mixing matrix
    Q = scipy.stats.ortho_group.rvs(n_space_dims, random_state=seed)
    Q = Q[0:n_space_dims, :]
    x = uv_tiled @ Q

    # COSTS expects time by space, so we transpose x.
    solution = x.T

    # For the scale separation we want to compare to the actual slow and fast
    # components.
    slow_modes = (
        uv_tiled[:, 0 : n_space_dims // 2] @ Q[0 : n_space_dims // 2, :]
    )
    fast_modes = uv_tiled[:, n_space_dims // 2 :] @ Q[n_space_dims // 2 :, :]

    return t_solution, solution, slow_modes.T, fast_modes.T


# Simulate data.
time, data, low_frequency, high_frequency = overlapping_oscillators()
# Define the true eigenvalues of the system.
expected_frequency_bands = np.array((1.1, 2.5))
# Define the expected error in the reconstructions.
expected_global_error = 0.045
expected_lf_error = 0.11
expected_hf_error = 0.17
expected_n_slides = 63

# Fit COSTS once and then test.
window = 60
step = 12
rank = 4
transform_method = "absolute"
pydmd_kwargs = {"eig_constraints": {"conjugate_pairs", "stable"}}
mrd = COSTS(
    svd_rank=rank,
    global_svd=True,
    pydmd_kwargs=pydmd_kwargs,
)
mrd.fit(data, np.atleast_2d(time), window, step, verbose=False)
# Force the clustering to use two components due to the nature of the toy data.
_ = mrd.cluster_omega(n_components=2, transform_method=transform_method)


def test_construction():
    """
    Test basic properties of building the windowed decomposition.
    """

    n_space_dims = np.shape(data)[0]

    assert mrd.n_slides == expected_n_slides
    assert mrd.window_length == window
    assert np.shape(mrd.time_array) == (mrd.n_slides, mrd.window_length)
    assert np.shape(mrd.modes_array) == (
        mrd.n_slides,
        n_space_dims,
        mrd.svd_rank,
    )
    assert np.shape(mrd.amplitudes_array) == (
        mrd.n_slides,
        mrd.svd_rank,
    )
    assert mrd.global_svd is True
    assert mrd.step_size == step


def test_bad_construction():
    mrd_alternative = COSTS()
    with raises(ValueError):
        mrd_alternative.fit(
            data, np.atleast_2d(time), len(time) + 1, step, verbose=False
        )


def test_window_construction():
    assert mrd.build_windows(data, window, step, integer_windows=True) == 61


def test_frequency_band_centroids():
    """
    Tests that the identified frequency bands are near the expected values.
    """

    np.testing.assert_allclose(
        mrd.cluster_centroids, expected_frequency_bands, atol=0.1
    )
    np.testing.assert_equal(np.unique(mrd.omega_classes), [0, 1])
    assert mrd.n_components == 2

    mrd_no_clustering = COSTS()
    assert mrd_no_clustering.n_components is None
    assert mrd_no_clustering.cluster_centroids is None


def test_reconstruction():
    """
    Tests the accuracy of the reconstructed data.
    """
    global_reconstruction = mrd.global_reconstruction()
    re_global = mrd.relative_error(global_reconstruction.real, data)
    np.testing.assert_allclose(re_global, expected_global_error, atol=0.01)


def test_scale_separation():
    """
    Tests the accuracy of the scale separation.
    """
    costs_lf, costs_hf = mrd.scale_separation()
    re_lf = mrd.relative_error(costs_lf.real, low_frequency)
    np.testing.assert_allclose(re_lf, expected_lf_error, atol=0.02)
    re_hf = mrd.relative_error(costs_hf.real, high_frequency)
    np.testing.assert_allclose(re_hf, expected_hf_error, atol=0.02)


def test_omega_transforms():
    """
    Tests that the COSTS module correctly transforms the eigenvalues yielding...
    - absolute values for "absolute"
    - squared eigenvalues for "square_frequencies"
    - 1 / absolute eigenvalues for "period"
    - log10 of the eignevalues for "log10"
    - a bad transformation which shouldn't work.
    """

    list_transform_methods = [
        "absolute",
        "square_frequencies",
        "period",
        "log10",
    ]

    for method in list_transform_methods:
        mrd_transform = COSTS(
            svd_rank=rank,
            global_svd=True,
            pydmd_kwargs=pydmd_kwargs,
        )

        omega_transform = mrd_transform.transform_omega(
            mrd.omega_array, transform_method=method
        )
        if method != "log10":
            assert np.all(omega_transform >= 0.0)
        elif method == "log10":
            assert np.allclose(
                np.log10(np.max(np.abs(mrd.omega_array.imag))),
                np.max(omega_transform),
            )
        assert np.all(np.isfinite(omega_transform))
        assert mrd_transform._omega_label is not None
        assert mrd_transform._hist_kwargs is not None

    with raises(ValueError):
        mrd_transform = COSTS(
            svd_rank=rank,
            global_svd=True,
            pydmd_kwargs=pydmd_kwargs,
        )

        mrd_transform.transform_omega(mrd.omega_array, transform_method="bad")


def test_to_xarray():
    """ """
    ds = mrd.to_xarray()
    mrd_convert = mrd.from_xarray(ds)

    assert np.allclose(mrd.omega_array, mrd_convert.omega_array)
    assert np.allclose(mrd.modes_array, mrd_convert.modes_array)
    assert np.allclose(mrd.cluster_centroids, mrd_convert.cluster_centroids)

    # The round trip of the pydmd_kwargs is sensitive to python and numpy version.
    assert np.allclose(
        mrd._pydmd_kwargs["proj_basis"], mrd_convert._pydmd_kwargs["proj_basis"]
    )
    for kw in mrd._pydmd_kwargs:
        if not kw == "proj_basis":
            assert mrd._pydmd_kwargs[kw] == mrd_convert._pydmd_kwargs[kw]


def test_plotters():
    mrd.plot_reconstructions(data)
    mrd.plot_scale_separation(data)
    mrd.plot_error(data)
    mrd.plot_time_series(1, data)
    mrd.plot_omega_histogram()
    mrd.plot_omega_time_series()
