import numpy as np
from pytest import raises
from scipy.integrate import solve_ivp
from numpy.testing import assert_equal

from pydmd import DMD
from pydmd import PiDMD
from pydmd import HAVOK


def generate_lorenz_data(t_eval):
    """
    Given a time vector t_eval = t1, t2, ..., evaluates and returns
    the snapshots of the Lorenz system as columns of the matrix X.
    """

    def lorenz_system(t, state):
        sigma, rho, beta = 10, 28, 8 / 3  # chaotic parameters
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
dt = 0.001  # time step
m = 50000  # number of data samples
t = np.arange(m) * dt
X = generate_lorenz_data(t)
x = X[0]


def test_error_fitted():
    """
    Ensure that attempting to get HAVOK attributes results
    in an error if fit() has not yet been called.
    """
    havok = HAVOK()

    with raises(ValueError):
        _ = havok.modes
    with raises(ValueError):
        _ = havok.singular_vals
    with raises(ValueError):
        _ = havok.delay_embeddings
    with raises(ValueError):
        _ = havok.linear_dynamics
    with raises(ValueError):
        _ = havok.forcing
    with raises(ValueError):
        _ = havok.operator
    with raises(ValueError):
        _ = havok.A
    with raises(ValueError):
        _ = havok.B
    with raises(ValueError):
        _ = havok.eigs
    with raises(ValueError):
        _ = havok.r


def test_hankel_1():
    """
    Test that the hankel and dehankel functions work as intended.
    Use 1-D data, lag = 1, and various delay values.
    """
    dummy_data = np.array([[1, 2, 3, 4]])

    havok = HAVOK(delays=1)
    assert_equal(
        havok.hankel(dummy_data),
        np.array(
            [
                [1, 2, 3, 4],
            ]
        ),
    )
    assert_equal(havok.dehankel(havok.hankel(dummy_data)), dummy_data)

    havok = HAVOK(delays=2)
    assert_equal(havok.hankel(dummy_data), np.array([[1, 2, 3], [2, 3, 4]]))
    assert_equal(havok.dehankel(havok.hankel(dummy_data)), dummy_data)

    havok = HAVOK(delays=3)
    assert_equal(havok.hankel(dummy_data), np.array([[1, 2], [2, 3], [3, 4]]))
    assert_equal(havok.dehankel(havok.hankel(dummy_data)), dummy_data)

    havok = HAVOK(delays=4)
    assert_equal(
        havok.hankel(dummy_data),
        np.array(
            [
                [
                    1,
                ],
                [
                    2,
                ],
                [
                    3,
                ],
                [
                    4,
                ],
            ]
        ),
    )
    assert_equal(havok.dehankel(havok.hankel(dummy_data)), dummy_data)

    with raises(ValueError):
        havok = HAVOK(delays=5)
        havok.hankel(dummy_data)


def test_hankel_2():
    """
    Test that the hankel and dehankel functions work as intended.
    Use 2-D data, lag = 1, and various delay values.
    """
    dummy_data = np.array([[1, 2, 3], [4, 5, 6]])
    H2 = np.array([[1, 2], [4, 5], [2, 3], [5, 6]])
    H3 = np.array([[1, 4, 2, 5, 3, 6]]).T

    havok = HAVOK(delays=1)
    assert_equal(havok.hankel(dummy_data), dummy_data)
    assert_equal(havok.dehankel(havok.hankel(dummy_data)), dummy_data)

    havok = HAVOK(delays=2)
    assert_equal(havok.hankel(dummy_data), H2)
    assert_equal(havok.dehankel(havok.hankel(dummy_data)), dummy_data)

    havok = HAVOK(delays=3)
    assert_equal(havok.hankel(dummy_data), H3)
    assert_equal(havok.dehankel(havok.hankel(dummy_data)), dummy_data)

    with raises(ValueError):
        havok = HAVOK(delays=4)
        havok.hankel(dummy_data)


def test_hankel_3():
    """
    Test that the hankel and dehankel functions work as intended.
    Use 1-D data, lag = 2, and various delay values.
    """
    dummy_data = np.array([[1, 2, 3, 4, 5, 6]])
    H2 = np.array([[1, 2, 3, 4], [3, 4, 5, 6]])
    H3 = np.array([[1, 2], [3, 4], [5, 6]])

    # If only 1 delay is requested, the lag won't matter.
    havok = HAVOK(delays=1, lag=2)
    assert_equal(havok.hankel(dummy_data), dummy_data)
    assert_equal(havok.dehankel(havok.hankel(dummy_data)), dummy_data)

    havok = HAVOK(delays=2, lag=2)
    assert_equal(havok.hankel(dummy_data), H2)
    assert_equal(havok.dehankel(havok.hankel(dummy_data)), dummy_data)

    havok = HAVOK(delays=3, lag=2)
    assert_equal(havok.hankel(dummy_data), H3)
    assert_equal(havok.dehankel(havok.hankel(dummy_data)), dummy_data)

    with raises(ValueError):
        havok = HAVOK(delays=4, lag=2)
        havok.hankel(dummy_data)


def test_shape_1():
    """
    Using the default HAVOK parameters, checks that the shapes of
    linear_embeddings, forcing_input, A, and B are accurate.
    """
    havok = HAVOK()
    havok.fit(x, t)
    time_length = len(t) - havok.delays + 1
    assert havok.linear_dynamics.shape == (time_length, havok.r - 1)
    assert havok.forcing.shape == (time_length, 1)
    assert havok.A.shape == (havok.r - 1, havok.r - 1)
    assert havok.B.shape == (havok.r - 1, 1)


def test_shape_2():
    """
    Using num_chaos = 2, checks that the shapes of
    linear_embeddings, forcing_input, A, and B are accurate.
    """
    havok = HAVOK(num_chaos=2)
    havok.fit(x, t)
    time_length = len(t) - havok.delays + 1
    assert havok.linear_dynamics.shape == (time_length, havok.r - 2)
    assert havok.forcing.shape == (time_length, 2)
    assert havok.A.shape == (havok.r - 2, havok.r - 2)
    assert havok.B.shape == (havok.r - 2, 2)


def test_snapshots_1():
    """
    Test the stored snapshots and ho_snapshots.
    Ensure that they are accurate in the default case with 1-D data.
    """
    havok = HAVOK()
    with raises(ValueError):
        _ = havok.snapshots
    with raises(ValueError):
        _ = havok.ho_snapshots
    havok.fit(x, t)
    assert_equal(havok.snapshots, x)
    assert_equal(havok.ho_snapshots, havok.hankel(x))


def test_snapshots_2():
    """
    Test the stored snapshots and ho_snapshots.
    Ensure that they are accurate in the default case with 2-D data.
    """
    havok = HAVOK().fit(X, t)
    assert_equal(havok.snapshots, X)
    assert_equal(havok.ho_snapshots, havok.hankel(X))


def test_time_1():
    """
    Test the stored time attribute.
    Ensure that it is accurate in the default case.
    """
    havok = HAVOK()
    with raises(ValueError):
        _ = havok.time
    havok.fit(x, t)
    assert_equal(havok.time, t)


def test_time_2():
    """
    Test that a HAVOK model fitted with a time vector is essentially
    the same as a HAVOK model fitted with the time-step dt. Check the
    stored time vector and the computed HAVOK operator.
    """
    havok_1 = HAVOK().fit(x, t)
    havok_2 = HAVOK().fit(x, dt)
    assert_equal(havok_1.time, havok_2.time)
    assert_equal(havok_1.operator, havok_2.operator)


def test_plot_summary_1():
    """
    Test that everything is fine if we fit a HAVOK
    model and ask for the default summary plot.
    """
    havok = HAVOK(svd_rank=16, delays=100)
    havok.fit(x, t)
    havok.plot_summary()


def test_plot_summary_2():
    """
    Test that everything is fine if we fit a HAVOK model and
    ask for various (but still valid) plot modifications.
    """
    havok = HAVOK(svd_rank=16, delays=100)
    havok.fit(x, t)
    havok.plot_summary(
        num_plot=15000,
        index_linear=(0, 1),
        forcing_threshold=0.005,
        min_jump_dist=200,
        figsize=(15, 4),
        dpi=100,
    )


def test_plot_summary_3():
    """
    Test that an error is thrown if we ask for
    a summary from an unfitted HAVOK model.
    """
    havok = HAVOK()
    with raises(ValueError):
        havok.plot_summary()


def test_reconstruction_1():
    """
    Test the accuracy of the HAVOK reconstruction.
    """
    havok = HAVOK(svd_rank=16, delays=100).fit(x, t)
    error = x - havok.reconstructed_data
    assert np.linalg.norm(error) / np.linalg.norm(x) < 0.05


def test_reconstruction_2():
    """
    Test the accuracy of the sHAVOK reconstruction.
    """
    havok = HAVOK(svd_rank=4, delays=100, structured=True).fit(x, t)
    error = x[:-1] - havok.reconstructed_data
    assert np.linalg.norm(error) / np.linalg.norm(x[:-1]) < 0.07


def test_predict_1():
    """
    Test the accuracy of the HAVOK prediction.
    Ensure that prediction with the HAVOK forcing term and the times
    of fitting simply yields the computed data reconstruction.
    """
    havok = HAVOK(svd_rank=16, delays=100).fit(x, t)
    assert_equal(
        havok.predict(havok.forcing, havok.time),
        havok.reconstructed_data,
    )


def test_predict_2():
    """
    Test the accuracy of the HAVOK prediction.
    Test that predicting beyond the training set isn't absurdly inaccurate.
    """
    havok = HAVOK(svd_rank=16, delays=100).fit(x, t)

    # Build a longer data set and fit a HAVOK model to it.
    t_long = np.arange(2 * m) * dt
    x_long = generate_lorenz_data(t_long)[0]
    havok_long = HAVOK(svd_rank=16, delays=100).fit(x_long, t_long)

    # We only use the long HAVOK model to obtain a long forcing signal.
    forcing_long = havok_long.forcing
    time_long = t_long[: len(forcing_long)]

    # Get the error of the full prediction.
    error = x_long - havok.predict(forcing_long, time_long)
    assert np.linalg.norm(error) / np.linalg.norm(x_long) < 0.45


def test_predict_3():
    """
    Test the accuracy of the HAVOK prediction.
    Test that predicting with V0 indices is functionally
    the same as predicting with array-valued V0 inputs.
    """
    havok = HAVOK(svd_rank=16, delays=100).fit(x, t)
    assert_equal(
        havok.predict(havok.forcing, havok.time, V0=0),
        havok.predict(havok.forcing, havok.time, V0=havok.linear_dynamics[0]),
    )
    assert_equal(
        havok.predict(havok.forcing, havok.time, V0=1),
        havok.predict(havok.forcing, havok.time, V0=havok.linear_dynamics[1]),
    )
    assert_equal(
        havok.predict(havok.forcing, havok.time, V0=-1),
        havok.predict(havok.forcing, havok.time, V0=havok.linear_dynamics[-1]),
    )


def test_threshold_1():
    """
    Test compute_threshold function.
    Test that threshold computation works as expected, whether you plot or not.
    """
    havok = HAVOK(svd_rank=16, delays=100).fit(x, t)
    thres_1 = havok.compute_threshold(p=0.1, bins=100, plot=False)
    thres_2 = havok.compute_threshold(p=0.1, bins=100, plot=True)
    assert thres_1 == thres_2


def test_threshold_2():
    """
    Test compute_threshold function.
    Test that threshold computation works as expected, whether you index
    the stored forcing term, or provide a custom array-valued forcing term.
    """
    havok = HAVOK(svd_rank=16, delays=100).fit(x, t)
    vr = havok.forcing.flatten()
    thres_1 = havok.compute_threshold(p=0.1, bins=100)
    thres_2 = havok.compute_threshold(forcing=vr, p=0.1, bins=100)
    assert thres_1 == thres_2


def test_dmd_1():
    """
    Test that HAVOK works when used with an externally-defined DMD model.
    Test the basic exact DMD model - ensure that plot_summary works just
    fine and that the data reconstruction is still accurate.
    """
    dmd = DMD(svd_rank=-1)
    havok = HAVOK(svd_rank=16, delays=100, dmd=dmd).fit(x, t)
    havok.plot_summary()
    error = x - havok.reconstructed_data
    assert np.linalg.norm(error) / np.linalg.norm(x) < 0.05


def test_dmd_2():
    """
    Test that HAVOK works when used with an externally-defined DMD model.
    Test a physics-informed DMD model - ensure that plot_summary works just
    fine and that the data reconstruction is still accurate.
    """
    pidmd = PiDMD(
        svd_rank=-1,
        manifold="diagonal",
        manifold_opt=2,
        compute_A=True,
    )
    havok = HAVOK(svd_rank=4, delays=100, dmd=pidmd).fit(x, t)
    havok.plot_summary()
    error = x - havok.reconstructed_data
    assert np.linalg.norm(error) / np.linalg.norm(x) < 0.1
