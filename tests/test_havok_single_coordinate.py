"""Regression tests for HAVOK models with one linear coordinate."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm
from scipy.signal import StateSpace, lsim

from pydmd import HAVOK


@pytest.mark.parametrize("num_linear", [1, 2])
@pytest.mark.parametrize("num_chaos", [1, 2])
@pytest.mark.parametrize("structured", [False, True])
def test_reconstructed_embeddings_shape(num_linear, num_chaos, structured):
    """Keep time samples as rows and linear coordinates as columns."""
    time = np.linspace(0, 10, 101)
    data = np.sin(time) + 0.2 * np.cos(2 * time)
    havok = HAVOK(
        svd_rank=num_linear + num_chaos,
        num_chaos=num_chaos,
        delays=4,
        structured=structured,
    ).fit(data, time)
    simulation_time = time[: len(havok.forcing)]
    system = StateSpace(
        havok.A, havok.B, np.eye(num_linear), np.zeros_like(havok.B)
    )
    expected_embeddings = lsim(
        system,
        U=havok.forcing,
        T=simulation_time,
        X0=havok.linear_dynamics[0],
    )[1].reshape(-1, num_linear)
    expected_data = havok.dehankel(
        (havok.modes[:, :num_linear] * havok.singular_vals[:num_linear])
        @ expected_embeddings.T
    )

    assert havok.reconstructed_embeddings.shape == havok.linear_dynamics.shape
    assert_allclose(havok.reconstructed_embeddings, expected_embeddings)
    assert havok.reconstructed_data.shape == expected_data.shape
    assert_allclose(havok.reconstructed_data, expected_data)
    assert_allclose(
        havok.predict(havok.forcing, simulation_time), expected_data
    )


@pytest.mark.parametrize("initial_condition", [0, 3, [0.7], 0.7])
@pytest.mark.parametrize("num_samples", [1, 11])
@pytest.mark.parametrize("multidimensional", [False, True])
def test_single_coordinate_prediction(
    initial_condition, num_samples, multidimensional
):
    """Check zero-forcing predictions against the scalar exponential."""
    time = np.linspace(0, 10, 101)
    data = np.sin(time) + 0.2 * np.cos(2 * time)
    if multidimensional:
        data = np.vstack([data, np.cos(time)])
    havok = HAVOK(svd_rank=2, delays=4).fit(data, time)
    prediction_time = time[:num_samples]
    if isinstance(initial_condition, int):
        initial_value = havok.linear_dynamics[initial_condition, 0]
    else:
        initial_value = np.asarray(initial_condition).item()
    expected_embeddings = initial_value * np.exp(
        havok.A[0, 0] * prediction_time
    )
    expected_data = havok.dehankel(
        havok.modes[:, :1]
        * havok.singular_vals[0]
        * expected_embeddings[None, :]
    )

    prediction = havok.predict(
        np.zeros((num_samples, 1)), prediction_time, V0=initial_condition
    )

    assert prediction.shape == expected_data.shape
    assert_allclose(prediction, expected_data)


@pytest.mark.parametrize("num_samples", [1, 11])
def test_multiple_coordinate_prediction(num_samples):
    """Preserve predictions when the model has multiple linear coordinates."""
    time = np.linspace(0, 10, 101)
    data = np.sin(time) + 0.2 * np.cos(2 * time)
    havok = HAVOK(svd_rank=3, delays=4).fit(data, time)
    prediction_time = time[:num_samples]
    initial_condition = np.array([0.7, -0.2])
    expected_embeddings = np.array(
        [expm(havok.A * value) @ initial_condition for value in prediction_time]
    )
    expected_data = havok.dehankel(
        (havok.modes[:, :2] * havok.singular_vals[:2]) @ expected_embeddings.T
    )

    prediction = havok.predict(
        np.zeros((num_samples, 1)), prediction_time, V0=initial_condition
    )

    assert prediction.shape == expected_data.shape
    assert_allclose(prediction, expected_data)
