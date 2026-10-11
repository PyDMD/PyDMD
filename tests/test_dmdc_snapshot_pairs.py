"""DMDc identification from aligned, independently sampled snapshot pairs."""

import numpy as np
import pytest

from pydmd import DMDc


def _system(seed=564, complex_data=False, n_controls=2, n_samples=40):
    """Generate independent pairs for a known controlled system."""
    rng = np.random.default_rng(seed)
    a = np.array([[0.8, 0.15, 0], [-0.1, 0.65, 0.2], [0.05, 0, 0.9]])
    b = np.array([[1, 0.2], [-0.3, 0.8], [0.5, -0.7]])[:, :n_controls]
    x = rng.normal(size=(3, n_samples))
    u = rng.normal(size=(n_controls, n_samples))
    if complex_data:
        a = a + 1j * np.diag([0.1, -0.2, 0.05])
        b = b * (1 + 0.4j)
        x = x + 1j * rng.normal(size=x.shape)
        u = u + 1j * rng.normal(size=u.shape)
    return a, b, x, u, a @ x + b @ u


def _model(**kwargs):
    """Build a full-rank model with optional configuration settings."""
    return DMDc(svd_rank=-1, svd_rank_omega=-1, **kwargs)


def _assert_operator(model, a, b):
    """Check the state and control operators without relying on mode order."""
    # Basis-vector predictions recover A without relying on mode ordering.
    np.testing.assert_allclose(
        model.predict(np.eye(a.shape[0]), np.zeros((b.shape[1], a.shape[0]))),
        a,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(model.B, b, rtol=1e-10, atol=1e-10)


def _trajectory(a, b, controls, lag=1):
    """Simulate a controlled recurrence at the requested snapshot lag."""
    rng = np.random.default_rng(912)
    snapshots = np.empty((a.shape[0], controls.shape[1] + lag))
    snapshots[:, :lag] = rng.normal(size=(a.shape[0], lag))
    for index, control in enumerate(controls.T):
        snapshots[:, index + lag] = a @ snapshots[:, index] + b @ control
    return snapshots


@pytest.mark.parametrize("known_b", [False, True])
@pytest.mark.parametrize("complex_data", [False, True])
def test_independent_pairs_identify_operator_and_predict(known_b, complex_data):
    """Identify A and B from independent pairs and predict unseen states."""
    a, b, x, u, y = _system(complex_data=complex_data)
    model = _model()
    result = model.fit(x, u, b if known_b else None, Y=y)
    np.testing.assert_equal(result is model, True)
    _assert_operator(model, a, b)

    _, _, held_out_x, held_out_u, held_out_y = _system(
        seed=565, complex_data=complex_data, n_samples=11
    )
    np.testing.assert_allclose(
        model.predict(held_out_x, control_input=held_out_u),
        held_out_y,
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize("known_b", [False, True])
def test_common_column_permutation_preserves_identified_system(known_b):
    """Learn the same system after jointly shuffling snapshots and controls."""
    a, b, x, u, y = _system()
    order = np.random.default_rng(123).permutation(x.shape[1])
    original = _model().fit(x, u, b if known_b else None, Y=y)
    shuffled = _model().fit(
        x[:, order], u[:, order], b if known_b else None, Y=y[:, order]
    )
    _assert_operator(shuffled, a, b)
    _, _, test_x, test_u, _ = _system(seed=566, n_samples=9)
    np.testing.assert_allclose(
        shuffled.predict(test_x, test_u),
        original.predict(test_x, test_u),
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize("known_b", [False, True])
def test_disjoint_experiments_do_not_create_boundary_transitions(known_b):
    """Learn valid transitions across experiments with independent resets."""
    a, b, _, _, _ = _system()
    rng = np.random.default_rng(44)
    inputs, targets, controls = [], [], []
    for length in (5, 8, 4, 7):
        state = 10 * rng.normal(size=3)
        for _ in range(length):
            control = rng.normal(size=2)
            next_state = a @ state + b @ control
            inputs.append(state)
            targets.append(next_state)
            controls.append(control)
            state = next_state
    model = _model().fit(
        np.column_stack(inputs),
        np.column_stack(controls),
        b if known_b else None,
        Y=np.column_stack(targets),
    )
    _assert_operator(model, a, b)


@pytest.mark.parametrize("known_b", [False, True])
def test_rank_reduced_model_predicts_on_identified_subspace(known_b):
    """Predict controlled motion within a rank-two state subspace."""
    basis, _ = np.linalg.qr(np.array([[1, 0], [0, 1], [1, 1], [-1, 1]]))
    latent_a = np.array([[0.85, 0.2], [-0.1, 0.65]])
    latent_b = np.array([[0.5], [1.2]])
    rng = np.random.default_rng(73)
    coordinates = rng.normal(size=(2, 40))
    controls = rng.normal(size=(1, 40))
    x = basis @ coordinates
    y = basis @ (latent_a @ coordinates + latent_b @ controls)
    b = basis @ latent_b
    model = DMDc(svd_rank=2, svd_rank_omega=3)
    # The ambient snapshots are deliberately confined to a
    # two-dimensional plane.
    with pytest.warns(UserWarning, match="condition number"):
        model.fit(x, controls, b if known_b else None, Y=y)
    np.testing.assert_equal(model.modes.shape, (4, 2))
    test_coordinates = rng.normal(size=(2, 9))
    test_controls = rng.normal(size=(1, 9))
    np.testing.assert_allclose(
        model.predict(basis @ test_coordinates, test_controls),
        basis @ (latent_a @ test_coordinates + latent_b @ test_controls),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(model.B, b, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("representation", ["array", "list", "generator"])
def test_spatial_snapshot_arrays_and_iterables(representation):
    """Normalize spatial snapshots consistently across supported containers."""
    a, b, x, u, y = _system()

    def convert(data):
        """Preserve spatial shape while varying the snapshot container."""
        spatial = data.reshape(1, 3, -1)
        if representation == "array":
            return spatial
        snapshots = [spatial[:, :, index] for index in range(data.shape[1])]
        return iter(snapshots) if representation == "generator" else snapshots

    model = _model().fit(convert(x), u, Y=convert(y))
    np.testing.assert_array_equal(model.snapshots, x)
    np.testing.assert_array_equal(model.snapshots_y, y)
    np.testing.assert_equal(model.snapshots_shape, (1, 3))
    _assert_operator(model, a, b)


@pytest.mark.parametrize("known_b", [False, True])
@pytest.mark.parametrize("lag", [1, 3])
def test_explicit_pairs_match_legacy_trajectory_fit(known_b, lag):
    """Match sequential fitting when explicit pairs encode the same lag."""
    a, b, _, controls, _ = _system()
    snapshots = _trajectory(a, b, controls, lag=lag)
    supplied_b = b if known_b else None
    legacy = _model(lag=lag).fit(snapshots, I=controls, B=supplied_b)
    paired = _model(lag=lag).fit(
        snapshots[:, :-lag], controls, supplied_b, Y=snapshots[:, lag:]
    )
    _assert_operator(legacy, a, b)
    _assert_operator(paired, a, b)
    np.testing.assert_allclose(
        legacy.reconstructed_data(), snapshots, rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        paired.predict(snapshots[:, :-lag], controls),
        snapshots[:, lag:],
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_equal(legacy.snapshots_y is None, True)


@pytest.mark.parametrize("opt", [False, 0, 2, -1])
def test_amplitudes_project_selected_input_snapshot(opt):
    """Project the input snapshot selected by the amplitude option."""
    _, _, x, u, y = _system()
    model = _model(opt=opt).fit(x, u, Y=y)
    column = 0 if opt is False else opt
    np.testing.assert_allclose(
        model.modes @ model.amplitudes, x[:, column], rtol=1e-10, atol=1e-10
    )


def test_optimized_amplitudes_require_ordered_snapshots():
    """Reject optimized amplitudes without an ordered training trajectory."""
    _, _, x, u, y = _system()
    with pytest.raises(ValueError, match="opt"):
        _model(opt=True).fit(x, u, Y=y)


@pytest.mark.parametrize("operation", ["dynamics", "reconstructed_data"])
def test_paired_model_rejects_temporal_reconstruction(operation):
    """Direct unordered-pair users to prediction for state transitions."""
    _, _, x, u, y = _system()
    model = _model().fit(x, u, Y=y)
    with pytest.raises(RuntimeError, match="predict"):
        if operation == "dynamics":
            _ = model.dynamics
        else:
            model.reconstructed_data()


@pytest.mark.parametrize(
    "n_controls,state_shape,control_shape",
    [
        (2, "vector", "vector"),
        (2, "vector", "matrix"),
        (2, "column", "vector"),
        (1, "vector", "scalar"),
        (1, "vector", "vector"),
        (1, "batch", "vector"),
        (1, "batch", "matrix"),
    ],
)
def test_prediction_shapes(n_controls, state_shape, control_shape):
    """Preserve vector or matrix state shapes for compatible controls."""
    a, b, x, u, y = _system(n_controls=n_controls)
    model = _model().fit(x, u.squeeze() if n_controls == 1 else u, Y=y)
    if state_shape == "vector":
        states, controls = x[:, 0], u[:, :1]
        expected = a @ states + b @ controls[:, 0]
    elif state_shape == "column":
        states, controls = x[:, :1], u[:, :1]
        expected = a @ states + b @ controls
    else:
        states, controls = x[:, :4], u[:, :4]
        expected = a @ states + b @ controls
    if control_shape == "vector":
        controls = controls.ravel()
    elif control_shape == "scalar":
        controls = controls.item()
    prediction = model.predict(states, controls)
    np.testing.assert_equal(prediction.shape, states.shape)
    np.testing.assert_allclose(prediction, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "states,controls",
    [
        (np.zeros(2), np.zeros(2)),
        (np.zeros((3, 2, 1)), np.zeros((2, 2))),
        (np.zeros(3), np.zeros(3)),
        (np.zeros((3, 3)), np.zeros((1, 3))),
        (np.zeros((3, 3)), np.zeros((2, 2))),
        (np.zeros((3, 3)), np.zeros(3)),
        (np.zeros(3), 1.0),
        (np.zeros((3, 3)), np.zeros((2, 3, 1))),
    ],
)
def test_prediction_rejects_incompatible_shapes_without_broadcast(
    states, controls
):
    """Reject state and control shapes that would require broadcasting."""
    _, _, x, u, y = _system()
    model = _model().fit(x, u, Y=y)
    with pytest.raises(ValueError):
        model.predict(states, controls)


def test_prediction_requires_fitted_model():
    """Require a learned operator before predicting controlled states."""
    with pytest.raises(RuntimeError):
        _model().predict(np.zeros(3), np.zeros(2))


@pytest.mark.parametrize("vector_input", [False, True])
def test_real_model_predicts_complex_states_and_controls(vector_input):
    """Promote real model outputs when controls or states are complex."""
    _, b, x, u, _ = _system()
    # Real eigenvalues keep the state contribution real in the batch case.
    a = np.diag([0.8, 0.65, 0.9])
    model = _model().fit(x, u, Y=a @ x + b @ u)
    states = x[:, :5]
    controls = u[:, :5] + 1j * u[:, 5:10]
    if vector_input:
        states = states[:, 0] + 1j * x[:, 5]
        controls = controls[:, 0]
    np.testing.assert_allclose(
        model.predict(states, controls),
        a @ states + b @ controls,
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize("lag", [0, -1])
def test_explicit_pairs_still_require_positive_lag(lag):
    """Reject nonpositive lags even when snapshot pairs are explicit."""
    _, _, x, u, y = _system()
    with pytest.raises(ValueError, match="lag"):
        _model(lag=lag).fit(x, u, Y=y)


@pytest.mark.parametrize(
    "invalid_input",
    [
        "target_rows",
        "target_columns",
        "too_few_controls",
        "too_many_controls",
        "control_dimensions",
        "scalar_controls",
        "b_rows",
        "b_columns",
        "b_dimensions",
    ],
)
def test_paired_fit_rejects_misaligned_shapes(invalid_input):
    """Reject mismatched snapshot, control, and known-matrix dimensions."""
    _, b, x, u, y = _system()
    if invalid_input == "target_rows":
        y = y[:2]
    elif invalid_input == "target_columns":
        y = y[:, :-1]
    elif invalid_input == "too_few_controls":
        u = u[:, :-1]
    elif invalid_input == "too_many_controls":
        u = np.column_stack((u, u[:, 0]))
    elif invalid_input == "control_dimensions":
        u = u[:, :, None]
    elif invalid_input == "scalar_controls":
        u = 1.0
    elif invalid_input == "b_rows":
        b = b[:2]
    elif invalid_input == "b_columns":
        b = b[:, :1]
    else:
        b = b[:, 0]
    with pytest.raises(ValueError):
        _model().fit(x, u, b, Y=y)


def test_paired_fit_accepts_known_control_matrix_as_list():
    """Accept array-like control inputs and a supplied control matrix."""
    a, b, x, u, y = _system()
    model = _model().fit(x, u.tolist(), b.tolist(), Y=y)
    _assert_operator(model, a, b)


def test_fit_and_prediction_preserve_input_arrays():
    """Keep the supplied training and prediction arrays unchanged."""
    _, b, x, u, y = _system()
    originals = [value.copy() for value in (b, x, u, y)]
    model = _model().fit(x, u, b, Y=y)
    model.predict(x, u)
    for actual, original in zip((b, x, u, y), originals):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize(
    "first_paired,second_paired", [(True, False), (False, True), (True, True)]
)
def test_refit_resets_pair_metadata_controls_and_mode_mask(
    first_paired, second_paired
):
    """Reset pair metadata, controls, and active modes on subsequent fits."""
    a, b, x, u, y = _system()
    model = _model()
    model.fit(
        x if first_paired else _trajectory(a, b, u),
        u,
        b,
        Y=y if first_paired else None,
    )
    model.modes_activation_bitmask = np.zeros(len(model.eigs), dtype=bool)

    a, b, x, u, y = _system(seed=211, n_controls=1, n_samples=27)
    snapshots = x if second_paired else _trajectory(a, b, u)
    model.fit(snapshots, u, Y=y if second_paired else None)
    np.testing.assert_array_equal(
        model.modes_activation_bitmask,
        np.ones_like(model.modes_activation_bitmask, dtype=bool),
    )
    np.testing.assert_array_equal(model.snapshots, snapshots)
    _assert_operator(model, a, b)
    if second_paired:
        np.testing.assert_array_equal(model.snapshots_y, y)
        with pytest.raises(RuntimeError, match="predict"):
            _ = model.dynamics
    else:
        np.testing.assert_equal(model.snapshots_y is None, True)
        np.testing.assert_equal(model.dynamics.shape[1], snapshots.shape[1])
        np.testing.assert_allclose(
            model.reconstructed_data(), snapshots, rtol=1e-10, atol=1e-10
        )


def test_prediction_respects_mode_mask_and_retains_control_forcing():
    """Mask state evolution while retaining the full control forcing."""
    a = np.diag([0.2, 0.6, 0.9])
    b = np.array([[1], [-0.3], [0.5]])
    rng = np.random.default_rng(321)
    x, u = rng.normal(size=(3, 40)), rng.normal(size=(1, 40))
    model = _model().fit(x, u, Y=a @ x + b @ u)
    model.modes_activation_bitmask = np.isclose(model.eigs, 0.6)
    np.testing.assert_allclose(
        model.predict(x, u),
        np.diag([0, 0.6, 0]) @ x + b @ u,
        rtol=1e-10,
        atol=1e-10,
    )
    model.modes_activation_bitmask = np.zeros(3, dtype=bool)
    np.testing.assert_allclose(
        model.predict(x, u), b @ u, rtol=1e-10, atol=1e-10
    )
    model.modes_activation_bitmask = np.ones(3, dtype=bool)
    _assert_operator(model, a, b)
