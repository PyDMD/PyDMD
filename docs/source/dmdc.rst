DMD with control
=================

.. currentmodule:: pydmd.dmdc

.. automodule:: pydmd.dmdc

Training with separate snapshot pairs
-------------------------------------

``DMDc.fit(X, I, Y=Y)`` learns the controlled model
:math:`Y \approx AX + BI` from aligned columns of state snapshots ``X``, target
snapshots ``Y``, and control inputs ``I``. Each column of ``Y`` must be the
successor of the corresponding column of ``X`` under the corresponding input
in ``I``. The columns do not have to form one continuous time series.

This is useful when combining independent experiments: concatenate their
valid transitions, without creating artificial transitions from the end of
one experiment to the start of the next. The same interface also supports
shuffling or selecting transitions, provided that the same column selection
is applied to all three matrices.

For example, generate two independent trajectories of a controlled linear
system, then collect their transitions:

.. testcode:: dmdc-pairs

    import numpy as np
    from pydmd import DMDc

    A = np.array([[0.8, 0.1], [0.0, 0.6]])
    B = np.array([[0.4], [0.2]])
    initial_states = [np.array([1.0, 0.0]), np.array([-0.5, 1.0])]
    controls = [
        np.array([[0.0, 1.0, -1.0, 0.5]]),
        np.array([[1.0, 0.0, 0.5, -0.5]]),
    ]
    trajectories = []
    for initial_state, inputs in zip(initial_states, controls):
        states = [initial_state]
        for control in inputs.T:
            states.append(A @ states[-1] + B @ control)
        trajectories.append(np.column_stack(states))

    X = np.hstack([trajectory[:, :-1] for trajectory in trajectories])
    Y = np.hstack([trajectory[:, 1:] for trajectory in trajectories])
    I = np.hstack(controls)

Fit the model using shuffled transitions. Without a supplied ``B``, both the
state evolution and the effect of the controls are learned from the data:

.. testcode:: dmdc-pairs

    order = [2, 6, 0, 4, 7, 1, 5, 3]
    model = DMDc(svd_rank=-1).fit(
        X[:, order], I[:, order], Y=Y[:, order]
    )
    np.testing.assert_allclose(model.B, B, atol=1e-12)
    np.testing.assert_allclose(model.predict(X, I), Y, atol=1e-12)

Use ``predict`` to evaluate a new state and control, or a batch of independent
states and controls. A vector input returns a vector; a matrix input returns
one predicted state per column:

.. testcode:: dmdc-pairs

    new_state = np.array([0.25, -0.5])
    new_control = np.array([0.3])
    prediction = model.predict(new_state, new_control)
    np.testing.assert_allclose(
        prediction, A @ new_state + B @ new_control, atol=1e-12
    )

    # A known control matrix can still be the third positional argument.
    known_control_model = DMDc(svd_rank=-1).fit(X, I, B, Y=Y)
    np.testing.assert_allclose(
        known_control_model.predict(X, I), Y, atol=1e-12
    )

Input shapes and time interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* For training with ``Y``, the normalized ``X`` and ``Y`` matrices must both
  have shape ``(n_states, n_pairs)``. Control inputs must have shape
  ``(n_controls, n_pairs)``; a one-dimensional series is accepted for a single
  control channel. As with the other DMD classes, training snapshots can also
  be supplied as an iterable of individual snapshots.
* ``predict`` accepts a state vector of shape ``(n_states,)`` or a matrix of
  shape ``(n_states, n_samples)``. Controls must match the number of channels
  and samples. One-dimensional controls are accepted for a single channel
  or a single sample; a scalar is accepted only for one channel and one
  sample. Controls are not broadcast across samples.
* When ``Y`` is supplied, its columns already contain the target states at
  the chosen ``lag``. No additional shift or slicing is applied. ``predict``
  applies this learned transition once; it does not raise the learned
  operator to another power of ``lag``.
* Unordered snapshot pairs do not define a shared time series. Consequently,
  ``opt=True``, ``dynamics``, and ``reconstructed_data()`` are unavailable for
  a model fitted with explicit ``Y``. Use ``predict(X, I)`` for independent
  transition predictions. With ``opt=False`` or an integer ``opt``, amplitudes
  describe the selected input snapshot.

Existing calls without ``Y`` continue to split one trajectory according to
``lag`` and support their original reconstruction behavior. ``predict`` can
also be used with these models.

.. autoclass:: DMDc
	:members:
	:private-members:
	:undoc-members:
	:show-inheritance:
	:noindex:

