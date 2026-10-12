"""Regression tests for selections on previously filtered DMD instances."""

from copy import deepcopy

import numpy as np
import pytest

from pydmd import DMD
from pydmd.dmd_modes_tuner import ModesTuner, select_modes


def _fit_selection_test_dmd():
    snapshots = np.array([0.8, 0.95, 1.05, 1.2])[:, None] ** np.arange(8)
    return DMD(svd_rank=4, sorted_eigs="abs").fit(snapshots)


@pytest.mark.parametrize("in_place", [True, False])
@pytest.mark.parametrize(
    "initial_mask",
    [
        [True, True, True, True],
        [False, True, True, True],
        [True, False, True, True],
        [False, True, False, True],
    ],
)
def test_select_modes_respects_existing_mask(initial_mask, in_place):
    """Selection indexes refer to the active modes of the input instance."""
    dmd = _fit_selection_test_dmd()
    dmd.modes_activation_bitmask = np.array(initial_mask)
    original_eigs = dmd.eigs.copy()
    original_modes = dmd.modes.copy()
    original_amplitudes = dmd.amplitudes.copy()
    selected = np.arange(len(original_eigs)) % 2 == 1
    expected_mask = np.array(initial_mask)
    expected_mask[np.flatnonzero(expected_mask)[~selected]] = False
    expected = deepcopy(dmd)
    expected.modes_activation_bitmask = expected_mask

    result, cut_indexes = select_modes(
        dmd,
        lambda model: np.arange(len(model.eigs)) % 2 == 1,
        in_place=in_place,
        return_indexes=True,
    )

    np.testing.assert_array_equal(cut_indexes, np.flatnonzero(~selected))
    np.testing.assert_array_equal(
        result.modes_activation_bitmask, expected_mask
    )
    np.testing.assert_allclose(result.eigs, original_eigs[selected])
    np.testing.assert_allclose(result.modes, original_modes[:, selected])
    np.testing.assert_allclose(result.amplitudes, original_amplitudes[selected])
    np.testing.assert_allclose(
        result.reconstructed_data, expected.reconstructed_data
    )
    if not in_place:
        np.testing.assert_array_equal(
            dmd.modes_activation_bitmask, initial_mask
        )
        np.testing.assert_allclose(dmd.eigs, original_eigs)
        np.testing.assert_allclose(dmd.modes, original_modes)
        np.testing.assert_allclose(dmd.amplitudes, original_amplitudes)


@pytest.mark.parametrize("in_place", [True, False])
def test_modes_tuner_consecutive_selections(in_place):
    """Chained tuning retains the same modes as the combined criterion."""
    dmd = _fit_selection_test_dmd()
    original = deepcopy(dmd)
    expected = deepcopy(dmd)
    select_modes(expected, lambda model: np.abs(model.eigs) > 1.0)

    tuner = ModesTuner(dmd, in_place=in_place)
    tuner.select(lambda model: np.abs(model.eigs) > 0.9)
    tuner.select(lambda model: np.abs(model.eigs) > 1.0)

    np.testing.assert_array_equal(
        tuner.get().modes_activation_bitmask, expected.modes_activation_bitmask
    )
    np.testing.assert_allclose(tuner.get().eigs, expected.eigs)
    np.testing.assert_allclose(
        tuner.get().reconstructed_data, expected.reconstructed_data
    )
    if not in_place:
        np.testing.assert_allclose(dmd.eigs, original.eigs)
        np.testing.assert_allclose(
            dmd.reconstructed_data, original.reconstructed_data
        )


@pytest.mark.parametrize("select_all", [True, False])
@pytest.mark.parametrize("in_place", [True, False])
def test_select_modes_all_or_none_with_existing_mask(select_all, in_place):
    """Keeping all or none preserves the original inactive modes."""
    dmd = _fit_selection_test_dmd()
    initial_mask = np.array([False, True, True, True])
    dmd.modes_activation_bitmask = initial_mask.copy()
    expected = deepcopy(dmd)
    expected.modes_activation_bitmask = initial_mask & select_all

    result, cut_indexes = select_modes(
        dmd,
        lambda model: np.full(len(model.eigs), select_all),
        in_place=in_place,
        return_indexes=True,
    )

    np.testing.assert_array_equal(
        result.modes_activation_bitmask, initial_mask & select_all
    )
    np.testing.assert_array_equal(cut_indexes, [] if select_all else [0, 1, 2])
    np.testing.assert_allclose(
        result.reconstructed_data, expected.reconstructed_data
    )
    if not in_place:
        np.testing.assert_array_equal(
            dmd.modes_activation_bitmask, initial_mask
        )
