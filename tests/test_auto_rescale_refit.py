"""Automatic rescaling is recomputed when a DMD instance is fitted again."""

import numpy as np
import pytest

from pydmd import CDMD, DMD, RDMD


def snapshots(dimension, dtype, scale=1.0):
    """Snapshots of an exactly known diagonal linear dynamical system."""
    if dtype == np.complex128:
        eigenvalues = np.array([0.85 + 0.15j, 0.45 - 0.25j, -0.35 + 0.1j])
    else:
        eigenvalues = np.array([0.92, 0.53, -0.41])
    eigenvalues = eigenvalues[:dimension].astype(dtype)
    amplitudes = scale * np.array([1.0, 1.7, 0.8])[:dimension]
    return amplitudes[:, None] * eigenvalues[:, None] ** np.arange(8)


def assert_matches_fresh(actual, expected, data):
    """Compare public observables without relying on eigenvector phases."""
    np.testing.assert_allclose(
        np.sort_complex(actual.eigs), np.sort_complex(expected.eigs), atol=1e-10
    )
    actual_order = np.argsort(actual.eigs)
    expected_order = np.argsort(expected.eigs)
    np.testing.assert_allclose(
        np.linalg.norm(actual.modes[:, actual_order], axis=0),
        np.linalg.norm(expected.modes[:, expected_order], axis=0),
        rtol=1e-9,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        actual.reconstructed_data,
        expected.reconstructed_data,
        rtol=1e-9,
        atol=1e-10,
    )
    if isinstance(actual, DMD):
        np.testing.assert_allclose(
            actual.predict(data[:, :1]),
            expected.predict(data[:, :1]),
            rtol=1e-9,
            atol=1e-10,
        )


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("svd_rank", [-1, 2])
@pytest.mark.parametrize("dimensions", [(3, 1), (1, 3), (2, 2)])
def test_auto_rescale_refit(dtype, exact, svd_rank, dimensions):
    """Different ranks and same-rank data both get fresh automatic scaling."""
    first = snapshots(dimensions[0], dtype)
    second = snapshots(dimensions[1], dtype, scale=3.0)
    reused = DMD(svd_rank=svd_rank, exact=exact, rescale_mode="auto")
    reused.fit(first)
    reused.fit(second)
    fresh = DMD(svd_rank=svd_rank, exact=exact, rescale_mode="auto").fit(second)
    assert_matches_fresh(reused, fresh, second)
    if svd_rank == -1:
        np.testing.assert_allclose(reused.reconstructed_data, second, atol=1e-9)
        np.testing.assert_allclose(
            reused.predict(second[:, :-1]), second[:, 1:], atol=1e-9
        )


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("scaling", [None, np.array([0.5, 2.0])])
def test_explicit_rescaling_refit_control(dtype, exact, scaling):
    """Disabled and user-supplied scaling retain the existing behavior."""
    first = snapshots(2, dtype)
    second = snapshots(2, dtype, scale=3.0)
    reused = DMD(svd_rank=-1, exact=exact, rescale_mode=scaling)
    reused.fit(first)
    reused.fit(second)
    fresh = DMD(svd_rank=-1, exact=exact, rescale_mode=scaling).fit(second)
    assert_matches_fresh(reused, fresh, second)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("method", ["compressed", "randomized"])
@pytest.mark.parametrize("scaling", ["auto", None, np.array([0.5, 2.0])])
def test_compressed_auto_rescale_refit(dtype, method, scaling):
    """Compressed operators also resolve auto scaling on each fit."""
    first = snapshots(2, dtype)
    second = snapshots(2, dtype, scale=3.0)
    if method == "compressed":
        reused = CDMD(
            svd_rank=-1, rescale_mode=scaling, compression_matrix=np.eye(2)
        )
        fresh = CDMD(
            svd_rank=-1, rescale_mode=scaling, compression_matrix=np.eye(2)
        )
    else:
        reused = RDMD(svd_rank=-1, rescale_mode=scaling, seed=42)
        fresh = RDMD(svd_rank=-1, rescale_mode=scaling, seed=42)
    reused.fit(first)
    reused.fit(second)
    fresh.fit(second)
    assert_matches_fresh(reused, fresh, second)
    np.testing.assert_allclose(reused.reconstructed_data, second, atol=1e-9)
