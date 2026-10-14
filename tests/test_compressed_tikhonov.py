"""Compressed Tikhonov regularization uses the current SVD input norm."""

import numpy as np
import pytest

from pydmd import CDMD, RDMD


def snapshots(dtype, scale):
    """An analytic three-component exponential trajectory."""
    if dtype == np.complex128:
        values = np.array([0.85 + 0.15j, 0.45 - 0.25j, -0.35 + 0.1j])
    else:
        values = np.array([0.92, 0.53, -0.41])
    amplitudes = scale * np.array([1.0, 1.7, 0.8])
    return amplitudes[:, None] * values.astype(dtype)[:, None] ** np.arange(8)


def estimator(method, regularization):
    """Build a deterministic compressed or randomized estimator."""
    if method == "randomized":
        return RDMD(
            svd_rank=-1, seed=42, tikhonov_regularization=regularization
        )
    compression = (
        np.eye(3)
        if method == "identity"
        else np.array([[1.0, 0.0, 0.2], [0.0, 1.0, -0.3]])
    )
    return CDMD(
        svd_rank=-1,
        compression_matrix=compression,
        tikhonov_regularization=regularization,
    )


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("method", ["identity", "compressed", "randomized"])
@pytest.mark.parametrize("regularization", [None, 0.0, 0.001, 0.1])
@pytest.mark.parametrize("refit", [False, True])
def test_compressed_tikhonov_norm(dtype, method, regularization, refit):
    """Match the regularized SVD expression and a fresh fit after reuse."""
    model = estimator(method, regularization)
    if refit:
        model.fit(snapshots(dtype, scale=1.0))
    data = snapshots(dtype, scale=3.0)
    model.fit(data)
    x = model.compression_matrix @ data[:, :-1]
    y = model.compression_matrix @ data[:, 1:]
    u, singular_values, vh = np.linalg.svd(x, full_matrices=False)
    penalty = (
        0.0 if regularization is None else regularization * np.linalg.norm(x)
    )
    inverse = singular_values / (singular_values**2 + penalty)
    expected = (u.conj().T @ y @ vh.conj().T) * inverse

    np.testing.assert_allclose(
        model.operator.as_numpy_array, expected, atol=1e-10
    )
    fresh = estimator(method, regularization).fit(data)
    np.testing.assert_allclose(
        model.reconstructed_data,
        fresh.reconstructed_data,
        rtol=1e-9,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.sort_complex(model.eigs), np.sort_complex(fresh.eigs), atol=1e-10
    )
