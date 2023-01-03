"""Utilities module."""

import warnings

import numpy as np

from .linalg import build_linalg_module, cast_as_array, is_array


def compute_tlsq(X, Y, tlsq_rank):
    """
    Compute Total Least Square.

    :param numpy.ndarray X: the first matrix;
    :param numpy.ndarray Y: the second matrix;
    :param int tlsq_rank: the rank for the truncation; If 0, the method
        does not compute any noise reduction; if positive number, the
        method uses the argument for the SVD truncation used in the TLSQ
        method.
    :return: the denoised matrix X, the denoised matrix Y
    :rtype: numpy.ndarray, numpy.ndarray

    References:
    https://arxiv.org/pdf/1703.11004.pdf
    https://arxiv.org/pdf/1502.03854.pdf
    """
    # Do not perform tlsq
    if tlsq_rank == 0:
        return X, Y

    linalg_module = build_linalg_module(X)
    concatenated = linalg_module.cat((X, Y), axis=-2)
    _, _, V = linalg_module.svd(concatenated)
    VV = linalg_module.dot(V[..., :tlsq_rank].conj().T, V[..., :tlsq_rank])
    return linalg_module.dot(X, VV), linalg_module.dot(Y, VV)


def compute_svd(X, svd_rank=0):
    """
    Truncated Singular Value Decomposition.

    :param numpy.ndarray X: the matrix to decompose.
    :param svd_rank: the rank for the truncation; If 0, the method computes
        the optimal rank and uses it for truncation; if positive interger,
        the method uses the argument for the truncation; if float between 0
        and 1, the rank is the number of the biggest singular values that
        are needed to reach the 'energy' specified by `svd_rank`; if -1,
        the method does not compute truncation. Default is 0.
    :type svd_rank: int or float
    :return: the truncated left-singular vectors matrix, the truncated
        singular values array, the truncated right-singular vectors matrix.
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray

    References:
    Gavish, Matan, and David L. Donoho, The optimal hard threshold for
    singular values is, IEEE Transactions on Information Theory 60.8
    (2014): 5040-5053.
    """
    if X.ndim > 2:
        if svd_rank == 0 or not isinstance(svd_rank, int):
            raise ValueError("Automatic SVD rank selection not available in batched DMD")

    linalg_module = build_linalg_module(X)

    U, s, V = linalg_module.svd(X, full_matrices=False)
    V = V.conj().swapaxes(-1, -2)

    def omega(x):
        return 0.56 * x**3 - 0.95 * x**2 + 1.82 * x + 1.43

    if svd_rank == 0:
        small, big = sorted(X.shape)
        beta = small / big
        tau = linalg_module.median(s) * omega(beta)
        rank = (s > tau).sum()
        if rank == 0:
            warnings.warn(
                'SVD optimal rank is 0. The largest singular values are '
                'indistinguishable from noise. Setting rank truncation to 1.',
                RuntimeWarning
            )
            rank = 1
    elif 0 < svd_rank < 1:
        cumulative_energy = (s**2 / (s**2).sum()).cumsum(0)
        rank = linalg_module.searchsorted(cumulative_energy, svd_rank) + 1
    elif svd_rank >= 1 and isinstance(svd_rank, int):
        rank = min(svd_rank, U.shape[-1])
    else:
        rank = X.shape[-1]

    return U[..., :rank], s[..., :rank], V[..., :rank]


def prepare_snapshots(X):
    snapshots = cast_as_array(X)

    linalg_module = build_linalg_module(snapshots)
    snapshots = linalg_module.atleast_2d(snapshots)
    if snapshots.ndim < 2:
        raise ValueError("Expected at least 2D array.")
    if snapshots.ndim > 2 and isinstance(snapshots, np.ndarray):
        raise ValueError("Batched DMD not supported in NumPy")

    # when snapshots are wrapped in a list each member of the list is
    # a snapshot
    if not is_array(X) and snapshots.ndim == 2:
        snapshots = snapshots.T

    # check condition number of the data passed in
    cond_number = linalg_module.cond(snapshots)
    if isinstance(cond_number, float) or hasattr(cond_number, "ndim") and cond_number.ndim == 0:
        max_cond_number = float(cond_number)
    else:
        max_cond_number = max(cond_number)

    if max_cond_number > 10e4:
        warnings.warn(
            f"Input data matrix X has condition number {max_cond_number}. "
            """Consider preprocessing data, passing in augmented data
matrix, or regularization methods."""
        )

    return snapshots