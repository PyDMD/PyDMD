"""Utilities module."""

import warnings

import numpy as np

from pydmd.linalg import build_linalg_module, cast_as_array, is_array


def compute_rank(X, svd_rank=0):
    """
    Rank computation for the truncated Singular Value Decomposition.
    :param numpy.ndarray X: the matrix to decompose.
    :param svd_rank: the rank for the truncation; If 0, the method computes
        the optimal rank and uses it for truncation; if positive interger,
        the method uses the argument for the truncation; if float between 0
        and 1, the rank is the number of the biggest singular values that
        are needed to reach the 'energy' specified by `svd_rank`; if -1,
        the method does not compute truncation. Default is 0.
    :type svd_rank: int or float
    :return: the computed rank truncation.
    :rtype: int
    References:
    Gavish, Matan, and David L. Donoho, The optimal hard threshold for
    singular values is, IEEE Transactions on Information Theory 60.8
    (2014): 5040-5053.
    """
    linalg_module = build_linalg_module(X)
    U, s, _ = linalg_module.svd(X, full_matrices=False)

    def omega(x):
        return 0.56 * x**3 - 0.95 * x**2 + 1.82 * x + 1.43

    if svd_rank == 0:
        small, big = sorted(X.shape[-2:])
        beta = small / big
        tau = linalg_module.median(s) * omega(beta)
        rank = (s > tau).sum()
        if rank == 0:
            warnings.warn(
                "SVD optimal rank is 0. The largest singular values are "
                "indistinguishable from noise. Setting rank truncation to 1.",
                RuntimeWarning,
            )
            rank = 1
    elif 0 < svd_rank < 1:
        cumulative_energy = (s**2 / (s**2).sum()).cumsum(0)
        rank = linalg_module.searchsorted(cumulative_energy, svd_rank) + 1
    elif svd_rank >= 1 and isinstance(svd_rank, int):
        rank = min(svd_rank, U.shape[-1])
    else:
        rank = X.shape[-1]

    return rank


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
    VV = linalg_module.dot(V[:tlsq_rank].conj().T, V[:tlsq_rank])
    return linalg_module.dot(X, VV), linalg_module.dot(Y, VV)


def compute_optimal_svd_rank(X):
    """
    Rank computation for the truncated Singular Value Decomposition.
    :param numpy.ndarray X: the matrix to decompose.
    :type svd_rank: int or float
    :return: the computed rank truncation.
    :rtype: int
    References:
    Gavish, Matan, and David L. Donoho, The optimal hard threshold for
    singular values is, IEEE Transactions on Information Theory 60.8
    (2014): 5040-5053.
    """
    return compute_svd(X)[0].shape[-1]


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
            raise ValueError(
                "Automatic SVD rank selection not available in tensorized DMD"
            )

    linalg_module = build_linalg_module(X)

    U, s, V = linalg_module.svd(X, full_matrices=False)
    V = V.conj().swapaxes(-1, -2)

    rank = compute_rank(X, svd_rank)
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

    cond_number = linalg_module.cond(snapshots)
    if (
        isinstance(cond_number, float)
        or hasattr(cond_number, "ndim")
        and cond_number.ndim == 0
    ):
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


def nan_average(arr, weights):
    """
    Cancels axis -2 by averaging all the samples on axis -3 using the given
    weights.
    """
    if weights.ndim != 1:
        raise ValueError("Expected 1D weights")

    linalg_module = build_linalg_module(arr)

    if arr.ndim == 4:
        arr0, arr1, _, arr3 = arr.shape
    else:
        arr0 = 0
        arr1, _, arr3 = arr.shape
    repeated_weights = linalg_module.repeat(weights[None], arr1, 0)
    repeated_weights = linalg_module.repeat(
        repeated_weights[..., None], arr3, 2
    )
    if arr.ndim == 4:
        repeated_weights = linalg_module.repeat(repeated_weights[None], arr0, 0)

    non_normalized_mean = linalg_module.nansum(arr * repeated_weights, axis=-2)

    weights_sum = linalg_module.nansum(repeated_weights, axis=-2)
    # avoid divide by zero
    weights_sum[weights_sum == 0.0] = 1
    return non_normalized_mean / weights_sum
