"""Utilities module."""

import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


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
    U, s, _ = np.linalg.svd(X, full_matrices=False)

    def omega(x):
        return 0.56 * x**3 - 0.95 * x**2 + 1.82 * x + 1.43

    if svd_rank == 0:
        beta = np.divide(*sorted(X.shape))
        tau = np.median(s) * omega(beta)
        rank = np.sum(s > tau)
        if rank == 0:
            warnings.warn(
                "SVD optimal rank is 0. The largest singular values are "
                "indistinguishable from noise. Setting rank truncation to 1.",
                RuntimeWarning,
            )
            rank = 1
    elif 0 < svd_rank < 1:
        cumulative_energy = np.cumsum(s**2 / (s**2).sum())
        rank = np.searchsorted(cumulative_energy, svd_rank) + 1
    elif svd_rank >= 1 and isinstance(svd_rank, int):
        rank = min(svd_rank, U.shape[1])
    else:
        rank = min(X.shape)

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

    V = np.linalg.svd(np.append(X, Y, axis=0), full_matrices=False)[-1]
    rank = min(tlsq_rank, V.shape[0])
    VV = V[:rank, :].conj().T.dot(V[:rank, :])

    return X.dot(VV), Y.dot(VV)


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
    rank = compute_rank(X, svd_rank)
    U, s, V = np.linalg.svd(X, full_matrices=False)
    V = V.conj().T

    U = U[:, :rank]
    V = V[:, :rank]
    s = s[:rank]

    return U, s, V


def pseudo_hankel_matrix(X: np.ndarray, d: int):
    """
    Arrange the snapshot in the matrix `X` into the (pseudo) Hankel
    matrix. The attribute `d` controls the number of snapshot from `X` in
    each snapshot of the Hankel matrix.

    :Example:

        >>> a = np.array([[1, 2, 3, 4, 5]])
        >>> _hankel_pre_processing(a, d=2)
        array([[1, 2, 3, 4],
               [2, 3, 4, 5]])
        >>> _hankel_pre_processing(a, d=4)
        array([[1, 2],
               [2, 3],
               [3, 4],
               [4, 5]])

        >>> a = np.array([1,2,3,4,5,6]).reshape(2,3)
        array([[1, 2, 3],
               [4, 5, 6]])
        >>> _hankel_pre_processing(a, d=2)
        array([[1, 2],
               [4, 5],
               [2, 3],
               [5, 6]])
    """
    return (
        sliding_window_view(X.T, (d, X.shape[0]))[:, 0]
        .reshape(X.shape[1] - d + 1, -1)
        .T
    )
