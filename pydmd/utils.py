"""Utilities module."""

import warnings
from numbers import Number
from typing import Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _svht(sigma_svd: np.ndarray, rows: int, cols: int) -> int:
    """
    Singular Value Hard Threshold.

    :param sigma_svd: Singual values computed by SVD
    :type sigma_svd: np.ndarray
    :param rows: Number of rows of original data matrix.
    :type rows: int
    :param cols: Number of columns of original data matrix.
    :type cols: int
    :return: Computed rank.
    :rtype: int

    References:
    Gavish, Matan, and David L. Donoho, The optimal hard threshold for
    singular values is, IEEE Transactions on Information Theory 60.8
    (2014): 5040-5053.
    https://ieeexplore.ieee.org/document/6846297
    """
    beta = np.divide(*sorted((rows, cols)))
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    tau = np.median(sigma_svd) * omega
    rank = np.sum(sigma_svd > tau)

    if rank == 0:
        warnings.warn(
            "SVD optimal rank is 0. The largest singular values are "
            "indistinguishable from noise. Setting rank truncation to 1.",
            RuntimeWarning,
        )
        rank = 1

    return rank


def _compute_rank(
    sigma_svd: np.ndarray, rows: int, cols: int, svd_rank: Number
) -> int:
    """
    Rank computation for the truncated Singular Value Decomposition.

    :param sigma_svd: 1D singular values of SVD.
    :type sigma_svd: np.ndarray
    :param rows: Number of rows of original matrix.
    :type rows: int
    :param cols: Number of columns of original matrix.
    :type cols: int
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
    if svd_rank == 0:
        rank = _svht(sigma_svd, rows, cols)
    elif 0 < svd_rank < 1:
        cumulative_energy = np.cumsum(sigma_svd**2 / (sigma_svd**2).sum())
        rank = np.searchsorted(cumulative_energy, svd_rank) + 1
    elif svd_rank >= 1 and isinstance(svd_rank, int):
        rank = min(svd_rank, sigma_svd.size)
    else:
        rank = min(rows, cols)

    return rank


def compute_rank(X: np.ndarray, svd_rank: Number = 0) -> int:
    """
    Rank computation for the truncated Singular Value Decomposition.

    :param X: the matrix to decompose.
    :type X: np.ndarray
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
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    return _compute_rank(s, X.shape[0], X.shape[1], svd_rank)


def compute_tlsq(
    X: np.ndarray, Y: np.ndarray, tlsq_rank: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Total Least Square.

    :param X: the first matrix;
    :type X: np.ndarray
    :param Y: the second matrix;
    :type Y: np.ndarray
    :param tlsq_rank: the rank for the truncation; If 0, the method
        does not compute any noise reduction; if positive number, the
        method uses the argument for the SVD truncation used in the TLSQ
        method.
    :type tlsq_rank: int
    :return: the denoised matrix X, the denoised matrix Y
    :rtype: Tuple[np.ndarray, np.ndarray]

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


def compute_svd(
    X: np.ndarray, svd_rank: Number = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Truncated Singular Value Decomposition.

    :param X: the matrix to decompose.
    :type X: np.ndarray
    :param svd_rank: the rank for the truncation; If 0, the method computes
        the optimal rank and uses it for truncation; if positive interger,
        the method uses the argument for the truncation; if float between 0
        and 1, the rank is the number of the biggest singular values that
        are needed to reach the 'energy' specified by `svd_rank`; if -1,
        the method does not compute truncation. Default is 0.
    :type svd_rank: int or float
    :return: the truncated left-singular vectors matrix, the truncated
        singular values array, the truncated right-singular vectors matrix.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]

    References:
    Gavish, Matan, and David L. Donoho, The optimal hard threshold for
    singular values is, IEEE Transactions on Information Theory 60.8
    (2014): 5040-5053.
    """
    U, s, V = np.linalg.svd(X, full_matrices=False)
    rank = _compute_rank(s, X.shape[0], X.shape[1], svd_rank)
    V = V.conj().T

    U = U[:, :rank]
    V = V[:, :rank]
    s = s[:rank]

    return U, s, V


def pseudo_hankel_matrix(X: np.ndarray, d: int) -> np.ndarray:
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
