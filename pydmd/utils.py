import numpy as np

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
