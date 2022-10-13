"""
Derived module from cdmd.py for Randomized DMD
References:
- N. Benjamin Erichson, Lionel Mathelin, J. Nathan Kutz, and Steven L. Brunton. Randomized
dynamic mode decomposition. SIAM Journal on Applied Dynamical Systems, 18, 2019.
"""

import numpy as np
from .cdmd import CDMD, CDMDOperator
from .utils import compute_tlsq, compute_svd

def rqb(X, k, p, q):
    """
    Randomized QB Decomposition.
    
    :param numpy.ndarray X: the matrix to decompose
    :param int k: target rank 
    :param int p: oversampling parameter 
    :param int q: number of power iterations 
    
    :return: a near-optimal orthonormal basis Q for the range of the matrix X, 
        and the projection B of the matrix X onto the low-dimensional space Q.
    :rtype: numpy.ndarray, numpy.ndarray

    References:
    N. Benjamin Erichson, Lionel Mathelin, J. Nathan Kutz, and Steven L. Brunton. 
    Randomized dynamic mode decomposition. SIAM Journal on Applied Dynamical Systems, 18, 2019.
    """
    n, m = X.shape
    l = k + p # slight oversampling
    Omega = np.random.randn(m, l) # generate random test matrix 
    Y = X @ Omega # compute sampling matrix 
    for j in range(q): # power iterations 
        Q, _ = np.linalg.qr(Y)
        Z, _ = np.linalg.qr(X.conj().T @ Q)
        Y = X @ Z
    Q, _ = np.linalg.qr(Y)
    B = Q.conj().T @ X
    return Q, B

class RDMD(CDMD):
    """
    Randomized Dynamic Mode Decomposition
    :param int p: Randomized QB decomposition oversampling parameter 
    :param int q: Randomized QB decomposition power iterations 
    """
    
    def __init__(self, svd_rank, p=10, q=2, tlsq_rank=0, opt=False, rescale_mode=None, 
                 forward_backward=False, sorted_eigs=False, tikhonov_regularization=None):
        super().__init__(
            svd_rank=svd_rank,
            tlsq_rank=tlsq_rank,
            compression_matrix=None,
            opt=opt,
            rescale_mode=rescale_mode,
            forward_backward=forward_backward,
            sorted_eigs=sorted_eigs,
            tikhonov_regularization=tikhonov_regularization,
        )
        self._svd_rank = svd_rank
        self._p = p 
        self._q = q
        
    def fit(self, X):
        """
        Compute the Dynamic Modes Decomposition to the input data.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        """
        self._snapshots, self._snapshots_shape = self._col_major_2darray(X)
        Q, compressed_snapshots = rqb(self._snapshots, self._svd_rank, self._p, self._q)
        self._compression_matrix = Q.conj().T

        n_samples = compressed_snapshots.shape[1]
        X = compressed_snapshots[:, :-1]
        Y = compressed_snapshots[:, 1:]

        X, Y = compute_tlsq(X, Y, self.tlsq_rank)
        self.operator.compute_operator(X, Y, self._snapshots[:, 1:])

        # Default timesteps
        self._set_initial_time_dictionary(
            {"t0": 0, "tend": n_samples - 1, "dt": 1}
        )

        self._b = self._compute_amplitudes()

        return self