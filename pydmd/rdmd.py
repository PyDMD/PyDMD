"""
Derived module from cdmd.py for Randomized DMD

Reference:
N. Benjamin Erichson, Lionel Mathelin, J. Nathan Kutz, Steven L. Brunton.
Randomized dynamic mode decomposition. SIAM Journal on Applied Dynamical
Systems, 18, 2019.
"""

import numpy as np
from .cdmd import CDMD
from .utils import compute_tlsq

def randomized_qb(X, target_rank, oversampling, power_iters):
    """
    Randomized QB Decomposition.

    :param numpy.ndarray X: (n, m) matrix to decompose.
    :param int target_rank: Target rank << min(m, n) of the matrix X.
    :param int oversampling: Number of additional samples to use when
        computing the random test matrix.
        Note that oversampling = {5,10} is often sufficient.
    :param int power_iters: Number of power iterations to perform.
        Note that power_iters = {1,2} leads to considerable improvements.
    
    :return: a near-optimal orthonormal basis Q for the range of the matrix X,
        and the projection B of the matrix X onto the low-dimensional space Q.
    :rtype: (n, target_rank) numpy.ndarray, (target_rank, m) numpy.ndarray

    Reference:
    N. Benjamin Erichson, Lionel Mathelin, J. Nathan Kutz, Steven L. Brunton.
    Randomized dynamic mode decomposition. SIAM Journal on Applied Dynamical
    Systems, 18, 2019.
    """
    m = X.shape[1]

    # Generate random test matrix (with slight oversampling)
    Omega = np.random.randn(m, target_rank + oversampling)

    # Compute sampling matrix
    Y = X.dot(Omega)

    # Perform power iterations
    for j in range(power_iters):
        Q = np.linalg.qr(Y)[0]
        Z = np.linalg.qr(X.conj().T.dot(Q))[0]
        Y = X.dot(Z)

    # Orthonormalize the sampling matrix
    Q = np.linalg.qr(Y)[0]

    # Project the input matrix X onto the smaller space
    B = Q.conj().T.dot(X)

    return Q, B


class RDMD(CDMD):
    """
    Randomized Dynamic Mode Decomposition

    :param int oversampling: Number of additional samples to use when
        computing the random test matrix.
        Note that oversampling = {5,10} is often sufficient.
    :param int power_iters: Number of power iterations to perform.
        Note that power_iters = {1,2} leads to considerable improvements.
    """
    
    def __init__(
        self,
        svd_rank,
        oversampling=10,
        power_iters=2,
        tlsq_rank=0,
        opt=False,
        rescale_mode=None,
        forward_backward=False,
        sorted_eigs=False,
        tikhonov_regularization=None
    ):
        super().__init__(
            svd_rank=svd_rank,
            tlsq_rank=tlsq_rank,
            compression_matrix=None,
            opt=opt,
            rescale_mode=rescale_mode,
            forward_backward=forward_backward,
            sorted_eigs=sorted_eigs,
            tikhonov_regularization=tikhonov_regularization
        )
        self._svd_rank = svd_rank
        self._oversampling = oversampling
        self._power_iters = power_iters
        

    def fit(self, X):
        """
        Compute the Dynamic Mode Decomposition to the input data.

        :param X: the input snapshots
        :type X: numpy.ndarray or iterable
        """
        self._snapshots, self._snapshots_shape = self._col_major_2darray(X)

        # Use randomized QB decomposition for the data compression
        Q, compressed_snapshots = randomized_qb(self._snapshots,
                                                self._svd_rank,
                                                self._oversampling,
                                                self._power_iters)
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
