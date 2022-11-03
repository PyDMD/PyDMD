"""
Derived module from cdmd.py for Randomized DMD

Reference:
N. Benjamin Erichson, Lionel Mathelin, J. Nathan Kutz, Steven L. Brunton.
Randomized dynamic mode decomposition. SIAM Journal on Applied Dynamical
Systems, 18, 2019.
"""

import numpy as np
from .cdmd import CDMD

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
    elif 0 < svd_rank < 1:
        cumulative_energy = np.cumsum(s**2 / (s**2).sum())
        rank = np.searchsorted(cumulative_energy, svd_rank) + 1
    elif svd_rank >= 1 and isinstance(svd_rank, int):
        rank = min(svd_rank, U.shape[1])
    else:
        rank = X.shape[1]

    return rank


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
        oversampling=10,
        power_iters=2,
        svd_rank=0,
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


    def _compress_snapshots(self):
        """
        Private method that compresses the snapshot matrix X by projecting X
        onto a near-optimal orthonormal basis for the range of X computed via
        the Randomized QB Decomposition.
        :return: the compressed snapshots
        :rtype: numpy.ndarray
        """
        # Perform the Randomized QB Decomposition
        m = self._snapshots.shape[1]

        # Compute the target rank
        self._svd_rank = compute_rank(self._snapshots, self._svd_rank)

        # Generate random test matrix (with slight oversampling)
        Omega = np.random.randn(m, self._svd_rank + self._oversampling)

        # Compute sampling matrix
        Y = self._snapshots.dot(Omega)

        # Perform power iterations
        for _ in range(self._power_iters):
            Q = np.linalg.qr(Y)[0]
            Z = np.linalg.qr(self._snapshots.conj().T.dot(Q))[0]
            Y = self._snapshots.dot(Z)

        # Orthonormalize the sampling matrix
        Q = np.linalg.qr(Y)[0]

        # Project the snapshot matrix onto the smaller space
        B = Q.conj().T.dot(self._snapshots)

        # Save the compression matrix
        self._compression_matrix = Q.conj().T

        return B
        