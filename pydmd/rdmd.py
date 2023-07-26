"""
Derived module from cdmd.py for Randomized DMD

Reference:
N. Benjamin Erichson, Lionel Mathelin, J. Nathan Kutz, Steven L. Brunton.
Randomized dynamic mode decomposition. SIAM Journal on Applied Dynamical
Systems, 18, 2019.
"""

import numpy as np

from .cdmd import CDMD
from .utils import compute_rank


class RDMD(CDMD):
    """
    Randomized Dynamic Mode Decomposition

    :param test_matrix: The random test matrix that will be used when executing
        the Randomized QB Decomposition. If not provided, the `svd_rank` and
        `oversampling` parameters will be used to compute the random matrix.
    :type test_matrix: numpy.ndarray
    :param oversampling: Number of additional samples (beyond the desired rank)
        to use when computing the random test matrix. Note that values {5,10}
        tend to be sufficient.
    :type oversampling: int
    :param power_iters: Number of power iterations to perform when executing
        the Randomized QB Decomposition. Note that values {1,2} often lead to
        considerable improvements.
    :type power_iters: int
    """

    def __init__(
        self,
        test_matrix=None,
        oversampling=10,
        power_iters=2,
        svd_rank=0,
        tlsq_rank=0,
        opt=False,
        rescale_mode=None,
        forward_backward=False,
        sorted_eigs=False,
        tikhonov_regularization=None,
    ):
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
        self._oversampling = oversampling
        self._power_iters = power_iters
        self._test_matrix = test_matrix

    def _compress_snapshots(self):
        """
        Private method that compresses the snapshot matrix X by projecting X
        onto a near-optimal orthonormal basis for the range of X computed via
        the Randomized QB Decomposition.

        :return: the compressed snapshots
        :rtype: numpy.ndarray
        """
        # Define the random test matrix if not provided.
        if self._test_matrix is None:
            m = self.snapshots.shape[-1]
            r = compute_rank(self.snapshots, self._svd_rank)
            self._test_matrix = np.random.randn(m, r + self._oversampling)

        # Compute sampling matrix.
        Y = self.snapshots.dot(self._test_matrix)

        # Perform power iterations.
        for _ in range(self._power_iters):
            Q = np.linalg.qr(Y)[0]
            Z = np.linalg.qr(self.snapshots.conj().T.dot(Q))[0]
            Y = self.snapshots.dot(Z)

        # Orthonormalize the sampling matrix.
        Q = np.linalg.qr(Y)[0]

        # Project the snapshot matrix onto the smaller space.
        B = Q.conj().T.dot(self.snapshots)

        # Save the compression matrix.
        self._compression_matrix = Q.conj().T

        return B
