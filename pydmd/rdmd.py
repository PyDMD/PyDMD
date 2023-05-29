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

    :param rand_mat: The random test matrix that will be used when executing
        the Randomized QB Decomposition. If not provided, the `svd_rank` and
        `oversampling` parameters will be used to compute the random matrix.
    :type rand_mat: numpy.ndarray
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
        rand_mat=None,
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
        self._rand_mat = rand_mat

    def _compress_snapshots(self):
        """
        Private method that compresses the snapshot matrix X by projecting X
        onto a near-optimal orthonormal basis for the range of X computed via
        the Randomized QB Decomposition.

        :return: the compressed snapshots
        :rtype: numpy.ndarray
        """
        linalg_module = build_linalg_module(self.snapshots)

        # Define the random test matrix if not provided.
        if self._rand_mat is None:
            m = self.snapshots.shape[-1]
            r = compute_rank(self.snapshots, self._svd_rank)
            self._rand_mat = np.random.randn(m, r + self._oversampling)
            self._rand_mat = linalg_module.to(self.snapshots, self._rand_mat)

        # Compute sampling matrix.
        Y = linalg_module.dot(self.snapshots, self._rand_mat)

        # Perform power iterations.
        for _ in range(self._power_iters):
            Q = linalg_module.qr_reduced(Y)[0]
            Z = linalg_module.qr_reduced(
                linalg_module.dot(self.snapshots.conj().swapaxes(-1, -2), Q)
            )[0]
            Y = linalg_module.dot(self.snapshots, Z)

        # Orthonormalize the sampling matrix.
        Q = linalg_module.qr_reduced(Y)[0]

        # Project the snapshot matrix onto the smaller space.
        B = linalg_module.dot(Q.conj().swapaxes(-1, -2), self.snapshots)

        # Save the compression matrix.
        self._compression_matrix = Q.conj().swapaxes(-1, -2)

        return B
