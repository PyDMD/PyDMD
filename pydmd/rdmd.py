"""
Derived module from cdmd.py for Randomized DMD

Reference:
N. Benjamin Erichson, Lionel Mathelin, J. Nathan Kutz, Steven L. Brunton.
Randomized dynamic mode decomposition. SIAM Journal on Applied Dynamical
Systems, 18, 2019.
"""

import numpy as np

from .cdmd import CDMD
from .utils import compute_optimal_svd_rank
from .linalg import build_linalg_module


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
        m = self.snapshots.shape[-1]

        # Compute the target rank
        if self.snapshots.ndim < 3:
            optimal_svd_rank = compute_optimal_svd_rank(self.snapshots)
        else:
            optimal_svd_rank = min(self.snapshots.shape)

        # Generate random test matrix (with slight oversampling)
        linalg_module = build_linalg_module(self.snapshots)
        Omega = linalg_module.random((m, optimal_svd_rank + self._oversampling))

        # Compute sampling matrix
        Y = linalg_module.dot(self.snapshots, Omega)

        # Perform power iterations
        for _ in range(self._power_iters):
            Q = linalg_module.qr_reduced(Y)[0]
            snapQ = linalg_module.dot(self.snapshots.conj().swapaxes(-1, -2), Q)
            Z = linalg_module.qr_reduced(snapQ)[0]
            Y = linalg_module.dot(self.snapshots, Z)

        # Orthonormalize the sampling matrix
        Q = linalg_module.qr_reduced(Y)[0]

        # Project the snapshot matrix onto the smaller space
        B = linalg_module.dot(Q.conj().swapaxes(-1, -2), self.snapshots)

        # Save the compression matrix
        self._compression_matrix = Q.conj().swapaxes(-1, -2)

        return B
