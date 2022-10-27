"""
Derived module from cdmd.py for Randomized DMD

Reference:
N. Benjamin Erichson, Lionel Mathelin, J. Nathan Kutz, Steven L. Brunton.
Randomized dynamic mode decomposition. SIAM Journal on Applied Dynamical
Systems, 18, 2019.
"""

import numpy as np
from .cdmd import CDMD

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
    
