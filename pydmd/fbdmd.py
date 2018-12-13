"""
Derived module from dmdbase.py for forward/backward dmd.
"""
import numpy as np

from scipy.linalg import sqrtm
from .dmdbase import DMDBase


class FbDMD(DMDBase):
    """
    Forward/backward DMD class.

    :param svd_rank: the rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive interger, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation.
    :type svd_rank: int or float
    :param int tlsq_rank: rank truncation computing Total Least Square. Default
        is 0, that means TLSQ is not applied.
    :param bool exact: flag to compute either exact DMD or projected DMD.
        Default is False.
    :param bool opt: flag to compute optimal amplitudes. See :class:`DMDBase`.
        Default is False.

    Reference: Dawson et al. https://arxiv.org/abs/1507.02264
    """

    def fit(self, X):
        """
        Compute the Dynamics Modes Decomposition to the input data.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        """
        self._snapshots, self._snapshots_shape = self._col_major_2darray(X)

        n_samples = self._snapshots.shape[1]
        X = self._snapshots[:, :-1]
        Y = self._snapshots[:, 1:]

        X, Y = self._compute_tlsq(X, Y, self.tlsq_rank)

        Uy, sy, Vy = self._compute_svd(Y, self.svd_rank)
        Ux, sx, Vx = self._compute_svd(X, self.svd_rank)

        if len(sy) != len(sx):
            raise ValueError(
                'The optimal truncation produced different number of singular'
                'values for the X and Y matrix, please specify different'
                'svd_rank')

        # Backward operator
        bAtilde = self._build_lowrank_op(Uy, sy, Vy, X)
        # Forward operator
        fAtilde = self._build_lowrank_op(Ux, sx, Vx, Y)

        self._Atilde = sqrtm(fAtilde.dot(np.linalg.inv(bAtilde)))

        self._eigs, self._modes = self._eig_from_lowrank_op(
            self._Atilde, Y, Ux, sx, Vx, self.exact)

        self.original_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}
        self.dmd_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}

        self._b = self._compute_amplitudes(self._modes, self._snapshots,
                                           self._eigs, self.opt)

        return self
