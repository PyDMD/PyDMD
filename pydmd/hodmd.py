"""
Derived module from dmdbase.py for higher order dmd.

Reference:
- S. L Clainche, J. M. Vega, Higher Order Dynamic Mode Decomposition.
Journal on Applied Dynamical Systems, 16(2), 882-925, 2017.
"""
import numpy as np

from .dmdbase import DMDBase


class HODMD(DMDBase):
    """
    Higher Order Dynamic Mode Decomposition

    :param int svd_rank: rank truncation in SVD. If 0, the method computes the
        optimal rank and uses it for truncation; if positive number, the method
        uses the argument for the truncation; if -1, the method does not
        compute truncation.
    :param int tlsq_rank: rank truncation computing Total Least Square. Default
        is 0, that means TLSQ is not applied.
    :param bool exact: flag to compute either exact DMD or projected DMD.
        Default is False.
    :param bool opt: flag to compute optimal amplitudes. See :class:`DMDBase`.
        Default is False.
    :param rescale_mode: Scale Atilde as shown in
            10.1016/j.jneumeth.2015.10.010 (section 2.4) before computing its
            eigendecomposition. None means no rescaling, 'auto' means automatic
            rescaling using singular values, otherwise the scaling factors.
    :type rescale_mode: {'auto'} or None or numpy.ndarray
    :param int d: the new order for spatial dimension of the input snapshots.
        Default is 1.
    """

    def __init__(self, svd_rank=0, tlsq_rank=0, exact=False, opt=False, d=1,
        rescale_mode=None):
        super(HODMD, self).__init__(svd_rank, tlsq_rank, exact, opt,
            rescale_mode)
        self.d = d

    def fit(self, X):
        """
        Compute the Dynamic Modes Decomposition to the input data.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        """
        self._snapshots, self._snapshots_shape = self._col_major_2darray(X)
        snaps = np.concatenate(
            [
                self._snapshots[:, i:self._snapshots.shape[1] - self.d + i + 1]
                for i in range(self.d)
            ],
            axis=0)

        n_samples = self._snapshots.shape[1]
        X = snaps[:, :-1]
        Y = snaps[:, 1:]

        X, Y = self._compute_tlsq(X, Y, self.tlsq_rank)

        U, s, V = self._compute_svd(X, self.svd_rank)

        self._Atilde = self._build_lowrank_op(U, s, V, Y)

        self._eigs, self._modes = self._eig_from_lowrank_op(
            self._Atilde, Y, U, s, V, self.exact,
            rescale_mode=self.rescale_mode)
        self._modes = self._modes[:self._snapshots.shape[0], :]

        # Default timesteps
        self.original_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}
        self.dmd_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}

        self._b = self._compute_amplitudes(self._modes, self._snapshots,
                                           self._eigs, self.opt)


        return self
