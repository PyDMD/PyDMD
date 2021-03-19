"""
Derived module from dmdbase.py for higher order dmd.

Reference:
- S. L Clainche, J. M. Vega, Higher Order Dynamic Mode Decomposition.
Journal on Applied Dynamical Systems, 16(2), 882-925, 2017.
"""
import numpy as np

from .dmdbase import DMDBase
from .utils import compute_tlsq


class HODMD(DMDBase):
    """
    Higher Order Dynamic Mode Decomposition

    :param svd_rank: the rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive interger, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation.
    :type svd_rank: int or float
    :param int tlsq_rank: rank truncation computing Total Least Square. Default
        is 0, that means no truncation.
    :param bool exact: flag to compute either exact DMD or projected DMD.
        Default is False.
    :param bool opt: flag to compute optimized DMD. Default is False.
    :param rescale_mode: Scale Atilde as shown in
            10.1016/j.jneumeth.2015.10.010 (section 2.4) before computing its
            eigendecomposition. None means no rescaling, 'auto' means automatic
            rescaling using singular values, otherwise the scaling factors.
    :type rescale_mode: {'auto'} or None or numpy.ndarray
    :param bool forward_backward: If True, the low-rank operator is computed
        like in fbDMD (reference: https://arxiv.org/abs/1507.02264). Default is
        False.
    :param int d: the new order for spatial dimension of the input snapshots.
        Default is 1.
    """

    def __init__(self, svd_rank=0, tlsq_rank=0, exact=False, opt=False,
        rescale_mode=None, forward_backward=False, d=1):
        super(HODMD, self).__init__(svd_rank=svd_rank, tlsq_rank=tlsq_rank,
            exact=exact, opt=opt, rescale_mode=rescale_mode)
        self._d = d

    @property
    def d(self):
        return self._d

    @DMDBase.modes.getter
    def modes(self):
        return super(HODMD, self).modes[:self._snapshots.shape[0], :]

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

        X, Y = compute_tlsq(X, Y, self.tlsq_rank)
        U, s, V = self._Atilde.compute_operator(X,Y)

        # Default timesteps
        self.original_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}
        self.dmd_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}

        self._b = self._compute_amplitudes()

        return self
