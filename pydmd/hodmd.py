"""
Derived module from hankeldmd.py for higher order dmd.

Reference:
- S. L Clainche, J. M. Vega, Higher Order Dynamic Mode Decomposition.
Journal on Applied Dynamical Systems, 16(2), 882-925, 2017.
"""
import warnings

import numpy as np

from .hankeldmd import HankelDMD
from .preprocessing.svd_projection import svd_projection_preprocessing
from .snapshots import Snapshots


class HODMD(HankelDMD):
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
    :param opt: argument to control the computation of DMD modes amplitudes.
        See :class:`DMDBase`. Default is False.
    :type opt: bool or int
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
    :param sorted_eigs: Sort eigenvalues (and modes/dynamics accordingly) by
        magnitude if `sorted_eigs='abs'`, by real part (and then by imaginary
        part to break ties) if `sorted_eigs='real'`. Default: False.
    :type sorted_eigs: {'real', 'abs'} or False
    :param reconstruction_method: Method used to reconstruct the snapshots of
        the dynamical system from the multiple versions available due to how
        HODMD is conceived. If `'first'` (default) the first version
        available is selected (i.e. the nearest to the 0-th row in the
        augmented matrix). If `'mean'` we compute the element-wise mean. If
        `reconstruction_method` is an array of float values we compute the
        weighted average (for each snapshots) using the given values as weights
        (the number of weights must be equal to `d`).
    :param svd_rank_extra: the rank for the initial reduction of the input
        data, performed before the rearrangement of the input data to the
        (pseudo) Hankel matrix format; If 0, the method computes the optimal
        rank and uses it for truncation; if positive interger, the method uses
        the argument for the truncation; if float between 0 and 1, the rank is
        the number of the biggest singular values that are needed to reach the
        'energy' specified by `svd_rank`; if -1, the method does not compute
        truncation.
    :type svd_rank: int or float
    """

    def __init__(
        self,
        svd_rank=0,
        tlsq_rank=0,
        exact=False,
        opt=False,
        rescale_mode=None,
        forward_backward=False,
        d=1,
        sorted_eigs=False,
        reconstruction_method="first",
        svd_rank_extra=0,
    ):
        super().__init__(
            svd_rank=svd_rank,
            tlsq_rank=tlsq_rank,
            exact=exact,
            opt=opt,
            rescale_mode=rescale_mode,
            forward_backward=forward_backward,
            d=d,
            sorted_eigs=sorted_eigs,
            reconstruction_method=reconstruction_method,
        )
        self._sub_dmd = svd_projection_preprocessing(
            self._sub_dmd, svd_rank_extra
        )

    def fit(self, X):
        """
        Compute the Dynamic Modes Decomposition to the input data.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        """
        snapshots_holder = Snapshots(X)

        super().fit(snapshots_holder.snapshots)
        self._snapshots_holder = snapshots_holder

        return self
