"""
Derived module from dmdbase.py for forward/backward dmd.
"""
import numpy as np

from scipy.linalg import sqrtm
from .dmd import DMD


class FbDMD(DMD):
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
        is 0, that means no truncation.
    :param bool exact: flag to compute either exact DMD or projected DMD.
        Default is False.
    :param bool opt: flag to compute optimized DMD. Default is False.
    :param rescale_mode: Scale Atilde as shown in
            10.1016/j.jneumeth.2015.10.010 (section 2.4) before computing its
            eigendecomposition. None means no rescaling, 'auto' means automatic
            rescaling using singular values, otherwise the scaling factors.
    :type rescale_mode: {'auto'} or None or numpy.ndarray
    :param int amplitudes_snapshot_index: The (temporal) index of the snapshot
        used to compute DMD modes amplitudes. The reconstruction will generally
        be better in time instants near the chosen snapshot; however increasing
        this value may lead to wrong results when the system presents small
        eigenvalues. For this reason a manual selection of the number of
        eigenvalues in the system may be needed (check svd_rank). Also setting
        svd_rank to a value between 0 and 1 can lead to better results. Default
        value is 0.

    Reference: Dawson et al. https://arxiv.org/abs/1507.02264
    """

    def __init__(self, svd_rank=0, tlsq_rank=0, exact=False, opt=False,
        rescale_mode=None, amplitudes_snapshot_index=0):
        super().__init__(svd_rank=svd_rank, tlsq_rank=tlsq_rank, exact=exact,
            opt=opt, rescale_mode=rescale_mode, forward_backward=True,
            amplitudes_snapshot_index=amplitudes_snapshot_index)
