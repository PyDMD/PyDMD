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

    Reference: Dawson et al. https://arxiv.org/abs/1507.02264
    """

    def __init__(self, svd_rank=0, tlsq_rank=0, exact=False, opt=False,
        rescale_mode=None):
        super().__init__(svd_rank=svd_rank, tlsq_rank=tlsq_rank, exact=exact,
            opt=opt, rescale_mode=rescale_mode, forward_backward=True)
