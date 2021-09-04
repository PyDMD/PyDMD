"""
Derived module from hankeldmd.py for higher order dmd.

Reference:
- S. L Clainche, J. M. Vega, Higher Order Dynamic Mode Decomposition.
Journal on Applied Dynamical Systems, 16(2), 882-925, 2017.
"""
import numpy as np

import warnings

from .hankeldmd import HankelDMD
from .utils import compute_tlsq, compute_svd


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
    :param opt: argument to control the computation of DMD modes amplitudes. See
        :class:`DMDBase`. Default is False.
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
    :param reconstruction_method: Due to how HODMD is defined, we have several
        versions of the same snapshot. The parameter `reconstruction_method`
        allows changing how these versions are combined in
        `reconstructed_data`.  If `'first'`, only the first version is selected
        (default behavior); if `'mean'` we take the mean of all the versions;
        if the parameter is an array of floats of size `d`, the return value is
        the weighted average of the versions.
    :type reconstruction_method: {'first', 'mean'} or array-like
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

        self.svd_rank_extra = svd_rank_extra  # TODO improve names
        self.U_extra = None

    def reconstructions_of_timeindex(self, timeindex=None):
        """
        Build a collection of all the available versions of the given
        `timeindex`. The indexing of time instants is the same used for
        :func:`reconstructed_data`. For each time instant there are at least
        one and at most `d` versions.  If `timeindex` is `None` the function
        returns the whole collection, for all the time instants.

        :param int timeindex: The index of the time snapshot.
        :return: A collection of all the available versions for the requested
            time instants, represented by a matrix (or tensor).
            Axes:
            0. Number of time instants;
            1. Copies of the snapshot;
            2. Space dimension of the snapshot.
            The first axis is omitted if only one single time instant is
            selected, in this case the output becomes a 2D matrix.
        :rtype: numpy.ndarray
        """
        snapshots = super().reconstructions_of_timeindex(timeindex)
        if snapshots.ndim == 2:  # single time instant
            snapshots = self.U_extra.dot(snapshots.T).T
        elif snapshots.ndim == 3:  # all time instants
            snapshots = np.array(
                [self.U_extra.dot(snapshot.T).T for snapshot in snapshots]
            )
        else:
            raise RuntimeError

        return snapshots

    def fit(self, X):
        """
        Compute the Dynamic Modes Decomposition to the input data.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable

        """
        org_snp, snapshots_shape = self._col_major_2darray(X)

        if org_snp.shape[0] == 1:
            self.U_extra, _, _ = compute_svd(org_snp, -1)
            warnings.warn(
                "The parameter 'svd_rank_extra={}' has been ignored because the given system is a scalar function".format(
                    self.svd_rank_extra
                )
            )
        else:
            self.U_extra, _, _ = compute_svd(org_snp, self.svd_rank_extra)

        snp = self.U_extra.T.dot(org_snp)

        super().fit(snp)
        self._snapshots_shape = snapshots_shape
        self._snapshots = org_snp

        return self
