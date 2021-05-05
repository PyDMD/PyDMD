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
        allows changing how these versions are combined in `reconstructed_data`.
        If `'first'`, only the first version is selected (default behavior);
        if `'mean'` we take the mean of all the versions; if the parameter is an
        array of floats of size `d`, the return value is the weighted average
        of the versions.
    :type reconstruction_method: {'first', 'mean'} or array-like
    """

    def __init__(self, svd_rank=0, tlsq_rank=0, exact=False, opt=False,
        rescale_mode=None, forward_backward=False, d=1, sorted_eigs=False,
        reconstruction_method='first'):
        super(HODMD, self).__init__(svd_rank=svd_rank, tlsq_rank=tlsq_rank,
            exact=exact, opt=opt, rescale_mode=rescale_mode,
            sorted_eigs=sorted_eigs)
        self._d = d

        if isinstance(reconstruction_method, list):
            if len(reconstruction_method) != d:
                raise ValueError('The length of the array of weights must be equal to d')
        elif isinstance(reconstruction_method, np.ndarray):
            if reconstruction_method.ndim > 1 or reconstruction_method.shape[0] != d:
                raise ValueError('The length of the array of weights must be equal to d')
        self._reconstruction_method = reconstruction_method

    @property
    def d(self):
        return self._d

    def reconstructions_of_timeindex(self, timeindex=None):
        """
        Build a collection of all the available versions of the given
        `timeindex`. The indexing of time instants is the same used for
        :func:`reconstructed_data`. For each time instant there are at least one
        and at most `d` versions. If `timeindex` is `None` the function returns
        the whole collection, for all the time instants.

        :param int timeindex: The index of the time snapshot.
        :return: a collection of all the available versions for the given
            time snapshot, or for all the time snapshots if `timeindex` is
            `None` (in the second case, time varies along the first dimension
            of the array returned).
        :rtype: numpy.ndarray or list
        """
        rec = super(HODMD, self).reconstructed_data
        space_dim = rec.shape[0] // self.d
        time_instants = rec.shape[1] + self.d - 1

        # for each time instance, we take the mean of all its appearences.
        # each snapshot appears at most d times (for instance, the first and the
        # last appear only once).
        reconstructed_snapshots = np.full((time_instants, self.d, space_dim), np.nan, dtype=np.complex128)

        first_empty = np.zeros((time_instants,), dtype=np.int8)

        for time_slice_idx in range(rec.shape[1]):
            time_slice = rec[:, time_slice_idx]

            for i in range(self.d):
                time_idx = time_slice_idx + i
                mx = time_slice[space_dim * i : space_dim * (i + 1)]
                reconstructed_snapshots[time_idx, first_empty[time_idx]] = mx
                first_empty[time_idx] += 1

        if timeindex is None:
            return reconstructed_snapshots
        else:
            return reconstructed_snapshots[timeindex][:first_empty[timeindex]]

    @property
    def reconstructed_data(self):
        rec = self.reconstructions_of_timeindex()
        rec = np.ma.array(rec, mask=np.isnan(rec))

        if self._reconstruction_method == 'first':
            result = rec[:,0].T
        elif self._reconstruction_method == 'mean':
            result = np.mean(rec, axis=1).T
        elif (isinstance(self._reconstruction_method, list) or
            isinstance(self._reconstruction_method, np.ndarray)):
            result = np.average(rec, axis=1, weights=self._reconstruction_method).T
        else:
            raise ValueError("The reconstruction method wasn't recognized: {}"
                .format(self._reconstruction_method))

        return result.filled(fill_value=0)

    def fit(self, X):
        """
        Compute the Dynamic Modes Decomposition to the input data.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        """
        snp, self._snapshots_shape = self._col_major_2darray(X)
        self._snapshots = np.concatenate(
            [
                snp[:, i:snp.shape[1] - self.d + i + 1]
                for i in range(self.d)
            ],
            axis=0)

        n_samples = self._snapshots.shape[1]
        X = self._snapshots[:, :-1]
        Y = self._snapshots[:, 1:]

        X, Y = compute_tlsq(X, Y, self.tlsq_rank)
        U, s, V = self.operator.compute_operator(X,Y)

        # Default timesteps
        self.original_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}
        self.dmd_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}

        self._b = self._compute_amplitudes()

        return self
