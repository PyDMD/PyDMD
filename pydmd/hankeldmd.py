"""
Derived module from dmdbase.py for hankel dmd.

Reference:
- H. Arbabi, I. Mezic, Ergodic theory, dynamic mode decomposition, and
computation of spectral properties of the Koopman operator. SIAM Journal on
Applied Dynamical Systems, 2017, 16.4: 2096-2126.
"""
from copy import copy
from numbers import Number

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

from .dmd import DMD
from .dmdbase import DMDBase
from .snapshots import Snapshots


class HankelDMD(DMDBase):
    """
    Hankel Dynamic Mode Decomposition

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
        HankelDMD is conceived. If `'first'` (default) the first version
        available is selected (i.e. the nearest to the 0-th row in the
        augmented matrix). If `'mean'` we compute the element-wise mean. If
        `reconstruction_method` is an array of float values we compute the
        weighted average (for each snapshots) using the given values as weights
        (the number of weights must be equal to `d`).
    :type reconstruction_method: {'first', 'mean'} or array-like
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
        tikhonov_regularization=None,
    ):
        super().__init__(
            svd_rank=svd_rank,
            tlsq_rank=tlsq_rank,
            exact=exact,
            opt=opt,
            rescale_mode=rescale_mode,
            sorted_eigs=sorted_eigs,
        )
        self._d = d
        self._ho_snapshots = None

        if isinstance(reconstruction_method, list):
            if len(reconstruction_method) != d:
                raise ValueError(
                    "The length of the array of weights must be equal to d"
                )
        elif isinstance(reconstruction_method, np.ndarray):
            if (
                reconstruction_method.ndim > 1
                or reconstruction_method.shape[0] != d
            ):
                raise ValueError(
                    "The length of the array of weights must be equal to d"
                )
        self._reconstruction_method = reconstruction_method

        self._sub_dmd = DMD(
            svd_rank=svd_rank,
            tlsq_rank=tlsq_rank,
            exact=exact,
            opt=opt,
            rescale_mode=rescale_mode,
            forward_backward=forward_backward,
            sorted_eigs=sorted_eigs,
            tikhonov_regularization=tikhonov_regularization,
        )

    @property
    def d(self):
        """The new order for spatial dimension of the input snapshots."""
        return self._d

    def _hankel_first_occurrence(self, time):
        r"""
        For a given `t` such that there is :math:`k \in \mathbb{N}` such that
        :math:`t = t_0 + k dt`, return the index of the first column in Hankel
        pseudo matrix (see also :func:`_pseudo_hankel_matrix`) which contains
        the snapshot corresponding to `t`.

        :param time: The time corresponding to the requested snapshot.
        :return: The index of the first appeareance of `time` in the columns of
            Hankel pseudo matrix.
        :rtype: int
        """
        assert isinstance(time, Number) or np.asarray(time).ndim == 1
        return max(
            0,
            (time - self.original_time["t0"]) // self.dmd_time["dt"]
            - (self.original_time["t0"] + self.d - 1),
        )

    def _update_sub_dmd_time(self):
        """
        Update the time dictionaries (`dmd_time` and `original_time`) of
        the auxiliary DMD instance `HankelDMD._sub_dmd` after an update of the
        time dictionaries of the time dictionaries of this instance of the
        higher level instance of `HankelDMD`.
        """
        self._sub_dmd.dmd_time["t0"] = self._hankel_first_occurrence(
            self.dmd_time["t0"]
        )
        self._sub_dmd.dmd_time["tend"] = self._hankel_first_occurrence(
            self.dmd_time["tend"]
        )

    def reconstructions_of_timeindex(self, timeindex=None):
        """
        Build a collection of all the available versions of the given
        `timeindex`. The indexing of time instants is the same used for
        :func:`reconstructed_data`. For each time instant there are at least
        one and at most `d` versions. If `timeindex` is `None` the function
        returns the whole collection, for all the time instants.

        :param int timeindex: The index of the time snapshot.
        :return: a collection of all the available versions for the given
            time snapshot, or for all the time snapshots if `timeindex` is
            `None` (in the second case, time varies along the first dimension
            of the array returned).
        :rtype: numpy.ndarray or list
        """
        self._update_sub_dmd_time()

        rec = self._sub_dmd.reconstructed_data
        space_dim = rec.shape[0] // self.d
        time_instants = rec.shape[1] + self.d - 1

        # for each time instance, we collect all its appearences.  each
        # snapshot appears at most d times (for instance, the first appears
        # only once).
        reconstructed_snapshots = np.full(
            (time_instants, self.d, space_dim), np.nan, dtype=rec.dtype
        )

        c_idxes = (
            np.array(range(self.d))[:, None]
            .repeat(2, axis=1)[None, :]
            .repeat(rec.shape[1], axis=0)
        )
        c_idxes[:, :, 0] += np.array(range(rec.shape[1]))[:, None]

        reconstructed_snapshots[c_idxes[:, :, 0], c_idxes[:, :, 1]] = np.array(
            np.swapaxes(np.split(rec.T, self.d, axis=1), 0, 1)
        )

        if timeindex is None:
            return reconstructed_snapshots

        return reconstructed_snapshots[timeindex]

    def _first_reconstructions(self, reconstructions):
        """Return the first occurrence of each snapshot available in the given
        matrix (which must be the result of `self._sub_dmd.reconstructed_data`,
        or have the same shape).

        :param reconstructions: A matrix of (higher-order) snapshots having
            shape `(space*self.d, time_instants)`
        :type reconstructions: np.ndarray
        :return: The first snapshot that occurs in `reconstructions` for each
            available time instant.
        :rtype: np.ndarray
        """
        first_nonmasked_idx = np.repeat(
            np.array(range(reconstructions.shape[0]))[:, None], 2, axis=1
        )
        first_nonmasked_idx[self.d - 1 :, 1] = self.d - 1

        return reconstructions[
            first_nonmasked_idx[:, 0], first_nonmasked_idx[:, 1]
        ].T

    @property
    def reconstructed_data(self):
        self._update_sub_dmd_time()

        rec = self.reconstructions_of_timeindex()
        rec = np.ma.array(rec, mask=np.isnan(rec))

        if self._reconstruction_method == "first":
            result = self._first_reconstructions(rec)
        elif self._reconstruction_method == "mean":
            result = np.mean(rec, axis=1).T
        elif isinstance(self._reconstruction_method, (np.ndarray, list)):
            result = np.average(
                rec, axis=1, weights=self._reconstruction_method
            ).T
        else:
            raise ValueError(
                "The reconstruction method wasn't recognized: {}".format(
                    self._reconstruction_method
                )
            )

        # we want to return only the requested timesteps
        time_index = min(
            self.d - 1,
            int(
                (self.dmd_time["t0"] - self.original_time["t0"])
                // self.dmd_time["dt"]
            ),
        )
        result = result[:, time_index : time_index + len(self.dmd_timesteps)]

        return result.filled(fill_value=0)

    def _pseudo_hankel_matrix(self, X):
        """
        Arrange the snapshot in the matrix `X` into the (pseudo) Hankel
        matrix. The attribute `d` controls the number of snapshot from `X` in
        each snapshot of the Hankel matrix.

        :Example:

            >>> from pydmd import HankelDMD
            >>> import numpy as np

            >>> dmd = HankelDMD(d=2)
            >>> a = np.array([[1, 2, 3, 4, 5]])
            >>> dmd._pseudo_hankel_matrix(a)
            array([[1, 2, 3, 4],
                   [2, 3, 4, 5]])
            >>> dmd = HankelDMD(d=4)
            >>> dmd._pseudo_hankel_matrix(a)
            array([[1, 2],
                   [2, 3],
                   [3, 4],
                   [4, 5]])

            >>> dmd = HankelDMD(d=2)
            >>> a = np.array([1,2,3,4,5,6]).reshape(2,3)
            >>> a
            array([[1, 2, 3],
                   [4, 5, 6]])
            >>> dmd._pseudo_hankel_matrix(a)
            array([[1, 2],
                   [4, 5],
                   [2, 3],
                   [5, 6]])
        """
        return (
            swv(X.T, (self.d, X.shape[0]))[:, 0]
            .reshape(X.shape[1] - self.d + 1, -1)
            .T
        )

    @property
    def modes(self):
        return self._sub_dmd.modes

    @property
    def eigs(self):
        return self._sub_dmd.eigs

    @property
    def amplitudes(self):
        return self._sub_dmd.amplitudes

    @property
    def operator(self):
        return self._sub_dmd.operator

    @property
    def svd_rank(self):
        return self._sub_dmd.svd_rank

    @property
    def ho_snapshots(self):
        """
        Get the time-delay data matrix.

        :return: the matrix that contains the time-delayed data.
        :rtype: numpy.ndarray
        """
        return self._ho_snapshots

    @property
    def modes_activation_bitmask(self):
        return self._sub_dmd.modes_activation_bitmask

    @modes_activation_bitmask.setter
    def modes_activation_bitmask(self, value):
        self._sub_dmd.modes_activation_bitmask = value

    # due to how we implemented HankelDMD we need an alternative implementation
    # of __getitem__
    def __getitem__(self, key):
        """
        Restrict the DMD modes used by this instance to a subset of indexes
        specified by keys. The value returned is a shallow copy of this DMD
        instance, with a different value in :func:`modes_activation_bitmask`.
        Therefore assignments to attributes are not reflected into the original
        instance.

        However the DMD instance returned should not be used for low-level
        manipulations on DMD modes, since the underlying DMD operator is shared
        with the original instance. For this reasons modifications to NumPy
        arrays may result in unwanted and unspecified situations which should
        be avoided in principle.

        :param key: An index (integer), slice or list of indexes.
        :type key: int or slice or list or np.ndarray
        :return: A shallow copy of this DMD instance having only a subset of
            DMD modes which are those indexed by `key`.
        :rtype: HankelDMD
        """

        sub_dmd_copy = copy(self._sub_dmd)
        sub_dmd_copy._allocate_modes_bitmask_proxy()

        shallow_copy = copy(self)
        shallow_copy._sub_dmd = sub_dmd_copy
        return DMDBase.__getitem__(shallow_copy, key)

    def fit(self, X):
        """
        Compute the Dynamic Modes Decomposition to the input data.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        """
        self._reset()

        self._snapshots_holder = Snapshots(X)

        n_samples = self.snapshots.shape[-1]
        if n_samples < self._d:
            msg = """The number of snapshots provided is not enough for d={}.
Expected at least d."""
            raise ValueError(msg.format(self._d))

        self._ho_snapshots = Snapshots(
            self._pseudo_hankel_matrix(self.snapshots)
        ).snapshots
        self._sub_dmd.fit(self._ho_snapshots)

        # Default timesteps
        self._set_initial_time_dictionary(
            {"t0": 0, "tend": n_samples - 1, "dt": 1}
        )

        return self
