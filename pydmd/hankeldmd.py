"""
Derived module from dmdbase.py for hankel dmd.

Reference:
- H. Arbabi, I. Mezic, Ergodic theory, dynamic mode decomposition, and
computation of spectral properties of the Koopman operator. SIAM Journal on
Applied Dynamical Systems, 2017, 16.4: 2096-2126.
"""
from copy import copy

import numpy as np

from .dmd import DMD
from .dmdbase import DMDBase
from .linalg import build_linalg_module, cast_as_array
from .utils import prepare_snapshots


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
        space_dim = rec.shape[-2] // self.d
        time_instants = rec.shape[-1] + self.d - 1

        linalg_module = build_linalg_module(rec)
        # for each time instance, we collect all its appearences.  each
        # snapshot appears at most d times (for instance, the first appears
        # only once).
        reconstructed_snapshots = linalg_module.full(
            (time_instants, self.d, space_dim), np.nan, dtype=rec.dtype
        )

        c_idxes = (
            np.arange(self.d)[:, None]
            .repeat(2, axis=1)[None]
            .repeat(rec.shape[-1], axis=0)
        )
        c_idxes[..., 0] += np.arange(rec.shape[-1])[:, None]

        splitted = cast_as_array(
            linalg_module.split(rec.swapaxes(-1, -2), self.d, axis=-1)
        )
        if self._snapshots.ndim == 2:
            reconstructed_snapshots[
                c_idxes[..., 0], c_idxes[..., 1]
            ] = splitted.swapaxes(0, 1)
        else:
            reconstructed_snapshots[
                :, c_idxes[..., 0], c_idxes[..., 1]
            ] = splitted.swapaxes(-2, -3)

        return (
            reconstructed_snapshots[timeindex]
            if timeindex
            else reconstructed_snapshots
        )

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
        first_nonmasked_idx_1 = np.arange(reconstructions.shape[-2])
        first_nonmasked_idx_2 = first_nonmasked_idx_1.copy()
        first_nonmasked_idx_2[self.d - 1 :] = self.d - 1
        if self._snapshots.ndim == 2:
            return reconstructions[
                first_nonmasked_idx_1, first_nonmasked_idx_2
            ].T
        else:
            return reconstructions[
                :, first_nonmasked_idx_1, first_nonmasked_idx_2
            ].swapaxes(-1, -2)

    @staticmethod
    def _masked_weighted_average(arr, weights):
        if weights.ndim != 1:
            raise ValueError("Expected 1D weights")

        linalg_module = build_linalg_module(arr)

        if arr.ndim == 4:
            arr0, arr1, _, arr3 = arr.shape
        else:
            arr0 = 0
            arr1, _, arr3 = arr.shape
        repeated_weights = linalg_module.repeat(weights[None], arr1, 0)
        repeated_weights = linalg_module.repeat(
            repeated_weights[..., None], arr3, 2
        )
        if arr.ndim == 4:
            repeated_weights = linalg_module.repeat(
                repeated_weights[None], arr0, 0
            )

        non_normalized_mean = linalg_module.nansum(arr * weights, axis=-2)

        weights_sum = linalg_module.nansum(repeated_weights, axis=-2)
        # avoid divide by zero
        weights_sum[weights_sum == 0.0] = 1
        return non_normalized_mean / weights_sum

    @property
    def reconstructed_data(self):
        self._update_sub_dmd_time()

        rec = self.reconstructions_of_timeindex()
        linalg_module = build_linalg_module(rec)

        if self._reconstruction_method == "first":
            result = self._first_reconstructions(rec)
        elif self._reconstruction_method == "mean":
            result = linalg_module.nanmean(rec, axis=1).T
        elif isinstance(self._reconstruction_method, (np.ndarray, list)):
            weights = linalg_module.new_array(self._reconstruction_method)
            result = HankelDMD._masked_weighted_average(rec, weights).T
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
        indexed_result = result[
            ..., time_index : time_index + len(self.dmd_timesteps)
        ]
        indexed_result = linalg_module.new_array(indexed_result)
        # in-place!
        indexed_result[linalg_module.isnan(indexed_result)] = 0
        return indexed_result

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
        linalg_module = build_linalg_module(X)

        n_ho_snapshots = X.shape[-1] - self._d + 1
        idxes = linalg_module.repeat(
            linalg_module.arange(self._d)[None], repeats=n_ho_snapshots, axis=0
        )
        idxes += linalg_module.arange(n_ho_snapshots)[:, None]
        if X.ndim == 2:
            return X.T[idxes].reshape((-1, X.shape[0] * self._d)).T
        else:
            return (
                X.swapaxes(-1, -2)[..., idxes]
                .reshape((-1, -1, X.shape[-2] * self._d))
                .swapaxes(-1, -2)
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
        sub_dmd_copy.allocate_modes_bitmask_proxy()

        shallow_copy = copy(self)
        shallow_copy._sub_dmd = sub_dmd_copy
        return DMDBase.__getitem__(shallow_copy, key)

    def fit(self, X):
        """
        Compute the Dynamic Modes Decomposition to the input data.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        """
        self.reset()

        snp = prepare_snapshots(X)
        n_samples = snp.shape[-1]
        if n_samples < self._d:
            raise ValueError(
                f"The number of snapshots provided is not enough for d={self._d}."
            )

        self._snapshots = self._pseudo_hankel_matrix(snp)
        self._sub_dmd.fit(self._snapshots)

        # Default timesteps
        self._set_initial_time_dictionary(
            {"t0": 0, "tend": n_samples - 1, "dt": 1}
        )

        return self
