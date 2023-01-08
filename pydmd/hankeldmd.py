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
from .utils import prepare_snapshots, nan_average


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

    def _update_sub_dmd_time(self):
        """
        Update the time dictionaries (`dmd_time` and `original_time`) of
        the auxiliary DMD instance `HankelDMD._sub_dmd` after an update of the
        time dictionaries of the time dictionaries of this instance of the
        higher level instance of `HankelDMD`.
        """
        t0_hankel_first_occurrence = max(
            0,
            (self.dmd_time["t0"] - self.original_time["t0"])
            // self.dmd_time["dt"]
            - (self.original_time["t0"] + self.d - 1),
        )
        self._sub_dmd.dmd_time["t0"] = t0_hankel_first_occurrence

        tend_hankel_first_occurrence = max(
            0,
            (self.dmd_time["tend"] - self.original_time["t0"])
            // self.dmd_time["dt"]
            - (self.original_time["t0"] + self.d - 1),
        )
        self._sub_dmd.dmd_time["tend"] = tend_hankel_first_occurrence

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
        tensorized = rec.ndim == 3

        rec_snapshots_shape = (time_instants, self.d, space_dim)
        if tensorized:
            rec_snapshots_shape = (len(rec),) + rec_snapshots_shape

        linalg_module = build_linalg_module(rec)
        reconstructed_snapshots = linalg_module.full(
            rec_snapshots_shape, np.nan, dtype=rec.dtype
        )

        time_idxes = np.arange(self.d)[None].repeat(rec.shape[-1], axis=0)
        time_idxes += np.arange(rec.shape[-1])[:, None]
        d_idxes = np.arange(self.d)[None].repeat(rec.shape[-1], axis=0)

        splitted = cast_as_array(linalg_module.split(rec, self.d, axis=-2))
        if tensorized:
            # move batch axis to the first place
            splitted = splitted.swapaxes(0, 1)
            splitted = splitted.swapaxes(-1, -2).swapaxes(-2, -3)
        else:
            splitted = linalg_module.moveaxis(splitted, -1, -3)

        reconstructed_snapshots[..., time_idxes, d_idxes, :] = splitted
        return (
            reconstructed_snapshots[timeindex]
            if timeindex
            else reconstructed_snapshots
        )

    def _first_reconstructions(self, reconstructions):
        """Return The first snapshot that occurs in `reconstructions` for each
        snapshot available.
        """
        n_time = reconstructions.shape[-3]
        time_idxes = np.arange(n_time)
        d_idxes = np.arange(n_time)
        d_idxes[self.d - 1 :] = self.d - 1
        return reconstructions[..., time_idxes, d_idxes, :].swapaxes(-1, -2)

    @property
    def reconstructed_data(self):
        self._update_sub_dmd_time()

        rec = self.reconstructions_of_timeindex()
        linalg_module = build_linalg_module(rec)

        if self._reconstruction_method == "first":
            result = self._first_reconstructions(rec)
        elif self._reconstruction_method == "mean":
            result = linalg_module.nanmean(rec, axis=-2).T
        elif isinstance(self._reconstruction_method, (np.ndarray, list)):
            weights = linalg_module.new_array(self._reconstruction_method)
            result = nan_average(rec, weights).T
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
        return result[..., time_index : time_index + len(self.dmd_timesteps)]

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

        linalg_module = build_linalg_module(X)

        snp = prepare_snapshots(X)
        n_samples = snp.shape[-1]
        if n_samples < self._d:
            raise ValueError(
                f"The number of snapshots provided is not enough for d={self._d}."
            )

        self._snapshots = linalg_module.pseudo_hankel_matrix(snp, self._d)
        self._sub_dmd.fit(self._snapshots)

        # Default timesteps
        self._set_initial_time_dictionary(
            {"t0": 0, "tend": n_samples - 1, "dt": 1}
        )

        return self
