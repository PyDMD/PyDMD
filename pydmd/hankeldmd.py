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

from .dmd import DMD
from .dmdbase import DMDBase
from .preprocessing.hankel import hankel_preprocessing
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

        sub_dmd = DMD(
            svd_rank=svd_rank,
            tlsq_rank=tlsq_rank,
            exact=exact,
            opt=opt,
            rescale_mode=rescale_mode,
            forward_backward=forward_backward,
            sorted_eigs=sorted_eigs,
            tikhonov_regularization=tikhonov_regularization,
        )
        self._sub_dmd = hankel_preprocessing(
            sub_dmd, d=d, reconstruction_method=reconstruction_method
        )

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
            - (self.original_time["t0"] + self._d - 1),
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

    @property
    def reconstructed_data(self):
        self._update_sub_dmd_time()

        # we want to return only the requested timesteps
        time_index = min(
            self._d - 1,
            int(
                (self.dmd_time["t0"] - self.original_time["t0"])
                // self.dmd_time["dt"]
            ),
        )
        return self._sub_dmd.reconstructed_data[
            :, time_index : time_index + len(self.dmd_timesteps)
        ]

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
        return self._sub_dmd.snapshots

    @property
    def modes_activation_bitmask(self):
        return self._sub_dmd.modes_activation_bitmask

    @modes_activation_bitmask.setter
    def modes_activation_bitmask(self, value):
        self._sub_dmd.modes_activation_bitmask = value

    def __getitem__(self, key):
        # The implementation was asking for problems...
        raise ValueError("This operation is not allowed for HankelDMD")

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

        self._sub_dmd.fit(X)

        # Default timesteps
        self._set_initial_time_dictionary(
            {"t0": 0, "tend": n_samples - 1, "dt": 1}
        )

        return self
