"""
Base module for the DMD: `fit` method must be implemented in inherited classes
"""
from __future__ import division

from builtins import object
from builtins import range
from os.path import splitext
import warnings
import pickle
from copy import copy, deepcopy

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from past.utils import old_div

from .dmdoperator import DMDOperator
from .utils import compute_svd
from .linalg import build_linalg_module, same_linalg_type, cast_as_array, is_array

mpl.rcParams["figure.max_open_warning"] = 0


class ActivationBitmaskProxy:
    """
    A proxy which stands in the middle between a bitmask and an instance of
    :class:`DMDBase`. The proxy holds the original values of modes,
    eigenvalues and amplitudes, and exposes (via
    :func:`ActivationBitmaskProxy.modes`, :func:`ActivationBitmaskProxy.eigs`
    and :func:`ActivationBitmaskProxy.amplitudes`) the proxied (i.e. filtered)
    those quantities, depending on the current value of the
    bitmask (see also :func:`ActivationBitmaskProxy.change_bitmask`).

    This machinery is needed in order to allow for the modification of the
    matrices containing modes, amplitudes and eigenvalues after the indexing
    provided by the bitmask. Since double indexing in NumPy does not deliver a
    modifiable view of the original array, we need to propagate any change
    on the selection to the original matrices at some point: we decided to
    propagate the changes just before a change in the bitmask, namely in the
    last available moment before losing the information provided by the ``old''
    bitmask.

    In order to make sure the proxy is invalidated when :func:`DMDBase.fit` is
    called multiple times we cache the Python object ID of the array representing
    the lowrank DMD operator. When the ID changes, the bitmask is invalidated (see
    :func:`DMDBase._try_invalidate_bitmask_proxy`).

    :param dmd_operator: DMD operator to be proxied.
    :type dmd_operator: DMDOperator
    :param amplitudes: DMD amplitudes.
    :type amplitudes: np.ndarray
    """

    def __init__(self, dmd_operator, amplitudes):
        linalg_module = build_linalg_module(dmd_operator.modes)
        self._original_modes = dmd_operator.modes
        self._original_eigs = linalg_module.atleast_1d(dmd_operator.eigenvalues)
        self._original_amplitudes = linalg_module.atleast_1d(amplitudes)

        self._operator_id = id(dmd_operator.as_numpy_array)

        self.old_bitmask = None

        self.change_bitmask(linalg_module.full(len(dmd_operator.eigenvalues), True))

    def change_bitmask(self, value):
        """
        Change the bitmask which regulates this proxy.

        Before changing the bitmask this method reflects any change performed
        on the proxied quantities provided by this proxy to the original values
        of the quantities.

        :param value: New value of the bitmask, represented by an array of
            `bool` whose size is the same of the number of DMD modes.
        :type value: np.ndarray
        """

        # apply changes made on the proxied values to the original values
        if self.old_bitmask is not None:
            self._original_modes[:, self.old_bitmask] = self.modes
            self._original_eigs[self.old_bitmask] = self.eigs
            self._original_amplitudes[self.old_bitmask] = self.amplitudes

        linalg_module = build_linalg_module(self._original_modes)
        self._modes = self._original_modes[:, value]
        self._eigs = self._original_eigs[value]
        self._amplitudes = self._original_amplitudes[value]

        self.old_bitmask = value

    @property
    def modes(self):
        """
        Proxied (i.e. filtered according to the bitmask) view on the matrix
        of DMD modes.

        :return: A matrix containing the selected DMD modes.
        :rtype: np.ndarray
        """
        return self._modes

    @property
    def eigs(self):
        """
        Proxied (i.e. filtered according to the bitmask) view on the array
        of DMD eigenvalues.

        :return: An array containing the selected DMD eigenvalues.
        :rtype: np.ndarray
        """
        return self._eigs

    @property
    def amplitudes(self):
        """
        Proxied (i.e. filtered according to the bitmask) view on the array
        of DMD amplitudes.

        :return: An array containing the selected DMD amplitudes.
        :rtype: np.ndarray
        """
        return self._amplitudes


class DMDBase(object):
    """
    Dynamic Mode Decomposition base class.

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
    :param opt: If True, amplitudes are computed like in optimized DMD  (see
        :func:`~dmdbase.DMDBase._compute_amplitudes` for reference). If
        False, amplitudes are computed following the standard algorithm. If
        `opt` is an integer, it is used as the (temporal) index of the snapshot
        used to compute DMD modes amplitudes (following the standard
        algorithm).
        The reconstruction will generally be better in time instants near the
        chosen snapshot; however increasing `opt` may lead to wrong results
        when the system presents small eigenvalues. For this reason a manual
        selection of the number of eigenvalues considered for the analyisis may
        be needed (check `svd_rank`). Also setting `svd_rank` to a value
        between 0 and 1 may give better results. Default is False.
    :type opt: bool or int
    :param rescale_mode: Scale Atilde as shown in
            10.1016/j.jneumeth.2015.10.010 (section 2.4) before computing its
            eigendecomposition. None means no rescaling, 'auto' means automatic
            rescaling using singular values, otherwise the scaling factors.
    :type rescale_mode: {'auto'} or None or numpy.ndarray
    :param bool forward_backward: If True, the low-rank operator is computed
        like in fbDMD (reference: https://arxiv.org/abs/1507.02264). Default is
        False.
    :param sorted_eigs: Sort eigenvalues (and modes/dynamics accordingly) by
        magnitude if `sorted_eigs='abs'`, by real part (and then by imaginary
        part to break ties) if `sorted_eigs='real'`. Default: False.
    :type sorted_eigs: {'real', 'abs'} or False
    :param tikhonov_regularization: Tikhonov parameter for the regularization.
        If `None`, no regularization is applied, if `float`, it is used as the
        :math:`\\lambda` tikhonov parameter.
    :type tikhonov_regularization: int or float

    :cvar dict original_time: dictionary that contains information about the
        time window where the system is sampled:

           - `t0` is the time of the first input snapshot;
           - `tend` is the time of the last input snapshot;
           - `dt` is the delta time between the snapshots.

    :cvar dict dmd_time: dictionary that contains information about the time
        window where the system is reconstructed:

            - `t0` is the time of the first approximated solution;
            - `tend` is the time of the last approximated solution;
            - `dt` is the delta time between the approximated solutions.

    """

    def __init__(
        self,
        svd_rank=0,
        tlsq_rank=0,
        exact=False,
        opt=False,
        rescale_mode=None,
        forward_backward=False,
        sorted_eigs=False,
        tikhonov_regularization=None,
    ):

        self._Atilde = DMDOperator(
            svd_rank=svd_rank,
            exact=exact,
            rescale_mode=rescale_mode,
            forward_backward=forward_backward,
            sorted_eigs=sorted_eigs,
            tikhonov_regularization=tikhonov_regularization,
        )

        self._tlsq_rank = tlsq_rank
        self._original_time = None
        self._dmd_time = None
        self._opt = opt

        self._b = None  # amplitudes
        self._snapshots = None
        self._snapshots_shape = None

        self._modes_activation_bitmask_proxy = None

    @property
    def opt(self):
        return self._opt

    @property
    def tlsq_rank(self):
        return self._tlsq_rank

    @property
    def svd_rank(self):
        return self.operator._svd_rank

    @property
    def rescale_mode(self):
        return self.operator._rescale_mode

    @property
    def exact(self):
        return self.operator._exact

    @property
    def forward_backward(self):
        return self.operator._forward_backward

    @property
    def dmd_timesteps(self):
        """
        Get the timesteps of the reconstructed states.

        :return: the time intervals of the original snapshots.
        :rtype: numpy.ndarray
        """
        linalg_module = build_linalg_module(self.eigs)
        return linalg_module.arange(
            self.dmd_time["t0"],
            self.dmd_time["tend"] + self.dmd_time["dt"],
            self.dmd_time["dt"],
        )

    @property
    def original_timesteps(self):
        """
        Get the timesteps of the original snapshot.

        :return: the time intervals of the original snapshots.
        :rtype: numpy.ndarray
        """
        linalg_module = build_linalg_module(self.eigs)
        return linalg_module.arange(
            self.original_time["t0"],
            self.original_time["tend"] + self.original_time["dt"],
            self.original_time["dt"],
        )

    def _allocate_proxy(self):
        # if this is not true, this call is probably a sub-call of some
        # get-access to self.modes (most likely in compute_amplitudes())
        if hasattr(self, "_b") and self._b is not None:
            self._modes_activation_bitmask_proxy = ActivationBitmaskProxy(self.operator, self._b)

    def _try_invalidate_bitmask_proxy(self):
        if not self._modes_activation_bitmask_proxy:
            pass
        else:
            old_id = self._modes_activation_bitmask_proxy._operator_id
            new_id = id(self.operator.as_numpy_array)
            if old_id != new_id:
                self._modes_activation_bitmask_proxy = None

    @property
    def modes(self):
        """
        Get the matrix containing the DMD modes, stored by column.

        :return: the matrix containing the DMD modes.
        :rtype: numpy.ndarray
        """
        if self.fitted:
            self._try_invalidate_bitmask_proxy()
            if not self._modes_activation_bitmask_proxy:
                self._allocate_proxy()
                # if the value is still None, it means that we cannot create
                # the proxy at the moment
                if not self._modes_activation_bitmask_proxy:
                    return self.operator.modes
            return self._modes_activation_bitmask_proxy.modes

    @property
    def atilde(self):
        """
        Get the reduced Koopman operator A, called A tilde.

        :return: the reduced Koopman operator A.
        :rtype: numpy.ndarray
        """
        return self.operator.as_array

    @property
    def operator(self):
        """
        Get the instance of DMDOperator.

        :return: the instance of DMDOperator
        :rtype: DMDOperator
        """
        return self._Atilde

    @property
    def eigs(self):
        """
        Get the eigenvalues of A tilde.

        :return: the eigenvalues from the eigendecomposition of `atilde`.
        :rtype: numpy.ndarray
        """
        if self.fitted:
            self._try_invalidate_bitmask_proxy()
            if not self._modes_activation_bitmask_proxy:
                self._allocate_proxy()
                # if the value is still None, it means that we cannot create
                # the proxy at the moment
                if not self._modes_activation_bitmask_proxy:
                    return self.operator.eigenvalues
            return self._modes_activation_bitmask_proxy.eigs

    def _translate_eigs_exponent(self, tpow):
        """
        Transforms the exponent of the eigenvalues in the dynamics formula
        according to the selected value of `self.opt` (check the documentation
        for `opt` in :func:`__init__ <dmdbase.DMDBase.__init__>`).

        :param tpow: the exponent(s) of Sigma in the original DMD formula.
        :type tpow: int or np.ndarray
        :return: the exponent(s) adjusted according to `self.opt`
        :rtype: int or np.ndarray
        """

        if isinstance(self.opt, bool):
            amplitudes_snapshot_index = 0
        else:
            amplitudes_snapshot_index = self.opt

        if amplitudes_snapshot_index < 0:
            # we take care of negative indexes: -n becomes T - n
            return tpow - (self.snapshots.shape[1] + amplitudes_snapshot_index)
        else:
            return tpow - amplitudes_snapshot_index

    @property
    def dynamics(self):
        """
        Get the time evolution of each mode.

        .. math::

            \\mathbf{x}(t) \\approx
            \\sum_{k=1}^{r} \\boldsymbol{\\phi}_{k} \\exp \\left( \\omega_{k} t
            \\right) b_{k} = \\sum_{k=1}^{r} \\boldsymbol{\\phi}_{k} \\left(
            \\lambda_{k} \\right)^{\\left( t / \\Delta t \\right)} b_{k}

        :return: the matrix that contains all the time evolution, stored by
            row.
        :rtype: numpy.ndarray
        """
        linalg_module = build_linalg_module(self.eigs)
        temp = linalg_module.repeat(
            self.eigs[:, None], self.dmd_timesteps.shape[0], axis=1
        )
        tpow = old_div(
            self.dmd_timesteps - self.original_time["t0"],
            self.original_time["dt"],
        )

        # The new formula is x_(k+j) = \Phi \Lambda^k \Phi^(-1) x_j.
        # Since j is fixed, for a given snapshot "u" we have the following
        # formula:
        # x_u = \Phi \Lambda^{u-j} \Phi^(-1) x_j
        # Therefore tpow must be scaled appropriately.
        tpow = self._translate_eigs_exponent(tpow)

        return linalg_module.pow(temp, tpow) * self.amplitudes[:, None]

    @property
    def reconstructed_data(self):
        """
        Get the reconstructed data.

        :return: the matrix that contains the reconstructed snapshots.
        :rtype: numpy.ndarray
        """
        linalg_module = build_linalg_module(self.modes)
        return linalg_module.dot(self.modes, self.dynamics)

    @property
    def snapshots(self):
        """
        Get the original input data.

        :return: the matrix that contains the original snapshots.
        :rtype: numpy.ndarray
        """
        return self._snapshots

    @property
    def frequency(self):
        """
        Get the amplitude spectrum.

        :return: the array that contains the frequencies of the eigenvalues.
        :rtype: numpy.ndarray
        """
        linalg_module = self.eigs
        div = 2 * np.pi * self.original_time["dt"]
        return linalg_module.log(self.eigs).imag / div

    @property
    def growth_rate(self):  # To check
        """
        Get the growth rate values relative to the modes.

        :return: the Floquet values
        :rtype: numpy.ndarray
        """
        return self.eigs.real / self.original_time["dt"]

    @property
    def amplitudes(self):
        """
        Get the coefficients that minimize the error between the original
        system and the reconstructed one. For futher information, see
        `dmdbase._compute_amplitudes`.

        :return: the array that contains the amplitudes coefficient.
        :rtype: numpy.ndarray
        """
        if self.fitted:
            self._try_invalidate_bitmask_proxy()
            if not self._modes_activation_bitmask_proxy:
                self._allocate_proxy()
            return self._modes_activation_bitmask_proxy.amplitudes

    @property
    def fitted(self):
        """Check whether this DMD instance has been fitted.

        :return: `True` is the instance has been fitted, `False` otherwise.
        :rtype: bool
        """
        try:
            return self.operator.modes is not None
        except (ValueError, AttributeError):
            return False

    @property
    def modes_activation_bitmask(self):
        """
        Get the bitmask which controls which DMD modes are enabled at the
        moment in this DMD instance.

        The DMD instance must be fitted before this property becomes valid.
        After :func:`fit` is called, the defalt value of
        `modes_activation_bitmask` is an array of `True` values of the same
        shape of :func:`amplitudes`.

        The array returned is read-only (this allow us to react appropriately
        to changes in the bitmask). In order to modify the bitmask you need to
        set the field to a brand-new value (see example below).

        Example:

        .. code-block:: python

            >>> # this is an error
            >>> dmd.modes_activation_bitmask[[1,2]] = False
            ValueError: assignment destination is read-only
            >>> tmp = np.array(dmd.modes_activation_bitmask)
            >>> tmp[[1,2]] = False
            >>> dmd.modes_activation_bitmask = tmp

        :return: The DMD modes activation bitmask.
        :rtype: numpy.ndarray
        """
        # check that the DMD was fitted
        if not self.fitted:
            raise RuntimeError("This DMD instance has not been fitted yet.")

        if not self._modes_activation_bitmask_proxy:
            self._allocate_proxy()

        bitmask = self._modes_activation_bitmask_proxy.old_bitmask
        # make sure that the array is immutable
        build_linalg_module(bitmask).make_not_writeable(bitmask)
        return bitmask

    @modes_activation_bitmask.setter
    def modes_activation_bitmask(self, value):
        # check that the DMD was fitted
        if not self.fitted:
            raise RuntimeError("This DMD instance has not been fitted yet.")

        if value.dtype != bool:
            raise RuntimeError(
                "Unxpected dtype, expected bool, got {}.".format(value.dtype)
            )

        # check that the shape is correct
        if value.shape != self.modes_activation_bitmask.shape:
            raise ValueError(
                "Expected shape {}, got {}".format(
                    self.modes_activation_bitmask.shape, value.shape
                )
            )

        self._modes_activation_bitmask_proxy.change_bitmask(value)

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
        :rtype: DMDBase
        """

        if isinstance(key, (slice, int, list, np.ndarray)):
            filter_function = lambda x: isinstance(x, int)

            if isinstance(key, (list, np.ndarray)):
                if not all(map(filter_function, key)):
                    raise ValueError(
                        "Invalid argument type, expected a slice, an int, or "
                        "a list of indexes."
                    )
                # no repeated elements
                if len(key) != len(set(key)):
                    raise ValueError("Repeated indexes are not supported.")
        else:
            raise ValueError(
                "Invalid argument type, expected a slice, an int, or a list "
                "of indexes, got {}".format(type(key))
            )

        mask = np.full(self.modes_activation_bitmask.shape, False)
        mask[key] = True

        shallow_copy = copy(self)
        shallow_copy._allocate_proxy()
        shallow_copy.modes_activation_bitmask = mask

        return shallow_copy

    @property
    def original_time(self):
        """
        A dictionary which contains information about the time window used to
        fit this DMD instance.

        Inside the dictionary:

        ======  ====================================================================================
        Key     Value
        ======  ====================================================================================
        `t0`    Time of the first input snapshot (0 by default).
        `tend`  Time of the last input snapshot (usually corresponds to the number of snapshots).
        `dt`    Timestep between two snapshots (1 by default).
        ======  ====================================================================================

        :return: A dict which contains info about the input time frame.
        :rtype: dict
        """
        if self._original_time is None:
            raise RuntimeError(
                """
_set_initial_time_dictionary() has not been called, did you call fit()?"""
            )
        return self._original_time

    @property
    def dmd_time(self):
        """
        A dictionary which contains information about the time window used to
        reconstruct/predict using this DMD instance. By default this is equal
        to :func:`original_time`.

        Inside the dictionary:

        ======  ====================================================================================
        Key     Value
        ======  ====================================================================================
        `t0`    Time of the first output snapshot.
        `tend`  Time of the last output snapshot.
        `dt`    Timestep between two snapshots.
        ======  ====================================================================================

        :return: A dict which contains info about the input time frame.
        :rtype: dict
        """
        if self._dmd_time is None:
            raise RuntimeError(
                """
_set_initial_time_dictionary() has not been called, did you call fit()?"""
            )
        return self._dmd_time

    @dmd_time.setter
    def dmd_time(self, value):
        self._dmd_time = deepcopy(value)

    def _set_initial_time_dictionary(self, time_dict):
        """
        Set the initial values for the class fields `time_dict` and
        `original_time`. This is usually called in `fit()` and never again.

        :param time_dict: Initial time dictionary for this DMD instance.
        :type time_dict: dict
        """
        if not (
            "t0" in time_dict and "tend" in time_dict and "dt" in time_dict
        ):
            raise ValueError(
                'time_dict must contain the keys "t0", "tend" and "dt".'
            )
        if len(time_dict) > 3:
            raise ValueError(
                'time_dict must contain only the keys "t0", "tend" and "dt".'
            )

        self._original_time = DMDTimeDict(dict(time_dict))
        self._dmd_time = DMDTimeDict(dict(time_dict))

    def fit(self, X):
        """
        Abstract method to fit the snapshots matrices.

        Not implemented, it has to be implemented in subclasses.
        """
        raise NotImplementedError(
            "Subclass must implement abstract method {}.fit".format(
                self.__class__.__name__
            )
        )

    def save(self, fname):
        """
        Save the object to `fname` using the pickle module.

        :param str fname: the name of file where the reduced order model will
            be saved.

        Example:

        >>> from pydmd import DMD
        >>> dmd = DMD(...) #  Construct here the rom
        >>> dmd.fit(...)
        >>> dmd.save('pydmd.dmd')
        """
        with open(fname, "wb") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(fname):
        """
        Load the object from `fname` using the pickle module.

        :return: The `ReducedOrderModel` loaded

        Example:

        >>> from pydmd import DMD
        >>> dmd = DMD.load('pydmd.dmd')
        >>> print(dmd.reconstructed_data)
        """
        with open(fname, "rb") as output:
            dmd = pickle.load(output)

        return dmd

    @staticmethod
    def _col_major_2darray(X):
        """
        Private method that takes as input the snapshots and stores them into a
        2D matrix, by column. If the input data is already formatted as 2D
        array, the method saves it, otherwise it also saves the original
        snapshots shape and reshapes the snapshots.

        :param X: the input snapshots.
        :type X: int or numpy.ndarray
        :return: the 2D matrix that contains the flatten snapshots
        :rtype: numpy.ndarray, tuple
        """
        snapshots = cast_as_array(X)

        linalg_module = build_linalg_module(snapshots)
        snapshots = linalg_module.atleast_2d(snapshots)
        n_snapshots, *snapshots_shape = snapshots.shape

        snapshots_2d = snapshots.reshape((n_snapshots, -1))
        # when snapshots are wrapped in a list each member of the list is
        # a snapshot
        if not is_array(X):
            snapshots_2d = snapshots_2d.T

        # check condition number of the data passed in
        cond_number = linalg_module.cond(snapshots_2d)
        if cond_number > 10e4:
            warnings.warn(
                "Input data matrix X has condition number {}. "
                """Consider preprocessing data, passing in augmented data
matrix, or regularization methods.""".format(
                    cond_number
                )
            )

        return snapshots_2d, snapshots_shape

    def _optimal_dmd_matrices(self):
        # compute the vandermonde matrix
        linalg_module = build_linalg_module(self.eigs)
        vander = linalg_module.vander(self.eigs, len(self.dmd_timesteps), True)

        a = linalg_module.dot(self.modes.conj().T, self.modes)
        b = linalg_module.dot(vander, vander.conj().T).conj()
        P = linalg_module.dot(a, b)

        if self.exact:
            q = (
                linalg_module.diag(
                    linalg_module.multi_dot(
                        (vander, self._snapshots.conj().T, self.modes)
                    )
                )
            ).conj()
        else:
            U, s, V = compute_svd(self._snapshots[:, :-1], self.svd_rank)

            q = (
                linalg_module.diag(
                    linalg_module.linalg.multi_dot(
                        [
                            vander[:, :-1],
                            V,
                            linalg_module.diag(s).conj(),
                            self.operator.eigenvectors,
                        ]
                    )
                )
            ).conj()

        return P, q

    def _compute_amplitudes(self):
        """
        Compute the amplitude coefficients. If `self.opt` is False the
        amplitudes are computed by minimizing the error between the modes and
        the first snapshot; if `self.opt` is True the amplitudes are computed
        by minimizing the error between the modes and all the snapshots, at the
        expense of bigger computational cost.

        This method uses the class variables self._snapshots (for the
        snapshots), self.modes and self.eigs.

        :return: the amplitudes array
        :rtype: numpy.ndarray

        References for optimal amplitudes:
        Jovanovic et al. 2014, Sparsity-promoting dynamic mode decomposition,
        https://hal-polytechnique.archives-ouvertes.fr/hal-00995141/document
        """
        if isinstance(self.opt, bool) and self.opt:
            # b optimal
            A, b = self._optimal_dmd_matrices()
            if not same_linalg_type(A, b):
                raise ValueError(
                    "A and b should belong to the same module. A: {}, b: {}".format(
                        type(A), type(b)
                    )
                )

            linalg_module = build_linalg_module(A)
            a = linalg_module.solve(A, b)
        else:
            if isinstance(self.opt, bool):
                amplitudes_snapshot_index = 0
            else:
                amplitudes_snapshot_index = self.opt

            linalg_module = build_linalg_module(self.modes)
            a = linalg_module.lstsq(
                self.modes,
                self._snapshots.T[amplitudes_snapshot_index],
                rcond=None,
            )[0]

        return a

    def _enforce_ratio(self, goal_ratio, supx, infx, supy, infy):
        """
        Computes the right value of `supx,infx,supy,infy` to obtain the desired
        ratio in :func:`plot_eigs`. Ratio is defined as
        ::
            dx = supx - infx
            dy = supy - infy
            max(dx,dy) / min(dx,dy)

        :param float goal_ratio: the desired ratio.
        :param float supx: the old value of `supx`, to be adjusted.
        :param float infx: the old value of `infx`, to be adjusted.
        :param float supy: the old value of `supy`, to be adjusted.
        :param float infy: the old value of `infy`, to be adjusted.
        :return tuple: a tuple which contains the updated values of
            `supx,infx,supy,infy` in this order.
        """

        dx = supx - infx
        if dx == 0:
            dx = 1.0e-16
        dy = supy - infy
        if dy == 0:
            dy = 1.0e-16
        ratio = max(dx, dy) / min(dx, dy)

        if ratio >= goal_ratio:
            if dx < dy:
                goal_size = dy / goal_ratio

                supx += (goal_size - dx) / 2
                infx -= (goal_size - dx) / 2
            elif dy < dx:
                goal_size = dx / goal_ratio

                supy += (goal_size - dy) / 2
                infy -= (goal_size - dy) / 2

        return (supx, infx, supy, infy)

    def _plot_limits(self, narrow_view):
        if narrow_view:
            supx = max(self.eigs.real) + 0.05
            infx = min(self.eigs.real) - 0.05

            supy = max(self.eigs.imag) + 0.05
            infy = min(self.eigs.imag) - 0.05

            return self._enforce_ratio(8, supx, infx, supy, infy)
        else:
            linalg_module = build_linalg_module(self.eigs)
            return linalg_module.ceil(linalg_module.abs(self.eigs)).max()

    def plot_eigs(
        self,
        show_axes=True,
        show_unit_circle=True,
        figsize=(8, 8),
        title="",
        narrow_view=False,
        dpi=None,
        filename=None,
    ):
        """
        Plot the eigenvalues.
        :param bool show_axes: if True, the axes will be showed in the plot.
            Default is True.
        :param bool show_unit_circle: if True, the circle with unitary radius
            and center in the origin will be showed. Default is True.
        :param tuple(int,int) figsize: tuple in inches defining the figure
            size. Default is (8, 8).
        :param str title: title of the plot.
        :param narrow_view bool: if True, the plot will show only the smallest
            rectangular area which contains all the eigenvalues, with a padding
            of 0.05. Not compatible with `show_axes=True`. Default is False.
        :param dpi int: If not None, the given value is passed to
            ``plt.figure``.
        :param str filename: if specified, the plot is saved at `filename`.
        """
        if self.eigs is None:
            raise ValueError(
                "The eigenvalues have not been computed."
                "You have to call the fit() method."
            )

        if dpi is not None:
            plt.figure(figsize=figsize, dpi=dpi)
        else:
            plt.figure(figsize=figsize)

        plt.title(title)
        plt.gcf()
        ax = plt.gca()

        (points,) = ax.plot(
            self.eigs.real, self.eigs.imag, "bo", label="Eigenvalues"
        )

        if narrow_view:
            supx, infx, supy, infy = self._plot_limits(narrow_view)

            # set limits for axis
            ax.set_xlim((infx, supx))
            ax.set_ylim((infy, supy))

            # x and y axes
            if show_axes:
                endx = min(supx, 1.0)
                ax.annotate(
                    "",
                    xy=(endx, 0.0),
                    xytext=(max(infx, -1.0), 0.0),
                    arrowprops=dict(arrowstyle=("->" if endx == 1.0 else "-")),
                )

                endy = min(supy, 1.0)
                ax.annotate(
                    "",
                    xy=(0.0, endy),
                    xytext=(0.0, max(infy, -1.0)),
                    arrowprops=dict(arrowstyle=("->" if endy == 1.0 else "-")),
                )
        else:
            # set limits for axis
            limit = self._plot_limits(narrow_view)

            ax.set_xlim((-limit, limit))
            ax.set_ylim((-limit, limit))

            # x and y axes
            if show_axes:
                ax.annotate(
                    "",
                    xy=(max(limit * 0.8, 1.0), 0.0),
                    xytext=(min(-limit * 0.8, -1.0), 0.0),
                    arrowprops=dict(arrowstyle="->"),
                )
                ax.annotate(
                    "",
                    xy=(0.0, max(limit * 0.8, 1.0)),
                    xytext=(0.0, min(-limit * 0.8, -1.0)),
                    arrowprops=dict(arrowstyle="->"),
                )

        plt.ylabel("Imaginary part")
        plt.xlabel("Real part")

        if show_unit_circle:
            unit_circle = plt.Circle(
                (0.0, 0.0),
                1.0,
                color="green",
                fill=False,
                label="Unit circle",
                linestyle="--",
            )
            ax.add_artist(unit_circle)

        # Dashed grid
        gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        for line in gridlines:
            line.set_linestyle("-.")
        ax.grid(True)

        # legend
        if show_unit_circle:
            ax.add_artist(
                plt.legend(
                    [points, unit_circle],
                    ["Eigenvalues", "Unit circle"],
                    loc="best",
                )
            )
        else:
            ax.add_artist(plt.legend([points], ["Eigenvalues"], loc="best"))

        ax.set_aspect("equal")

        if filename:
            plt.savefig(filename)
        else:
            plt.show()

    def plot_modes_2D(
        self,
        index_mode=None,
        filename=None,
        x=None,
        y=None,
        order="C",
        figsize=(8, 8),
    ):
        """
        Plot the DMD Modes.

        :param index_mode: the index of the modes to plot. By default, all
            the modes are plotted.
        :type index_mode: int or sequence(int)
        :param str filename: if specified, the plot is saved at `filename`.
        :param numpy.ndarray x: domain abscissa.
        :param numpy.ndarray y: domain ordinate
        :param tuple(int,int) figsize: tuple in inches defining the figure
            size. Default is (8, 8).
        """
        if self.modes is None:
            raise ValueError(
                "The modes have not been computed."
                "You have to perform the fit method."
            )

        if x is None and y is None:
            if self._snapshots_shape is None:
                raise ValueError(
                    "No information about the original shape of the snapshots."
                )

            if len(self._snapshots_shape) != 2:
                raise ValueError(
                    "The dimension of the input snapshots is not 2D."
                )

        # If domain dimensions have not been passed as argument,
        # use the snapshots dimensions
        if x is None and y is None:
            x = np.arange(self._snapshots_shape[0])
            y = np.arange(self._snapshots_shape[1])

        xgrid, ygrid = np.meshgrid(x, y)

        if index_mode is None:
            index_mode = list(range(self.modes.shape[1]))
        elif isinstance(index_mode, int):
            index_mode = [index_mode]

        if filename:
            basename, ext = splitext(filename)

        for idx in index_mode:
            fig = plt.figure(figsize=figsize)
            fig.suptitle("DMD Mode {}".format(idx))

            real_ax = fig.add_subplot(1, 2, 1)
            imag_ax = fig.add_subplot(1, 2, 2)

            mode = self.modes.T[idx].reshape(xgrid.shape)

            real = real_ax.pcolor(
                xgrid,
                ygrid,
                mode.real,
                cmap="jet",
                vmin=mode.real.min(),
                vmax=mode.real.max(),
            )
            imag = imag_ax.pcolor(
                xgrid,
                ygrid,
                mode.imag,
                vmin=mode.imag.min(),
                vmax=mode.imag.max(),
            )

            fig.colorbar(real, ax=real_ax)
            fig.colorbar(imag, ax=imag_ax)

            real_ax.set_aspect("auto")
            imag_ax.set_aspect("auto")

            real_ax.set_title("Real")
            imag_ax.set_title("Imag")

            # padding between elements
            plt.tight_layout(pad=2.0)

            if filename:
                plt.savefig("{0}.{1}{2}".format(basename, idx, ext))
                plt.close(fig)

        if not filename:
            plt.show()

    def plot_snapshots_2D(
        self,
        index_snap=None,
        filename=None,
        x=None,
        y=None,
        order="C",
        figsize=(8, 8),
    ):
        """
        Plot the snapshots.

        :param index_snap: the index of the snapshots to plot. By default, all
            the snapshots are plotted.
        :type index_snap: int or sequence(int)
        :param str filename: if specified, the plot is saved at `filename`.
        :param numpy.ndarray x: domain abscissa.
        :param numpy.ndarray y: domain ordinate
        :param tuple(int,int) figsize: tuple in inches defining the figure
            size. Default is (8, 8).
        """
        if self._snapshots is None:
            raise ValueError("Input snapshots not found.")

        if x is None and y is None:
            if self._snapshots_shape is None:
                raise ValueError(
                    "No information about the original shape of the snapshots."
                )

            if len(self._snapshots_shape) != 2:
                raise ValueError(
                    "The dimension of the input snapshots is not 2D."
                )

        # If domain dimensions have not been passed as argument,
        # use the snapshots dimensions
        if x is None and y is None:
            x = np.arange(self._snapshots_shape[0])
            y = np.arange(self._snapshots_shape[1])

        xgrid, ygrid = np.meshgrid(x, y)

        if index_snap is None:
            index_snap = list(range(self._snapshots.shape[1]))
        elif isinstance(index_snap, int):
            index_snap = [index_snap]

        if filename:
            basename, ext = splitext(filename)

        for idx in index_snap:
            fig = plt.figure(figsize=figsize)
            fig.suptitle("Snapshot {}".format(idx))

            snapshot = self._snapshots.T[idx].real.reshape(xgrid.shape)

            contour = plt.pcolor(
                xgrid,
                ygrid,
                snapshot,
                vmin=snapshot.min(),
                vmax=snapshot.max(),
            )

            fig.colorbar(contour)

            if filename:
                plt.savefig("{0}.{1}{2}".format(basename, idx, ext))
                plt.close(fig)

        if not filename:
            plt.show()


class DMDTimeDict(dict):
    def __setitem__(self, key, value):
        if key in ["t0", "tend", "dt"]:
            dict.__setitem__(self, key, value)
        else:
            raise KeyError(
                """DMDBase.dmd_time accepts only the following keys: "t0",
"tend", "dt", {} is not allowed.""".format(
                    key
                )
            )

    def __eq__(self, o):
        if isinstance(o, dict):
            return all(map(lambda s: o[s] == self[s], ["t0", "tend", "dt"]))
        return False
