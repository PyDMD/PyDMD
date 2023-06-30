"""
Base module for the DMD: `fit` method must be implemented in inherited classes
"""
import pickle
from copy import copy, deepcopy

import numpy as np

from .dmdoperator import DMDOperator
from .utils import compute_svd


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

    :param dmd_operator: DMD operator to be proxied.
    :type dmd_operator: DMDOperator
    :param amplitudes: DMD amplitudes.
    :type amplitudes: np.ndarray
    """

    def __init__(self, dmd_operator, amplitudes):
        self._original_modes = dmd_operator.modes
        self._original_eigs = np.atleast_1d(dmd_operator.eigenvalues)
        self._original_amplitudes = np.atleast_1d(amplitudes)

        self.old_bitmask = None
        self.change_bitmask(np.full(len(dmd_operator.eigenvalues), True))

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

        self._modes = np.array(self._original_modes)[:, value]
        self._eigs = np.array(self._original_eigs)[value]
        self._amplitudes = np.array(self._original_amplitudes)[value]

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


class DMDBase:
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
        self._exact = exact

        self._b = None  # amplitudes
        self._snapshots_holder = None

        self._modes_activation_bitmask_proxy = None

    @property
    def dmd_timesteps(self):
        """
        Get the timesteps of the reconstructed states.

        :return: the time intervals of the original snapshots.
        :rtype: numpy.ndarray
        """
        return np.arange(
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
        return np.arange(
            self.original_time["t0"],
            self.original_time["tend"] + self.original_time["dt"],
            self.original_time["dt"],
        )

    @property
    def modes(self):
        """
        Get the matrix containing the DMD modes, stored by column.

        :return: the matrix containing the DMD modes.
        :rtype: numpy.ndarray
        """
        if self.fitted:
            if not self._modes_activation_bitmask_proxy:
                self._allocate_modes_bitmask_proxy()
                # if the value is still None, it means that we cannot create
                # the proxy at the moment
                if not self._modes_activation_bitmask_proxy:
                    return self.operator.modes
            return self._modes_activation_bitmask_proxy.modes

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
            if not self._modes_activation_bitmask_proxy:
                self._allocate_modes_bitmask_proxy()
                # if the value is still None, it means that we cannot create
                # the proxy at the moment
                if not self._modes_activation_bitmask_proxy:
                    return self.operator.eigenvalues
            return self._modes_activation_bitmask_proxy.eigs

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
        temp = np.repeat(
            self.eigs[:, None], self.dmd_timesteps.shape[0], axis=1
        )
        tpow = (
            self.dmd_timesteps - self.original_time["t0"]
        ) / self.original_time["dt"]

        # The new formula is x_(k+j) = \Phi \Lambda^k \Phi^(-1) x_j.
        # Since j is fixed, for a given snapshot "u" we have the following
        # formula:
        # x_u = \Phi \Lambda^{u-j} \Phi^(-1) x_j
        # Therefore tpow must be scaled appropriately.
        tpow = self._translate_eigs_exponent(tpow)

        return np.power(temp, tpow) * self.amplitudes[:, None]

    def _translate_eigs_exponent(self, tpow):
        """
        Transforms the exponent of the eigenvalues in the dynamics formula
        according to the selected value of `self._opt` (check the documentation
        for `opt` in :func:`__init__ <dmdbase.DMDBase.__init__>`).

        :param tpow: the exponent(s) of Sigma in the original DMD formula.
        :type tpow: int or np.ndarray
        :return: the exponent(s) adjusted according to `self._opt`
        :rtype: int or np.ndarray
        """

        if isinstance(self._opt, bool):
            amplitudes_snapshot_index = 0
        else:
            amplitudes_snapshot_index = self._opt

        if amplitudes_snapshot_index < 0:
            # we take care of negative indexes: -n becomes T - n
            return tpow - (self.snapshots.shape[1] + amplitudes_snapshot_index)
        else:
            return tpow - amplitudes_snapshot_index

    @property
    def reconstructed_data(self):
        """
        Get the reconstructed data.

        :return: the matrix that contains the reconstructed snapshots.
        :rtype: numpy.ndarray
        """
        return self.modes.dot(self.dynamics)

    @property
    def snapshots(self):
        """
        Get the input data (space flattened).

        :return: the matrix that contains the flattened snapshots.
        :rtype: numpy.ndarray
        """
        if self._snapshots_holder:
            return self._snapshots_holder.snapshots
        return None

    @property
    def snapshots_shape(self):
        """
        Get the original input snapshot shape.

        :return: input snapshots shape.
        :rtype: tuple
        """
        if self._snapshots_holder:
            return self._snapshots_holder.snapshots_shape
        return None

    @property
    def frequency(self):
        """
        Get the amplitude spectrum.

        :return: the array that contains the frequencies of the eigenvalues.
        :rtype: numpy.ndarray
        """
        return np.log(self.eigs).imag / (2 * np.pi * self.original_time["dt"])

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
            if not self._modes_activation_bitmask_proxy:
                self._allocate_modes_bitmask_proxy()
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
            self._allocate_modes_bitmask_proxy()

        bitmask = self._modes_activation_bitmask_proxy.old_bitmask
        # make sure that the array is immutable
        bitmask.flags.writeable = False
        return bitmask

    @modes_activation_bitmask.setter
    def modes_activation_bitmask(self, value):
        # check that the DMD was fitted
        if not self.fitted:
            raise RuntimeError("This DMD instance has not been fitted yet.")

        value = np.array(value)
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

    def _allocate_modes_bitmask_proxy(self):
        """
        Utility method which allocates the activation bitmask proxy using the
        quantities that are currently available in this DMD instance. Fails
        quietly if the amplitudes are not set.
        """
        if hasattr(self, "_b") and self._b is not None:
            self._modes_activation_bitmask_proxy = ActivationBitmaskProxy(
                self.operator, self._b
            )

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
        shallow_copy._allocate_modes_bitmask_proxy()
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
        name = self.__class__.__name__
        msg = f"Subclass must implement abstract method {name}.fit"
        raise NotImplementedError(msg)

    def _reset(self):
        """
        Reset this instance. Should be called in :func:`fit`.
        """
        self._modes_activation_bitmask_proxy = None
        self._b = None
        self._snapshots_holder = None

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
            return pickle.load(output)

    def _optimal_dmd_matrices(self):
        # compute the vandermonde matrix
        vander = np.vander(self.eigs, len(self.dmd_timesteps), True)

        P = np.multiply(
            np.dot(self.modes.conj().T, self.modes),
            np.conj(np.dot(vander, vander.conj().T)),
        )

        if self._exact:
            q = np.conj(
                np.diag(
                    np.linalg.multi_dot(
                        [vander, self.snapshots.conj().T, self.modes]
                    )
                )
            )
        else:
            _, s, V = compute_svd(self.snapshots[:, :-1], self.modes.shape[-1])

            q = np.conj(
                np.diag(
                    np.linalg.multi_dot(
                        [
                            vander[:, :-1],
                            V,
                            np.diag(s).conj(),
                            self.operator.eigenvectors,
                        ]
                    )
                )
            )

        return P, q

    def _compute_amplitudes(self):
        """
        Compute the amplitude coefficients. If `self._opt` is False the
        amplitudes are computed by minimizing the error between the modes and
        the first snapshot; if `self._opt` is True the amplitudes are computed
        by minimizing the error between the modes and all the snapshots, at the
        expense of bigger computational cost.

        This method uses the class variables self.snapshots (for the
        snapshots), self.modes and self.eigs.

        :return: the amplitudes array
        :rtype: numpy.ndarray

        References for optimal amplitudes:
        Jovanovic et al. 2014, Sparsity-promoting dynamic mode decomposition,
        https://hal-polytechnique.archives-ouvertes.fr/hal-00995141/document
        """
        if isinstance(self._opt, bool) and self._opt:
            # b optimal
            a = np.linalg.solve(*self._optimal_dmd_matrices())
        else:
            if isinstance(self._opt, bool):
                amplitudes_snapshot_index = 0
            else:
                amplitudes_snapshot_index = self._opt

            a = np.linalg.lstsq(
                self.modes,
                self.snapshots.T[amplitudes_snapshot_index],
                rcond=None,
            )[0]

        return a


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
