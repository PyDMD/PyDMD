"""
A module which contains several functions to tune (i.e. improve) DMD instances
through the "manual" modification of DMD modes.
"""
from copy import deepcopy
from functools import partial

import numpy as np


def select_modes(dmd, criteria, in_place=True, return_indexes=False):
    """
    Select the DMD modes by using the given `criteria`.
    `criteria` is a function which takes as input the DMD
    object itself and return a numpy.ndarray of boolean where `False`
    indicates that the corresponding mode will be discarded.
    The class :class:`ModesSelectors` contains some pre-packed selector
    functions.

    Example:

    .. code-block:: python

        >>> dmd = ...
        >>> def stable_modes(dmd):
        >>>    toll = 1e-3
        >>>    return np.abs(np.abs(dmd.eigs) - 1) < toll
        >>> select_modes(dmd, stable_modes)

    :param pydmd.DMDBase dmd: An instance of DMD from which we want to delete
        modes according to some criteria.
    :param callable criteria: The function used to select the modes. Must
        return a boolean array (whose length is the number of DMD modes in
        `dmd`) such that `True` items correspond to retained DMD modes, while
        `False` items correspond to deleted modes.
    :param bool in_place: If `True`, the given DMD instance will be modified
        according to the given `criteria`. Otherwise, a new instance will be
        created (via `copy.deepcopy`).
    :param bool return_indexes: If `True`, this function returns the indexes
        corresponding to DMD modes cut using the given `criteria` (default
        `False`).
    :returns: If `return_indexes` is `True`, the returned value is a tuple
        whose items are:

        0. The modified DMD instance;
        1. The indexes (on the old DMD instance) corresponding to DMD modes
            cut.

        Otherwise, the returned value is the modified DMD instance.
    """
    if not in_place:
        dmd = deepcopy(dmd)

    selected_indexes = np.where(criteria(dmd))[0]

    all_indexes = set(np.arange(len(dmd.eigs)))
    cut_indexes = np.array(list(all_indexes - set(selected_indexes)))

    dmd.operator._eigenvalues = dmd.operator._eigenvalues[selected_indexes]
    dmd.operator._Lambda = dmd.operator._Lambda[selected_indexes]

    dmd.operator._eigenvectors = dmd.operator._eigenvectors[
        :, selected_indexes
    ]
    dmd.operator._modes = dmd.operator._modes[:, selected_indexes]

    # TODO: should improve this [code repetition]
    dmd.operator._Atilde = np.linalg.multi_dot(
        [
            dmd.operator._eigenvectors,
            np.diag(dmd.operator._eigenvalues),
            np.linalg.pinv(dmd.operator._eigenvectors),
        ]
    )

    dmd._b = dmd._compute_amplitudes()

    if return_indexes:
        return dmd, cut_indexes
    return dmd


def stabilize_modes(
    dmd, inner_radius, outer_radius=np.inf, in_place=True, return_indexes=False
):
    """
    Stabilize modes in a circular sector of radius [`inner_radius`,
    `outer_radius`].

    Stabilizing a mode means that the corresponding eigenvalue is divided
    by its module (i.e. normalized) in order to make the associated
    dynamic a trigonometric function with respect to the time (since the
    eigenvalue is projected on the unit circle). At the same time, the
    corresponding mode amplitude is multiplied by the former module of the
    eigenvalue, in order to "recover" the correctness of the result in the
    first time instants.

    This approach may give better results in the prediction when one or
    more eigenvalues are strongly unstable (i.e. the corresponding DMD mode
    "explodes" several instants after the known time frame).

    In order to stabilize an unbounded (above) circular sector, the
    parameter `outer_radius` should be set to `np.inf` (default).

    :param pydmd.DMDBase dmd: An instance of DMD which we want to stabilize.
    :param float inner_radius: The inner radius of the circular sector to
        be stabilized.
    :param float outer_radius: The outer radius of the circular sector to
        be stabilized.
    :param bool in_place: If `True`, the given DMD instance will be modified
        according to the given `criteria`. Otherwise, a new instance will be
        created (via `copy.deepcopy`).
    :param bool return_indexes: If `True`, this function returns the indexes
        corresponding to DMD modes stabilized (default `False`).
    :returns: If `return_indexes` is `True`, the returned value is a tuple
        whose items are:

        0. The modified DMD instance;
        1. The indexes (on the old DMD instance) corresponding to DMD modes
            stabilized.

        Otherwise, the returned value is the modified DMD instance.
    """
    if not in_place:
        dmd = deepcopy(dmd)

    eigs_module = np.abs(dmd.eigs)

    # indexes associated with eigenvalues that must be stabilized
    fixable_eigs_indexes = np.logical_and(
        inner_radius < eigs_module,
        eigs_module < outer_radius,
    )

    dmd._b[fixable_eigs_indexes] *= np.abs(dmd.eigs[fixable_eigs_indexes])
    dmd.operator._eigenvalues[fixable_eigs_indexes] /= np.abs(
        dmd.eigs[fixable_eigs_indexes]
    )

    stabilized_indexes = np.where(fixable_eigs_indexes)[0]

    if return_indexes:
        return dmd, stabilized_indexes
    return dmd


class ModesSelectors:
    """
    A container class which defines some static methods for pre-packed
    modes selectors functions to be used in `select_modes`.

    For instance, to select the first `x` modes by integral contributions:

    Example:

    .. code-block:: python

        >>> from pydmd.dmd_modes_tuner import ModesSelectors, select_modes
        >>> select_modes(dmd, ModesSelectors.integral_contribution(x))

    Most private static methods in this class are "non-partialized", which
    means that they also take the parameters that characterize the selector.
    By contrast, public static method are ready mode selector, whose only
    parameter is the DMD instance on which that selector should be applied, and
    are the output of a call to `functools.partial` applied to a
    non-partialized selector. This mechanism is employed to reduce the
    boilerplate code needed while applying a selector.
    """

    @staticmethod
    def _threshold(dmd, low_threshold, up_threshold):
        """
        Non-partialized function of the modes selector `threshold`.

        :param DMDBase dmd: An instance of DMDBase.
        :param float low_threshold: The minimum accepted module of an
            eigenvalue.
        :param float up_threshold: The maximum accepted module of an
            eigenvalue.
        :return np.ndarray: An array of bool, where each "True" index means
            that the corresponding DMD mode is selected.
        """
        eigs_module = np.abs(dmd.eigs)

        return np.logical_and(
            eigs_module < up_threshold,
            eigs_module > low_threshold,
        )

    @staticmethod
    def threshold(low_threshold, up_threshold):
        """
        Retain only DMD modes associated with an eigenvalue whose module is
        between `low_threshold` and `up_threshold` (inclusive on both sides).

        :param float low_threshold: The minimum accepted module of an
            eigenvalue.
        :param float up_threshold: The maximum accepted module of an
            eigenvalue.
        :return np.ndarray: An array of bool, where each "True" index means
            that the corresponding DMD mode is selected.
        """
        return partial(
            ModesSelectors._threshold,
            low_threshold=low_threshold,
            up_threshold=up_threshold,
        )

    @staticmethod
    def _stable_modes(
        dmd,
        max_distance_from_unity_inside,
        max_distance_from_unity_outside,
    ):
        """
        Non-partialized function of the modes selector `stable_modes`.

        :param DMDBase dmd: An instance of DMDBase.
        :param float max_distance_from_unity_inside: The maximum distance
            from the unit circle for points inside it.
        :param float max_distance_from_unity_outside: The maximum distance
            from the unit circle for points outside it.
        :return np.ndarray: An array of bool, where each "True" index means
            that the corresponding DMD mode is selected.
        """
        return ModesSelectors._threshold(
            dmd,
            1 - max_distance_from_unity_inside,
            1 + max_distance_from_unity_outside,
        )

    @staticmethod
    def stable_modes(
        max_distance_from_unity=None,
        max_distance_from_unity_inside=None,
        max_distance_from_unity_outside=None,
    ):
        """
        Select all the modes corresponding to eigenvalues whose distance
        from the unit circle is less than or equal to a specified threshold. It
        is possible to specify the distance separately for eigenvalues inside
        and outside the unit circle, but you cannot set clashing
        thresholds.

        The following are allowed combinations of parameters:

        .. code-block:: python

            >>> # the maximum allowed distance from the unit circle (both
            ... # inside and outside) is 1.e-3.
            >>> stable_modes(max_distance_from_unity=1.e-3)
            >>> # the maximum allowed distance from the unit circle is 1.e-3
            ... # inside and 1.e-4 outside.
            >>> stable_modes(max_distance_from_unity_inside=1.e-3,
            ...   max_distance_from_unity_outside=1.e-4)
            >>> # the maximum allowed distance from the unit circle is 1.e-4
            ... # outside and unspecified (i.e. infinity) inside.
            >>> stable_modes(max_distance_from_unity_outside=1.e-4)

        Since `max_distance_from_unity` controls both inside and outside
        distance, you cannot set also `max_distance_from_unity_inside` or
        `max_distance_from_unity_outside` simultaneously:

        >>> # this is not allowed
        >>> stable_modes(max_distance_from_unity=1.e-3,
        ...     max_distance_from_unity_inside=1.e-4)

        For code clarity reasons, the snippet above would have failed even if
        `max_distance_from_unity_inside=1.e-3`.

        :param float max_distance_from_unity: The maximum distance from the
            unit circle. Defaults to `None`.
        :param float max_distance_from_unity_inside: The maximum distance
            from the unit circle for points inside it. Defaults to `None`.
        :param float max_distance_from_unity_outside: The maximum distance
            from the unit circle for points outside it. Defaults to `None`.
        :return callable: A function which can be used as the parameter
            of `select_modes` to select DMD modes according to
            the criteria of stability.
        """

        if max_distance_from_unity and max_distance_from_unity_inside:
            raise ValueError(
                """Only one between `max_distance_from_unity`
and `max_distance_from_unity_inside` can be not `None`"""
            )
        if max_distance_from_unity and max_distance_from_unity_outside:
            raise ValueError(
                """Only one between `max_distance_from_unity`
and `max_distance_from_unity_outside` can be not `None`"""
            )

        if max_distance_from_unity:
            max_distance_from_unity_outside = max_distance_from_unity
            max_distance_from_unity_inside = max_distance_from_unity

        if max_distance_from_unity_outside is None:
            max_distance_from_unity_outside = float("inf")
        if max_distance_from_unity_inside is None:
            max_distance_from_unity_inside = float("inf")

        if max_distance_from_unity_outside == float(
            "inf"
        ) and max_distance_from_unity_inside == float("inf"):
            raise ValueError(
                """The combination of parameters does not make sense"""
            )

        return partial(
            ModesSelectors._stable_modes,
            max_distance_from_unity_inside=max_distance_from_unity_inside,
            max_distance_from_unity_outside=max_distance_from_unity_outside,
        )

    @staticmethod
    def _compute_integral_contribution(mode, dynamic):
        """
        Compute the integral contribution across time of the given DMD mode,
        given the mode and its dynamic, as shown in
        http://dx.doi.org/10.1016/j.euromechflu.2016.11.015

        :param numpy.ndarray mode: The DMD mode.
        :param numpy.ndarray dynamic: The dynamic of the given DMD mode, as
            returned by `dmd.dynamics[mode_index]`.
        :return float: the integral contribution of the given DMD mode.
        """
        return pow(np.linalg.norm(mode), 2) * sum(np.abs(dynamic))

    @staticmethod
    def _integral_contribution(dmd, n):
        """
        Non-partialized function of the modes selector `integral_contribution`.

        :param DMDBase dmd: An instance of DMDBase.
        :param int n: The number of DMD modes to be selected.
        :return np.ndarray: An array of bool, where each "True" index means
            that the corresponding DMD mode is selected.
        """

        # temporary reset dmd_time to original_time
        temp = dmd.dmd_time
        dmd.dmd_time = dmd.original_time

        dynamics = dmd.dynamics
        modes = dmd.modes

        # reset dmd_time
        dmd.dmd_time = temp

        n_of_modes = modes.shape[1]
        integral_contributions = [
            ModesSelectors._compute_integral_contribution(*tp)
            for tp in zip(modes.T, dynamics)
        ]

        indexes_first_n = np.array(integral_contributions).argsort()[-n:]

        truefalse_array = np.array([False for _ in range(n_of_modes)])
        truefalse_array[indexes_first_n] = True
        return truefalse_array

    @staticmethod
    def integral_contribution(n):
        """
        Reference: http://dx.doi.org/10.1016/j.euromechflu.2016.11.015

        :param int n: The number of DMD modes to be selected.
        :return callable: A function which can be used as the parameter
            of `select_modes` to select DMD modes according to
            the criteria of integral contribution.
        """
        return partial(ModesSelectors._integral_contribution, n=n)


selectors = {
    "module_threshold": ModesSelectors.threshold,
    "stable_modes": ModesSelectors.stable_modes,
    "integral_contribution": ModesSelectors.integral_contribution,
}


class ModesTuner:
    """Class for semi-automatic tuning of DMD modes.

    This class generates a new instance from the instance passed to the
    constructor, and modifies that one whenever one of the tuning methods
    is called. Therefore there is no need to worry about subsequent
    unwanted changes in the given instance.

    `ModesTuner` provides a simplified interface to the tuning functions
    :func:`select_modes` and :func:`stabilize_modes`, but in order to
    have more control on what is happening (i.e. when to use in-place
    tuning, or to check which modes have been changed) you may prefer to
    use them instead.

    :param dmd: An instance of DMD (will be copied via `deepcopy`,
        therefore the given reference won't be modified afterwards).
    :type dmd: pydmd.DMDBase
    """

    def __init__(self, dmd):
        self._dmd = deepcopy(dmd)

    @property
    def dmd(self):
        """Returns the private DMD instance that `ModesTuner` is working on.
        Be aware that this instance is the internal instance owned by
        `ModesTuner`, therefore it is going to be modified by calls to tuning
        methods.

        :return: The private DMD instance owned by `ModesTuner`.
        :rtype: pydmd.DMDBase
        """
        return self._dmd

    @property
    def secure_copy(self):
        """Returns a deep copy of the private DMD instance that `ModesTuner` is
        working on. This is not going to be modified by calls to tuning
        methods.

        :return: A copy of the private DMD instance owned by `ModesTuner`.
        :rtype: pydmd.DMDBase
        """
        return deepcopy(self.dmd)

    def select(self, criteria, **kwargs):
        r"""
        Select the DMD modes by using the given `criteria`, which can be either
        a string or a function. You can choose pre-packed criteria by passing
        one of the allowed string values for criteria. In this case you need to
        pass (as keyword arguments) the arguments needed to construct the
        criteria (see example below).

        Allowed string values for `criteria`:

        * `'module_threshold'`: Retain modes such that the module of the corresponding eigenvalue is included in the interval [`low_threshold`, `up_threshold`] (cfr. :func:`ModesSelectors.threshold`);
        * `'stable_modes'`: Retain modes such that the corresponding eigenvalue is not far from the unit circle (cfr. :func:`ModesSelectors.stable_modes`);
        * `'integral_contribution'`: Retain the first `n` modes in terms of integral contribution (cfr. :func:`ModesSelectors.integral_contribution`).

        You might want to read the documentation of
        :class:`ModesSelectors` in order to get detailed info regarding the
        behavior of each argument.

        Example:

        .. code-block:: python

            >>> from pydmd.dmd_modes_tuner import ModesTuner
            >>> mtuner = ModesTuner(dmd)
            >>> mtuner.select('stable_modes', max_distance_from_unity_inside=1.e-1,
                    max_distance_from_unity_outside=1.e-3)

        :param criteria: Criteria used to select DMD modes. The allowed strings
            are `module_threshold`, `stable_modes` and `integral_contribution`.
            If `criteria` is a function it must take an instance of DMD as the
            only parameter.
        :type criteria: str or callable
        :param \**kwargs: Parameters passed to the chosen criteria (if
            `criteria` is a string).
        """

        if isinstance(criteria, str):
            if criteria not in selectors:
                raise ValueError("Could't find the specified criteria")
            criteria = selectors[criteria](**kwargs)
        if not callable(criteria):
            raise ValueError("""You should provide a criteria to select DMD
modes (either a string or a function)""")

        select_modes(self.dmd, criteria)

    def stabilize(self, inner_radius, outer_radius=np.inf):
        """
        Stabilize modes in a circular sector of radius [`inner_radius`,
        `outer_radius`].

        Stabilizing a mode means that the corresponding eigenvalue is divided
        by its module (i.e. normalized) in order to make the associated
        dynamic a trigonometric function with respect to the time (since the
        eigenvalue is projected on the unit circle). At the same time, the
        corresponding mode amplitude is multiplied by the former module of the
        eigenvalue, in order to "recover" the correctness of the result in the
        first time instants.

        This approach may give better results in the prediction when one or
        more eigenvalues are strongly unstable (i.e. the corresponding DMD mode
        "explodes" several instants after the known time frame).

        In order to stabilize an unbounded (above) circular sector, the
        parameter `outer_radius` should be set to `np.inf` (default).

        :param float inner_radius: The inner radius of the circular sector to
            be stabilized.
        :param float outer_radius: The outer radius of the circular sector to
            be stabilized.
        """

        stabilize_modes(self.dmd, inner_radius, outer_radius)
