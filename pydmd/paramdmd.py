"""
Module for the parametric Dynamic Mode Decomposition.
"""
import pickle
import numpy as np


class ParametricDMD:
    """
    Implementation of the parametric Dynamic Mode Decomposition proposed in
    arXiv:2110.09155v1. Both the *monolithic* and *partitioned* approaches are
    available, see the documentation of the parameter `dmd` for more details.

    :param dmd: Instance(s) of :class:`dmdbase.DMDBase`, used by the
        paramtric DMD for the prediction of future spatial modal coefficients.
        If `dmd` is a `list` the *partitioned* approach is selected, in this
        case the number of parameters in the training set should be equal to
        the number of DMD instances provided. If `dmd` is not a list, we employ
        the monolithic approach.
    :type dmd: DMDBase
    :param spatial_pod: Instance of an object usable for the generation of a
        ROM of the given dataset (see for instance the class
        `POD <https://mathlab.github.io/EZyRB/pod.html>`_ from the Python
        library `EZyRB <https://github.com/mathLab/EZyRB>`_).
    :param approximation: An interpolator following the standard
        learning-prediction pattern (`fit()` -> `predict()`). For some
        convenient wrappers see those implemented in
        `EZyRB <https://github.com/mathLab/EZyRB>`_).
    """

    def __init__(self, dmd, spatial_pod, approximation):
        self._dmd = dmd
        self._spatial_pod = spatial_pod
        self._approximation = approximation

        self._training_parameters = None
        self._parameters = None
        self._ntrain = None
        self._time_instants = None
        self._space_dim = None

    @property
    def is_partitioned(self):
        """
        Return `True` if this instance is partitioned, `False` if it is
        monolithic.

        :type: bool
        """
        return self._dmd is not None and isinstance(self._dmd, list)

    @property
    def _reference_dmd(self):
        """
        An object used as a reference for several properties like
        :func:`dmd_time` and :func:`dmd_timesteps`. If this instance is
        monolithic the returned value is `self._dmd`, otherwise it is the first
        item of the list `self._dmd`.

        :return: The object used as a reference.
        :rtype: pydmd.DMDBase
        """
        if self.is_partitioned:
            return self._dmd[0]
        return self._dmd

    @property
    def dmd_time(self):
        """
        The time dictionary used by the reference DMD instance (see also
        :func:`_reference_dmd`).

        :getter: Return the time dictionary used by the reference DMD instance.
        :setter: Set the given time dictionary in the field `dmd_time` for all
            DMD instances.
        :type: pydmd.dmdbase.DMDTimeDict
        """
        return self._reference_dmd.dmd_time

    @dmd_time.setter
    def dmd_time(self, value):
        if isinstance(self._dmd, list):
            for dmd in self._dmd:
                dmd.dmd_time = value
        else:
            self._reference_dmd.dmd_time = value

    @property
    def dmd_timesteps(self):
        """
        The timesteps in the output of this instance, which coincides with the
        timesteps in the output of the reference of this instance (see
        :func:`_reference_dmd`).

        :return: The timesteps in the output of this instance.
        :rtype: list
        """
        return self._reference_dmd.dmd_timesteps

    @property
    def original_time(self):
        """
        The original time dictionary used by this instance, which coincides
        with the original dictionary used by the reference of this instance
        (see :func:`_reference_dmd`).

        :return: The original time dictionary used by this instance.
        :rtype: pydmd.dmdbase.DMDTimeDict
        """
        return self._reference_dmd.original_time

    @property
    def original_timesteps(self):
        """
        The original timesteps in the input fed to this instance, which
        coincides with the original timesteps in the input fed to the reference
        of this instance (see :func:`_reference_dmd`).

        :return: The original timesteps in the input fed to this instance.
        :rtype: list
        """
        return self._reference_dmd.original_timesteps

    @property
    def training_parameters(self):
        """
        The original parameters passed when `self.fit` was called, represented
        as a 2D array (the index of the parameter vary along the first
        dimension).

        :type: numpy.ndarray
        """
        return self._training_parameters

    def _set_training_parameters(self, params):
        """
        Set the value of `self._original_parameters`, while checking that the
        value provided is a 2D array.

        :param numpy.ndarray: A 2D array which contains the original
            parameters.
        """
        if isinstance(params, list):
            params = np.array(params)
        if params.ndim == 1:
            params = params[:, None]
        if params.ndim > 2:
            raise ValueError("Parameters must be stored in 2D arrays.")

        self._training_parameters = params

    @property
    def parameters(self):
        """
        The new parameters to be used in `reconstructed_data`, represented
        as a 2D array (the index of the parameter vary along the first
        dimension).

        :getter: Return the current parameters.
        :setter: Change the current parameters.
        :type: numpy.ndarray
        """
        return self._parameters if hasattr(self, "_parameters") else None

    @parameters.setter
    def parameters(self, value):
        if isinstance(value, list):
            value = np.array(value)
        if value.ndim == 1:
            value = value[:, None]
        elif value.ndim > 2:
            raise ValueError("Parameters must be stored in 2D arrays.")

        self._parameters = value

    def _arrange_parametric_snapshots(self, X):
        """
        Arrange the given parametric snapshots (see :func:`fit` for an overview
        of the shape of `X`) into a 2D matrix in which the shape is distributed
        as follows:

        - 0: Space;
        - 1: Time/Parameter.

        Time varies faster than the parameter along the columns of the matrix.

        An overview of the shape of the resulting matrix:

         .. math::

            M = \\begin{bmatrix}
                    x_1(t_1,\\mu_1) & \dots & x_1(t_n,\\mu_1) & x_1(t_1,\\mu_1)
                        & \dots & x_1(t_{n-1},\\mu_k) & x_1(t_n,\\mu_k)\\\\
                    \\vdots & \\dots & \\vdots & \\vdots & \\dots & \\vdots
                        & \\dots\\\\
                    x_m(t_1,\\mu_1) & \dots & x_m(t_n,\\mu_1) & x_m(t_1,\\mu_1)
                        & \dots & x_m(t_{n-1},\\mu_k) & x_m(t_n,\\mu_k)
                \\end{bmatrix}

        :math:`x(t, \mu) \in \mathbb{R}^m` is the functon which represents the
        parametric system at time :math:`t` with the parameter :math:`\\mu`.

        :param X: Parametric snapshots.
        :type X: numpy.ndarray
        :return: The given parametric snapshots rearranged in a 2D matrix.
        :rtype: numpy.ndarray
        """
        return np.reshape(
            np.ravel(X, "C"),
            (self._space_dim, self._time_instants * self._ntrain),
            "F",
        )

    def _training_modal_coefficients(self, space_timemu):
        """
        Compute the POD modal coefficient from the given matrix, and put
        the resulting coefficients (along with their time evolution in matrix
        form) into a list.

        In symbols, from the given matrix :math:`X^x_{t,\mu} \in
        \mathbb{R}^{m \\times nk}` we compute the modal
        coefficients corresponding to its columns. At this point we have
        something like this:

        .. math::

            \\widetilde{X}^s_{t,\mu} = \\begin{bmatrix}
                    \\widetilde{x}_1(t_1,\\mu_1), & \dots &
                        \\widetilde{x}_1(t_n,\\mu_1), &
                        \\widetilde{x}_1(t_1,\\mu_1), & \dots &
                        \\widetilde{x}_1(t_{n-1},\\mu_k), &
                        \\widetilde{x}_1(t_n,\\mu_k)\\\\
                    \\vdots & \\dots & \\vdots & \\vdots & \\dots & \\vdots &
                        \\dots\\\\
                    \\widetilde{x}_p(t_1,\\mu_1), & \dots & x_p(t_n,\\mu_1) &
                        \\widetilde{x}_p(t_1,\\mu_1), & \dots &
                        \\widetilde{x}_p(t_{n-1},\\mu_k), &
                        \\widetilde{x}_p(t_n,\\mu_k)
                \\end{bmatrix} \in \mathbb{R}^{p \\times nk}

        Detecting the sub-matrices corresponding to the time evolution of the
        POD modal coefficients corresponding to a particular realization of the
        system for some parameter :math:`\\mu_i`, we may rewrite this matrix as
        follows:

        .. math::

            \\widetilde{X}^s_{t,\mu} = \\begin{bmatrix}
                    \\widetilde{X}_{\\mu_1}, & \dots & \\widetilde{X}_{\\mu_1}
            \\end{bmatrix}

        The returned list contains the matrices
        :math:`\\widetilde{X}_{\\mu_i} \in \\mathbb{p \\times n}`.

        :param space_timemu: A matrix containing parametric/time snapshots as
            returned by :func:`_arrange_parametric_snapshots`.
        :type space_timemu: numpy.ndarray
        :return: A list of matrices. Each matrix contain the time evolution of
            the POD modal coefficients corresponding to a parameter from the
            training set.
        :rtype: list
        """

        spatial_modal_coefficients = self._spatial_pod.fit(
            space_timemu
        ).reduce(space_timemu)
        return np.split(spatial_modal_coefficients, self._ntrain, axis=1)

    def _fit_dmd(self, training_modal_coefficients):
        """
        Train the DMD instance(s) on the given training modal coefficients.

        :param training_modal_coefficients: Matrix (or list of matrices) of
            modal coefficients. The time varies along columns.
        :type training_modal_coefficients: numpy.ndarray
        """

        if self.is_partitioned:
            # partitioned parametric DMD
            for dmd, data in zip(self._dmd, training_modal_coefficients):
                dmd.fit(data)

                if self._reference_dmd.dmd_time is None:
                    raise ValueError(
                        "For some reason the reference DMD has "
                        "not been fit before the others."
                    )
                dmd.dmd_time = self._reference_dmd.dmd_time
        else:
            spacemu_time = np.vstack(training_modal_coefficients)
            self._dmd.fit(spacemu_time)

    def fit(self, X, training_parameters):
        """
        Compute the parametric Dynamic Modes Decomposition from the input data
        stored in the array `X`. The shape of the parameter `X` must be
        used as follows:

        - 0: Parameters;
        - 1: Time;
        - 2: Space.

        Which means that along the first axis the parameter varies, along the
        second axis varies the time, and along the last axis varies the
        space (i.e. the dimensionality of a single snapshot of the dynamical
        system).

        The second parameter contains the list of training parameters
        corresponding to the given array of snapshots `X`. It is fundamental
        that `X.shape[0] == len(training_parameters)`, otherwise the number
        of parametric snapshots would be different than the number of training
        parameters, which obviously cannot happen.

        :param numpy.ndarray X: The input snapshots, in multiple time instants
            and parameters.
        :param numpy.ndarray training_parameters: The parameters used for the
            training, corresponding to the snapshots in `X`.
        """

        if X.shape[0] != len(training_parameters):
            raise ValueError(
                "Unexpected number of snapshots for the given"
                "parameters. Received {} parameters, and {} snapshots".format(
                    len(training_parameters), X.shape[0]
                )
            )

        # we store these values for faster access
        self._ntrain, self._time_instants, self._space_dim = X.shape
        if self.is_partitioned and self._ntrain != len(self._dmd):
            raise ValueError(
                "Invalid number of DMD instances provided: "
                "expected n_train={}, got {}".format(
                    self._ntrain, len(self._dmd)
                )
            )

        # store the training parameters: they will be used in
        # `reconstructed_data`
        self._set_training_parameters(training_parameters)

        # arrange the parametric snapshots in a convenient way to perform POD
        space_timemu = self._arrange_parametric_snapshots(X)

        # obtain POD modal coefficients from the training set
        training_modal_coefficients = self._training_modal_coefficients(
            space_timemu
        )

        # fit DMD(s) with POD modal coefficients
        self._fit_dmd(training_modal_coefficients)

    @property
    def reconstructed_data(self):
        """
        Get the reconstructed data, for the time instants specified in
        `dmd_time`, and the parameters stored in `parameters`.

        The shape of the returned data is distributed as follows:

        - 0: Parameters;
        - 1: Time;
        - 2: Space.

        :return: Snapshots predicted/interpolated using parametric Dynamic Mode
            Decomposition.
        :rtype: numpy.ndarray
        """
        forecasted_modal_coefficients = self._predict_modal_coefficients()
        interpolated_pod_modal_coefficients = (
            self._interpolate_missing_modal_coefficients(
                forecasted_modal_coefficients
            )
        )

        interpolated_pod_modal_coefficients = np.swapaxes(
            interpolated_pod_modal_coefficients, 0, 1
        )

        return np.apply_along_axis(
            self._spatial_pod.expand, 2, interpolated_pod_modal_coefficients
        )

    def save(self, fname):
        """
        Save the object to `fname` using the pickle module.

        :param str fname: the name of file where the reduced order model will
            be saved.

        Example:

        >>> from pydmd import ParametricDMD
        >>> pdmd = ParametricDMD(...) #  Construct here the rom
        >>> pdmd.fit(...)
        >>> pdmd.save('pydmd.pdmd')
        """
        with open(fname, "wb") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(fname):
        """
        Load the object from `fname` using the pickle module.

        :return: The `ReducedOrderModel` loaded

        Example:

        >>> from pydmd import ParametricDMD
        >>> pdmd = ParametricDMD.load('pydmd.pdmd')
        >>> print(pdmd.reconstructed_data)
        """
        with open(fname, "rb") as output:
            dmd = pickle.load(output)

        return dmd

    def _predict_modal_coefficients(self):
        """
        Predict future spatial modal coefficients in the time instants in
        `dmd_time`.

        :return: Predicted spatial modal coefficients.
        :rtype: numpy.ndarray
        """
        if self.is_partitioned:
            return np.vstack(
                list(map(lambda dmd: dmd.reconstructed_data, self._dmd))
            )
        return self._dmd.reconstructed_data

    def _interpolate_missing_modal_coefficients(
        self, forecasted_modal_coefficients
    ):
        """
        Interpolate spatial modal coefficients for the (untested) parameters
        stored in `parameters`. The interpolation uses the interpolator
        provided in the constructor of this instance.

        The returned value is a 3D tensor, its shape is used as follows:

        - 0: Time;
        - 1: Parameters;
        - 2: POD reduced space.

        :param numpy.ndarray forecasted_modal_coefficients: An array of spatial
            modal coefficients for tested parameters.
        :return: An array of (interpolated) spatial modal coefficients for
            untested parameters.
        :rtype: numpy.ndarray
        """
        if forecasted_modal_coefficients.shape[1] != len(self.dmd_timesteps):
            raise ValueError(
                "Invalid number of time instants provided: "
                "expected {}, got {}.".format(
                    forecasted_modal_coefficients.shape[1],
                    len(self.dmd_timesteps),
                )
            )

        approx = self._approximation

        forecasted_modal_coefficients = np.array(
            np.split(forecasted_modal_coefficients, self._ntrain, axis=0)
        )

        def interpolate_future_pod_coefficients(future_training_coefficients):
            approx.fit(
                self.training_parameters, future_training_coefficients.T
            )
            return approx.predict(self.parameters)

        return np.array(
            list(
                map(
                    interpolate_future_pod_coefficients,
                    forecasted_modal_coefficients.T,
                )
            )
        )
