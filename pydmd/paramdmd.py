import numpy as np


class ParametricDMD:
    def __init__(self, dmd, spatial_pod, approximation):
        self._dmd = dmd
        self._spatial_pod = spatial_pod
        self._approximation = approximation

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
        else:
            return self._dmd

    @property
    def dmd_time(self):
        """
        The time dictionary used by this instance, which coincides with the
        dictionary used by the reference of this instance (see
        :func:`_reference_dmd`).

        :return: The time dictionary used by this instance.
        :rtype: pydmd.dmdbase.DMDTimeDict
        """
        return self._reference_dmd.dmd_time

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
            training, corresponding to the snapshots in the first parameter.
        """

        if X.shape[0] != len(training_parameters):
            raise ValueError(
                """Unexpected number of snapshot for the given
                parameters. Received {} parameters, and {} snapshots
                """.format(
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

        self._set_training_parameters(training_parameters)

        """
        space_timemu has the following form:
         ____                                                           ____
        |x1(t0,mu0) ... x1(tn,mu0) x1(t0,mu1) ... x1(t{n-1},muk) x1(tn,muk)|
        |     .                     .                           .          |
        |     .                     .                           .          |
        |xm(t0,mu0)     xm(tn,mu0) xm(tn,mu1)     xm(t{n-1},muk) xm(tn,muk)|
         ---                                                            ---
        time varies faster than mu
        """
        space_timemu = np.reshape(
            np.ravel(X, "C"),
            (self._space_dim, self._time_instants * self._ntrain),
            "F",
        )

        spatial_modal_coefficients = self._spatial_pod.fit(
            space_timemu
        ).reduce(space_timemu)

        split_modal_coefficients = np.split(
            spatial_modal_coefficients, self._ntrain, axis=1
        )
        if self.is_partitioned:
            # partitioned parametric DMD
            for dmd, data in zip(self._dmd, split_modal_coefficients):
                dmd.fit(data)
        else:
            """
            spacemu_time has the following form:
            ____                   ____
            |a1(t0,mu0) ... a1(tn,mu0)|
            |     .                   |
            |     .                   |
            |ap(t0,mu0) ... ap(tn,mu0)|
            |a1(t0,mu1) ... a1(tn,mu1)|
            |     .                   |
            |     .                   |
            |ap(t0,mu1) ... ap(tn,mu1)|
            |     .                   |
            |     .                   |
            |a1(t0,muk) ... a1(tn,muk)|
            |     .                   |
            |     .                   |
            |ap(t0,muk) ... ap(tn,muk)|
            ----                   ----
            Time varies along columns. p is the number of POD modal
            coefficients.
            """
            spacemu_time = np.vstack(split_modal_coefficients)
            self._dmd.fit(spacemu_time)

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
        pod_modes_count = self._spatial_pod.modes.shape[1]

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
        else:
            return self._dmd.reconstructed_data

    def _interpolate_missing_modal_coefficients(
        self, forecasted_modal_coefficients
    ):
        """
        Interpolate spatial modal coefficients for the (untested) parameters
        stored in `parameters`. The interpolation uses the interpolator
        provided in the constructor of this instance.

        :param numpy.ndarray forecasted_modal_coefficients: An array of spatial
            modal coefficients for tested parameters.
        :return: An array of (interpolated) spatial modal coefficients for
            untested parameters.
        :rtype: numpy.ndarray
        """
        pod_modes_count = self._spatial_pod.modes.shape[1]
        predicted_time_instants = forecasted_modal_coefficients.shape[1]
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
