"""
Module for the parametric Dynamic Mode Decomposition.

References:
- A Dynamic Mode Decomposition Extension for the Forecasting of Parametric 
Dynamical Systems, F. Andreuzzi, N. Demo, G. Rozza, 2023, SIAM Journal on 
Applied Dynamical Systems
"""
import pickle

import numpy as np


# roll by one position the shape of X. if X.shape == (a,b,c), the returned
# NumPy array's shape is (b,c,a)
def back_roll_shape(X):
    """
    Roll by one position the shape of `X`. if `X.shape == (a,b,c)`, the returned
    NumPy array's shape is `(b,c,a)`.
    """
    return np.swapaxes(np.swapaxes(X, 0, 1), 1, 2)


def roll_shape(X):
    """
    Roll by one position the shape of `X`. if `X.shape == (a,b,c)`, the returned
    NumPy array's shape is `(c,a,b)`.
    """
    return np.swapaxes(np.swapaxes(X, 0, 2), 1, 2)


class ParametricDMD:
    """
    Implementation of the parametric Dynamic Mode Decomposition proposed in
    arXiv:2110.09155v1. Both the *monolithic* and *partitioned* methods are
    available, see the documentation of the parameter `dmd` for more details.

    :param dmd: Instance(s) of :class:`dmdbase.DMDBase`, used by the
        paramtric DMD for the prediction of future spatial modal coefficients.
        If `dmd` is a `list` the *partitioned* approach is selected, in this
        case the number of parameters in the training set should be equal to
        the number of DMD instances provided. If `dmd` is not a list, we employ
        the monolithic approach.
    :type dmd: DMDBase or list
    :param spatial_pod: Instance of an object usable for the generation of a
        ROM of the dataset (see for instance the class
        `POD <https://mathlab.github.io/EZyRB/pod.html>`_ from the Python
        library `EZyRB <https://github.com/mathLab/EZyRB>`_).
    :param approximation: An interpolator following the standard
        learning-prediction pattern (`fit()` -> `predict()`). For some
        convenient wrappers see those implemented in
        `EZyRB <https://github.com/mathLab/EZyRB>`_).
    :param bool light: Whether this instance should be light or not. A light
        instance uses less memory since it caches a smaller number of resources.
        Setting `light=True` might invalidate several properties (see also
        :meth:`training_modal_coefficients`).
    """

    def __init__(self, dmd, spatial_pod, approximation, light=False):
        self._dmd = dmd
        self._spatial_pod = spatial_pod
        self._approximation = approximation

        self._training_parameters = None
        self._parameters = None
        self._ntrain = None
        self._time_instants = None
        self._space_dim = None
        self._light = light

        self._training_modal_coefficients = None

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
        :func:`_reference_dmd`). Note that when you set this attribute the
        value is set only for the reference DMD (see :func:`_reference_dmd`),
        however when :func:`_predict_modal_coefficients` is called the values
        of all DMDs become consistent.

        :getter: Return the time dictionary used by the reference DMD instance.
        :setter: Set the given time dictionary in the field `dmd_time` for all
            DMD instances.
        :type: pydmd.dmdbase.DMDTimeDict
        """
        return self._reference_dmd.dmd_time

    @dmd_time.setter
    def dmd_time(self, value):
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
        :rtype: dict
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
        dimension). For, instance, the following feeds a set of four 3D
        parameters to `ParametricDMD`:

            >>> from pydmd import ParametricDMD
            >>> pdmd = ParametricDMD(...)
            >>> pdmd.fit(...)
            >>> p0 = [0.1, 0.2, 0.1]
            >>> p1 = [0.1, 0.2, 0.3],
            >>> p2 = [0.2, 0.2, 0.2],
            >>> p3 = [0.1, 0.2, 0.2]
            >>> pdmd.parameters = np.array([p0,p1,p2,p3])

        Therefore, when we collect the results from `reconstructed_data`:

            >>> result = pdmd.reconstructed_data
            >>> # reconstruction corresponding to p0
            >>> rec_p0 = result[0]
            >>> # reconstruction corresponding to p1
            >>> rec_p1 = result[1]
            >>> ...

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
        of the shape of `X`) into a 2D matrix such that the shape is distributed
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

        :param X: Parametric snapshots (distribition of axes like in
            :func:`fit`).
        :type X: numpy.ndarray
        :return: Parametric snapshots arranged in a 2D matrix like explained
            above.
        :rtype: numpy.ndarray
        """
        # swap parameters dimension and space dimension
        X = np.swapaxes(X, 0, 1)
        return X.reshape((X.shape[0], -1), order="C")

    def _compute_training_modal_coefficients(self, space_timemu):
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

        :param space_timemu: A matrix containing parametric/time snapshots like
            the matrix returned by :func:`_arrange_parametric_snapshots`. The
            input size should be `p x nk` where `p` is the dimensionality of
            the full-dimensional space, `k` is the number of training parameters
            and `n` is the number of time instants used for the training.
        :type space_timemu: numpy.ndarray
        :return: A list of `k` matrices. Each matrix has shape `r x n` where `r`
            is the dimensionality of the reduced POD space, and `n`, `k` are the
            same of the parameter `space_timemu`.
        :rtype: list
        """

        spatial_modal_coefficients = self._spatial_pod.fit(space_timemu).reduce(
            space_timemu
        )
        return np.split(spatial_modal_coefficients, self._ntrain, axis=1)

    def fit(self, X, training_parameters):
        """
        Compute the parametric Dynamic Modes Decomposition from the input data
        stored in the array `X`. The shape of the parameter `X` must be
        used as follows:

        - 0: Training parameters;
        - 1: Space;
        - 2: Training time instants.

        The parameter `training_parameters` contains the list of training
        parameters corresponding to the training datasets in `X`. For instance,
        `training_parameters[0]` is the parameter which generated the dataset
        in `X[0]`. For this reason `len(training_parameters)` should be equal
        to `X.shape[0]`.

        :param numpy.ndarray X: Training snapshots of the parametric system,
            observed for two or more parameters and in multiple time instants.
        :param numpy.ndarray training_parameters: Training parameters
            corresponding to the snapshots in `X`.
        """

        if X.shape[0] != len(training_parameters):
            raise ValueError(
                "Unexpected number of snapshots for the given"
                "parameters. Received {} parameters, and {} snapshots".format(
                    len(training_parameters), X.shape[0]
                )
            )

        # we store these values for faster access
        self._ntrain, self._space_dim, self._time_instants = X.shape
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
        training_modal_coefficients = self._compute_training_modal_coefficients(
            space_timemu
        )

        if not self._light:
            self._training_modal_coefficients = np.array(
                training_modal_coefficients
            )

        # fit DMD(s) with POD modal coefficients
        if self.is_partitioned:
            # partitioned parametric DMD
            for dmd, data in zip(self._dmd, training_modal_coefficients):
                dmd.fit(data)
        else:
            spacemu_time = np.vstack(training_modal_coefficients)
            self._dmd.fit(spacemu_time)

    # ------------------------------------------------------------
    # getter properties for intermediate values of the computation

    @property
    def training_modal_coefficients(self):
        """
        Modal coefficients of the input dataset. Since this is cached after
        calls to :func:`fit` this property needs to be called after :func:`fit`,
        and `light` should be set to `False` in the constructor of the class.

        The tensor returned has the following shape:

        - 0: Training parameters;
        - 1: Dimensionality of the POD sub-space;
        - 2: Time.
        """
        if self._light:
            raise RuntimeError(
                """Light instances do not cache the property
`training_modal_coefficients`."""
            )

        if self._training_modal_coefficients is None:
            raise RuntimeError(
                """
Property not available now, did you call `fit()`?"""
            )

        return self._training_modal_coefficients

    @property
    def forecasted_modal_coefficients(self):
        """
        Modal coefficients forecasted for the input parameters.

        The tensor returned has the following shape:

        - 0: Training parameters;
        - 1: Dimensionality of the POD sub-space;
        - 2: Time.
        """
        forecasted = self._predict_modal_coefficients()
        return forecasted.reshape((self._ntrain, -1, forecasted.shape[1]))

    @property
    def interpolated_modal_coefficients(self):
        """
        Modal coefficients forecasted and then interpolated for the untested
        parameters.

        The tensor returned has the following shape:

        - 0: Parameters;
        - 1: Dimensionality of the POD sub-space;
        - 2: Time.
        """
        forecasted = self._predict_modal_coefficients()
        return self._interpolate_missing_modal_coefficients(forecasted)

    # ------------------------------------------------------------

    def _predict_modal_coefficients(self):
        """
        Predict future spatial modal coefficients in the time instants in
        `dmd_time`.

        :return: Predicted spatial modal coefficients. Shape: `rk x n` (`r`:
            dimensionality of POD subspace, `k`: number of training parameters,
            `n`: number of snapshots).
        :rtype: numpy.ndarray
        """
        if self.is_partitioned:
            for dmd in self._dmd:
                # we want to "bound" this DMD objects' dmd_time
                dmd.dmd_time = self._reference_dmd.dmd_time
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

        - 0: Parameters;
        - 1: Reduced POD space;
        - 2: Time.

        :param numpy.ndarray forecasted_modal_coefficients: An array of spatial
            modal coefficients for tested parameters. The shape is used like in
            the matrix returned by :func:`_predict_modal_coefficients`.
        :return: An array of (interpolated) spatial modal coefficients for
            untested parameters.
        :rtype: numpy.ndarray
        """

        if self.parameters is None or len(self.parameters) == 0:
            raise ValueError(
                """
Unknown parameters not found. Did you set `ParametricDMD.parameters`?"""
            )

        approx = self._approximation
        forecasted_modal_coefficients = forecasted_modal_coefficients.reshape(
            (self._ntrain, -1, forecasted_modal_coefficients.shape[1]),
            order="C",
        )

        def interpolate_future_pod_coefficients(time_slice):
            approx.fit(self.training_parameters, time_slice)
            return approx.predict(self.parameters)

        return np.dstack(
            [
                interpolate_future_pod_coefficients(time_slice)[..., None]
                for time_slice in roll_shape(forecasted_modal_coefficients)
            ]
        )

    @property
    def reconstructed_data(self):
        """
        Get the reconstructed data, for the time instants specified in
        `dmd_time`, and the parameters stored in `parameters`.

        The shape of the returned data is distributed as follows:

        - 0: Parameters;
        - 1: Space;
        - 2: Time.

        :return: Snapshots predicted/interpolated using parametric DMD and the
            given method for ROM.
        :rtype: numpy.ndarray
        """
        forecasted_modal_coefficients = self._predict_modal_coefficients()
        interpolated_modal_coefficients = (
            self._interpolate_missing_modal_coefficients(
                forecasted_modal_coefficients
            )
        )

        return np.apply_along_axis(
            self._spatial_pod.expand, 1, interpolated_modal_coefficients
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
            return pickle.load(output)
