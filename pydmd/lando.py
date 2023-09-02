"""
Derived module from dmdbase.py for linear and
nonlinear disambiguation optimization (LANDO).

References:
- Baddoo Peter J., Herrmann Benjamin, McKeon Beverley J. and Brunton Steven L.
2022 Kernel learning for robust dynamic mode decomposition: linear and
nonlinear disambiguation optimization. Proc. R. Soc. A. 478: 20210830.
20210830. http://doi.org/10.1098/rspa.2021.0830
"""
import warnings
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.metrics.pairwise import pairwise_kernels

from .dmdbase import DMDBase
from .dmdoperator import DMDOperator
from .snapshots import Snapshots
from .utils import compute_svd, compute_tlsq

SUPPORTED_KERNELS = ["linear", "poly", "rbf"]


class LANDOOperator(DMDOperator):
    """
    LANDO operator class, which is used to compute and keep track of the
    dictionary-based kernel model and the diagnostics of the linear model
    about a fixed point.

    :param svd_rank: the rank for the truncation of the linear operator; If 0,
        the method computes the optimal rank and uses it for truncation; if a
        positive integer, the method uses the argument for the truncation; if
        float between 0 and 1, the rank is the number of the biggest singular
        values that are needed to reach the 'energy' specified by `svd_rank`;
        if -1, the method does not compute a truncation.
    :type svd_rank: int or float
    :param kernel_metric: the kernel function to apply. Supported kernel
        metrics for `LANDO` are `"linear"`, `"poly"`, and `"rbf"`.
    :type kernel_metric: str
    :param kernel_params: additional parameters to be inputted to the
        `sklearn.metrics.pairwise_kernels` function. This includes
        kernel-specific function parameters.
    :type kernel_params: dict
    """

    def __init__(
        self,
        svd_rank,
        kernel_metric,
        kernel_params,
    ):
        super().__init__(
            svd_rank=svd_rank,
            exact=True,
            forward_backward=False,
            rescale_mode=None,
            sorted_eigs=False,
            tikhonov_regularization=None,
        )
        self._kernel_metric = kernel_metric
        self._kernel_params = kernel_params
        self._weights = None
        self._eigenvalues = None
        self._eigenvectors = None
        self._modes = None
        self._Atilde = None
        self._A = None

    def kernel_function(self, X, Y):
        """
        Calls `sklearn.metrics.pairwise_kernels` to evaluate the kernel
        function at the given feature matrices X and Y.

        :param X: feature matrix with shape (n_features, n_samples_X)
        :type X: numpy.ndarray
        :param Y: second feature matrix with shape (n_features, n_samples_Y)
        :type Y: numpy.ndarray
        :return: kernel matrix with shape (n_samples_X, n_samples_Y)
        :rtype: numpy.ndarray
        """
        return pairwise_kernels(
            X.T, Y.T, metric=self._kernel_metric, **self._kernel_params
        )

    def kernel_gradient(self, X, y):
        """
        Computes the gradient of the kernel with respect to the second feature
        vector x. Then evaluates the gradient at the feature matrix X and the
        feature vector x = y. Currently, this method is only compatible with
        the linear, polynomial, and RBF kernels.

        :param X: feature matrix with shape (n_features, n_samples_X)
        :type X: numpy.ndarray
        :param y: feature vector with shape (n_features,)
        :type y: numpy.ndarray
        :return: kernel gradient matrix with shape (n_samples_X, n_features)
        :rtype: numpy.ndarray
        """
        if self._kernel_metric == "linear":
            return X.T

        # Kernel metric must be polynomial or RBF.
        if "gamma" in self._kernel_params.keys():
            gamma = self._kernel_params["gamma"]
        else:  # set the pairwise_kernels gamma default
            gamma = 1.0 / X.shape[0]

        if self._kernel_metric == "poly":
            coef0 = self._kernel_params["coef0"]
            degree = self._kernel_params["degree"]
            return np.diag(
                gamma * degree * (coef0 + gamma * X.T.dot(y)) ** (degree - 1)
            ).dot(X.T)

        # Kernel metric is RBF.
        centered_X = X - y[:, None]
        return np.diag(
            -2
            * gamma
            * np.exp(-gamma * np.linalg.norm(centered_X, 2, axis=0) ** 2)
        ).dot(centered_X.T)

    @property
    def weights(self):
        """
        Get the dictionary-based kernel model weights.
        Referred to as W_tilde in the original reference.

        :return: the kernel model weights.
        :rtype: numpy.ndarray
        """
        if self._weights is None:
            raise ValueError("You need to call fit.")
        return self._weights

    @property
    def eigenvalues(self):
        """
        Get the eigenvalues of the full linear operator evaluated about a fixed
        point. This operator is referred to as L in the original reference.

        :return: the eigenvalues of the full linear operator.
        :rtype: numpy.ndarray
        """
        if self._eigenvalues is None:
            raise ValueError("You need to call fit and analyze_fixed_point.")
        return self._eigenvalues

    @property
    def eigenvectors(self):
        """
        Get the eigenvectors of the reduced linear operator evaluated about a
        fixed point. This operator is referred to as L_hat in the original
        reference.

        :return: the eigenvectors of the reduced linear operator.
        :rtype: numpy.ndarray
        """
        if self._eigenvectors is None:
            raise ValueError("You need to call fit and analyze_fixed_point.")
        return self._eigenvectors

    @property
    def modes(self):
        """
        Get the eigenvectors of the full linear operator evaluated about a
        fixed point. This operator is referred to as L in the original
        reference.

        :return: the eigenvectors of the full linear operator.
        :rtype: numpy.ndarray
        """
        if self._modes is None:
            raise ValueError("You need to call fit and analyze_fixed_point.")
        return self._modes

    @property
    def as_numpy_array(self):
        """
        Get the reduced linear operator A_tilde.
        Referred to as L_hat in the original reference.

        :return: the reduced linear operator A_tilde.
        :rtype: numpy.ndarray
        """
        if self._Atilde is None:
            raise ValueError("You need to call fit and analyze_fixed_point.")
        return self._Atilde

    @property
    def A(self):
        """
        Get the full linear operator A.
        Referred to as L in the original reference.

        :return: the full linear operator A.
        :rtype: numpy.ndarray
        """
        if self._A is None:
            msg = (
                "Full linear operator currently not computed. You may need to "
                "call fit and analyze_fixed_point. If already done so, call "
                "analyze_fixed_point again using compute_A=True."
            )
            warnings.warn(msg)
        return self._A

    def compute_operator(self, X, Y, X_dict):
        """
        Compute the dictionary-based kernel model weights.

        :param X: the input snapshots x.
        :type X: numpy.ndarray
        :param Y: input snapshots y such that F(x) = y.
        :type Y: numpy.ndarray
        :param X_dict: the sparse feature dictionary.
        :type X_dict: numpy.ndarray
        """
        self._weights = Y.dot(np.linalg.pinv(self.kernel_function(X_dict, X)))

    def compute_linear_operator(
        self, fixed_point, compute_A, X_dict, x_rescale
    ):
        """
        Compute the DMD diagnostics of the linear model about a fixed point.

        :param fixed_point: base state about which to perturb the system.
        :type fixed_point: numpy.ndarray
        :param compute_A: if True, the full linear operator is explicitly
            computed and stored. If False, the full linear operator isn't
            computed and stored explicitly.
        :type compute_A: bool
        :param X_dict: the sparse feature dictionary.
        :type X_dict: numpy.ndarray
        """
        kernel_grad = self.kernel_gradient(X_dict, fixed_point)
        kernel_grad = np.multiply(kernel_grad, x_rescale)
        U, s, V = compute_svd(kernel_grad.T, self._svd_rank)
        if compute_A:
            self._A = self.weights.dot(kernel_grad)
            self._Atilde = np.linalg.multi_dot([U.conj().T, self._A, U])
        else:
            self._Atilde = np.linalg.multi_dot(
                [U.conj().T, self.weights, kernel_grad, U]
            )

        # Filter out zero eigenvalues.
        eigenvalues, eigenvectors = np.linalg.eig(self._Atilde)
        self._eigenvalues = eigenvalues[np.abs(eigenvalues) > 1e-16]
        self._eigenvectors = eigenvectors[:, np.abs(eigenvalues) > 1e-16]
        self._modes = np.linalg.multi_dot(
            [
                self._weights,
                V,
                np.diag(s),
                self._eigenvectors,
                np.diag(1 / self._eigenvalues),
            ]
        )


class LANDO(DMDBase):
    """
    Linear and nonlinear disambiguation optimization (LANDO).

    :param svd_rank: the rank for the truncation of the linear operator; If 0,
        the method computes the optimal rank and uses it for truncation; if a
        positive integer, the method uses the argument for the truncation; if
        float between 0 and 1, the rank is the number of the biggest singular
        values that are needed to reach the 'energy' specified by `svd_rank`;
        if -1, the method does not compute a truncation.
    :type svd_rank: int or float
    :param tlsq_rank: rank truncation computing Total Least Square. Default is
        0, which means no truncation.
    :type tlsq_rank: int
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
    :param kernel_metric: the kernel function to apply. Supported kernel
        metrics for `LANDO` are `"linear"`, `"poly"`, and `"rbf"`.
    :type kernel_metric: str
    :param kernel_params: additional parameters to be inputted to the
        `sklearn.metrics.pairwise_kernels` function. This includes
        kernel-specific function parameters.
    :type kernel_params: dict
    :param x_rescale: value or (n_features,) array of values for rescaling the
        the features of the input data. Can be used to improve conditioning.
    :type x_rescale: float or numpy.ndarray
    :param dict_tol: threshold at which delta_t, the degree to which a snapshot
        x_t can be represented by the snapshot dictionary in feature space, is
        considered high enough that x_t should be added to the snapshot
        dictionary. Referred to as \nu in the original paper.
    :type dict_tol: float
    :param permute: whether or not to randomly permute the order of the
        snapshots prior to learning the snapshot dictionary. Default is True,
        to permute the snapshots, as doing so is generally recommended in order
        to avoid generating ill-conditioned dictionaries.
    :type permute: bool
    """

    def __init__(
        self,
        svd_rank=0,
        tlsq_rank=0,
        opt=False,
        kernel_metric="linear",
        kernel_params=None,
        x_rescale=1.0,
        dict_tol=1e-6,
        permute=True,
    ):
        self._test_kernel_inputs(kernel_metric, kernel_params)

        if kernel_params is None:
            kernel_params = {}
        # set the pairwise_kernels polynomial degree default
        if kernel_metric == "poly" and "degree" not in kernel_params.keys():
            kernel_params["degree"] = 3
        # set the pairwise_kernels polynomial coef0 default
        if kernel_metric == "poly" and "coef0" not in kernel_params.keys():
            kernel_params["coef0"] = 1

        super().__init__(
            svd_rank=svd_rank,
            tlsq_rank=tlsq_rank,
            exact=True,
            opt=opt,
        )

        self._Atilde = LANDOOperator(
            svd_rank=svd_rank,
            kernel_metric=kernel_metric,
            kernel_params=kernel_params,
        )

        # Keep track of the computed sparse dictionary.
        self._sparse_dictionary = None
        self._x_rescale = x_rescale
        self._dict_tol = dict_tol
        self._permute = permute

        # Keep track of the last fixed point analysis.
        self._fixed_point = None
        self._bias = None

    @property
    def partially_fitted(self):
        """
        Check whether the weights for this LANDO instance have been computed.
        Note that a LANDO model is only fully fitted once the weights have been
        computed, AND an analysis about a fixed point has been performed.

        :return: `True` if instance has been partially fitted, else `False`.
        :rtype: bool
        """
        try:
            return self.operator.weights is not None
        except ValueError:
            return False

    @property
    def sparse_dictionary(self):
        """
        Get the learned sparse feature dictionary.

        :return: the sparse feature dictionary.
        :rtype: numpy.ndarray
        """
        if not self.partially_fitted:
            warnings.warn("You need to call fit.")
        return self._sparse_dictionary

    @property
    def fixed_point(self):
        """
        Get the fixed point from the last fixed point analysis.

        :return: the fixed point from the last fixed point analysis.
        :rtype: numpy.ndarray
        """
        if not self.fitted:
            warnings.warn("You need to call fit and analyze_fixed_point.")
        return self._fixed_point

    @property
    def bias(self):
        """
        Get the bias term from the last fixed point analysis. This is in
        reference to the term c in the expression f(x) = c + Ax' + N(x') for
        our system perturbed about a base state x_bar where x = x_bar + x'.
        f is our model approximation, A is a linear operator, and N is a
        nonlinear operator.

        :return: the bias term from the last fixed point analysis.
        :rtype: numpy.ndarray
        """
        if not self.fitted:
            warnings.warn("You need to call fit and analyze_fixed_point.")
        return self._bias

    def f(self, x):
        """
        Prediction of the true system model F, where F(x) = y.

        :param x: feature vector from the data domain.
        :type x: numpy.ndarray
        :return: f(x), an approximation of the true quantity F(x).
        :rtype: numpy.ndarray
        """
        if not self.partially_fitted:
            raise RuntimeError("You need to call fit.")
        x = self._check_input_shape(x)
        x = self._rescale(x)
        return self.operator.weights.dot(
            self.operator.kernel_function(self._sparse_dictionary, x[:, None])
        )

    def nonlinear(self, x):
        """
        Get the nonlinear term from the last fixed point analysis, evaluated
        at the input vector x. This is in reference to the term N(x') in the
        expression f(x) = c + Ax' + N(x') for our system perturbed about a
        base state x_bar where x = x_bar + x'. f is our model approximation,
        c is a constant bias term, and A is a linear operator.

        :param x: feature vector from the data domain.
        :type x: numpy.ndarray
        :return: the nonlinear term from the last fixed point analysis,
            evaluated at the input vector x.
        :rtype: numpy.ndarray
        """
        # TODO: I may be missing a rescaling somewhere here. Check later.
        if not self.fitted:
            raise RuntimeError("You need to call fit and analyze_fixed_point.")
        x = self._check_input_shape(x)
        kernel_grad = self.operator.kernel_gradient(
            self._sparse_dictionary, self._rescale(self._fixed_point)
        )
        kernel_grad = np.multiply(kernel_grad, self._x_rescale)
        return (
            self.f(self._fixed_point + x)
            - self._bias
            - np.linalg.multi_dot(
                [
                    self.operator.weights,
                    kernel_grad,
                    x,
                ]
            )
        )

    def fit(self, X, Y=None):
        """
        Compute the linear and nonlinear disambiguation optimization (LANDO).

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        :param Y: additional input snapshots such that F(x) = y. If not given,
            snapshots from X are used to build a discrete-time model.
        :type Y: numpy.ndarray or iterable
        """
        self._reset()
        self._snapshots_holder = Snapshots(X)
        self._check_x_rescale()

        if Y is None:
            X = self.snapshots[:, :-1]
            Y = self.snapshots[:, 1:]
        else:
            self._compare_data_shapes(Snapshots(Y).snapshots)
            self._snapshots_holder_y = Snapshots(Y)
            X = self.snapshots
            Y = self.snapshots_y

        X, Y = compute_tlsq(X, Y, self._tlsq_rank)
        X_rescaled = self._rescale(X)
        self._sparse_dictionary = self._learn_sparse_dictionary(X_rescaled)
        self.operator.compute_operator(X_rescaled, Y, self._sparse_dictionary)

        # Default timesteps
        self._set_initial_time_dictionary(
            {"t0": 0, "tend": self.snapshots.shape[1] - 1, "dt": 1}
        )

        return self

    def analyze_fixed_point(self, fixed_point, compute_A=False):
        """
        Use a partially-fitted LANDO model to examine the system perturbed
        about a given base state x_bar such that x = x_bar + x'. Extract the
        bias term, linear portion, and nonlinear portion from the expression
        f(x) = c + Ax' + N(x'), where f is our model approximation, c is a
        constant bias term, A is a linear operator, and N is a nonlinear
        operator. Additionally compute and obtain the DMD diagnostics of the
        linear model A, along with the DMD amplitudes associated with the
        computed linear model.
        """
        if not self.partially_fitted:
            msg = (
                "You need to call fit before analyzing "
                "the system about a fixed point."
            )
            raise RuntimeError(msg)

        fixed_point = self._check_input_shape(fixed_point)

        self._fixed_point = fixed_point
        self._bias = self.f(fixed_point)
        self.operator.compute_linear_operator(
            self._rescale(fixed_point),
            compute_A,
            self._sparse_dictionary,
            self._x_rescale,
        )

        self._b = self._compute_amplitudes()

    def predict(self, x0, tend, continuous=True, dt=1.0, solve_ivp_opts=None):
        """
        Reconstruct or predict the state of the system using the fitted model.

        :param x0: initial condition from which to propagate.
        :type x0: numpy.ndarray
        :param tend: number of data points to compute.
        :type tend: int
        :param continuous: if True, the method assumes that the fitted model
            f(x) is continuous, in which case f(x) predicts the time derivative
            x'. If False, the method assumes that the fitted model is discrete,
            in which case f(x_n) predicts the the future data point x_{n+1}.
        :type continuous: bool
        :param dt: desired time step between each computed data point. This
            parameter is only used if `continuous=True`.
        :type dt: float
        :param solve_ivp_opts: dictionary of additional parameters to be passed
            to the `scipy.integrate.solve_ivp` function. This parameter is only
            used if `continuous=True`.
        :type solve_ivp_opts: dict
        """
        if not self.partially_fitted:
            raise RuntimeError("You need to call fit.")

        x0 = self._check_input_shape(x0)

        if continuous:

            def ode_sys(xt, x):
                return self.f(x)

            if solve_ivp_opts is None:
                solve_ivp_opts = {}

            t_eval = np.arange(0, tend * dt, dt)

            sol = solve_ivp(
                ode_sys,
                [t_eval[0], t_eval[-1]],
                x0,
                t_eval=t_eval,
                **solve_ivp_opts
            )

            return sol.y

        # Otherwise, assume a discrete-time system.
        Y = np.empty((len(x0), tend))
        Y[:, 0] = x0
        for i in range(tend - 1):
            Y[:, i + 1] = self.f(Y[:, i])
        return Y

    def _learn_sparse_dictionary(self, X):
        """
        Sparse dictionary learning with Cholesky updates.

        :param X: matrix containing the right-hand side snapshots by column.
        :type X: numpy.ndarray
        :return: matrix containing the snapshots from X that make up the
            sparse snapshot dictionary.
        :rtype: numpy.ndarray
        """
        # Determine the order in which to parse the data snapshots.
        parsing_inds = np.arange(X.shape[1])
        if self._permute:
            np.random.shuffle(parsing_inds)

        # Initialize the Cholesky factorization routine.
        ind_0 = parsing_inds[0]
        x_0 = X[:, ind_0, None]
        cholesky_factor = np.sqrt(self.operator.kernel_function(x_0, x_0))
        dict_inds = [ind_0]

        for ind_t in parsing_inds[1:]:
            # Equation (3.11): Evaluate the kernel using the current dictionary
            # items and the next candidate addition to the dictionary.
            x_t = X[:, ind_t, None]
            k_tilde_next = self.operator.kernel_function(X[:, dict_inds], x_t)

            # Equation (3.10): Use backsubstitution to compute the span of the
            # current dictionary.
            s_t = np.linalg.lstsq(cholesky_factor, k_tilde_next, rcond=None)[0]
            pi_t = np.linalg.lstsq(cholesky_factor.conj().T, s_t, rcond=None)[0]

            # Equation (3.9): Compute the minimum (squared) distance between
            # the current sample and the span of the current dictionary.
            k_tt = self.operator.kernel_function(x_t, x_t)
            delta_t = k_tt - k_tilde_next.conj().T.dot(pi_t)

            if np.abs(delta_t) > self._dict_tol:
                # Update the dictionary.
                dict_inds.append(ind_t)

                # Update the Cholesky factor.
                cholesky_factor = np.hstack(
                    [cholesky_factor, np.zeros((len(cholesky_factor), 1))]
                )
                cholesky_factor = np.vstack(
                    [
                        cholesky_factor,
                        np.append(
                            s_t.conj().T,
                            max(0, np.abs(np.sqrt(k_tt - np.sum(s_t**2)))),
                        ),
                    ]
                )

                if k_tt < np.sum(s_t**2):
                    msg = (
                        "The Cholesky factor is ill-conditioned. "
                        "Consider increasing the sparsity parameter "
                        "or changing the kernel hyperparameters."
                    )
                    warnings.warn(msg)

        return X[:, dict_inds]

    def _rescale(self, X):
        if isinstance(self._x_rescale, np.ndarray):
            if np.ndim(X) == 1:
                return np.multiply(X, self._x_rescale)
            return np.multiply(X, self._x_rescale[:, None])
        return X * self._x_rescale

    def _check_x_rescale(self):
        if not isinstance(self._x_rescale, (int, float, np.ndarray)):
            raise TypeError("x_rescale must be a float or a numpy array.")
        if (
            isinstance(self._x_rescale, np.ndarray)
            and self._x_rescale.shape != self.snapshots_shape
        ):
            msg = (
                "If a numpy array, x_rescale must have the "
                "same shape {} as the input features X."
            )
            raise RuntimeError(msg.format(self.snapshots_shape))
        if isinstance(self._x_rescale, np.ndarray):
            self._x_rescale = self._x_rescale.flatten()

    def _check_input_shape(self, x):
        """
        Helper function that ensures that the given x is a numpy array with the
        same shape as the input snapshots. Returns the flattened version of x.
        """
        if not isinstance(x, np.ndarray) or x.shape != self.snapshots_shape:
            msg = "Input x vector must be a {} numpy.ndarray."
            raise RuntimeError(msg.format(self.snapshots_shape))
        return x.flatten()

    @staticmethod
    def _test_kernel_inputs(kernel_metric, kernel_params):
        """
        Helper function that checks the validity of the provided kernel metric
        and parameters. Additionally uses a dummy array of data in order to
        call `sklearn.metrics.pairwise_kernels` using the given parameters.
        """
        if kernel_metric not in SUPPORTED_KERNELS:
            msg = (
                "Invalid kernel metric '{}' provided. "
                "Please use one of the following metrics: {}"
            )
            raise ValueError(msg.format(kernel_metric, SUPPORTED_KERNELS))

        if kernel_params is None:
            kernel_params = {}
        elif not isinstance(kernel_params, dict):
            raise TypeError("kernel_params must be a dict.")

        # Test that sklearn.metrics.pairwise_kernels can be called.
        x_dummy = np.arange(4).reshape(2, 2)
        try:
            pairwise_kernels(
                x_dummy, x_dummy, metric=kernel_metric, **kernel_params
            )
        except Exception as e:
            msg = (
                "Invalid kernel parameters {} provided. "
                "Please verify that kernel_params are passable to "
                "sklearn.metrics.pairwise_kernels with metric '{}'."
            )
            raise ValueError(msg.format(kernel_params, kernel_metric)) from e
