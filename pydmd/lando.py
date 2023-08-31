"""
Derived module from dmdbase.py for linear and
nonlinear disambiguation optimization (LANDO).

References:
- 
"""
import warnings
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.metrics.pairwise import pairwise_kernels

from .dmdbase import DMDBase
from .dmdoperator import DMDOperator
from .utils import compute_svd, compute_tlsq
from .snapshots import Snapshots

SUPPORTED_KERNELS = ["poly", "rbf", "linear"]
DEFAULT_DEGREE = 3
DEFAULT_COEF0 = 1.0
DEFAULT_GAMMA = 1.0


class LANDOOperator(DMDOperator):
    """
    LANDO operator.
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
        return pairwise_kernels(
            X.T, Y.T, metric=self._kernel_metric, **self._kernel_params
        )

    def kernel_gradient(self, X, y):
        if self._kernel_metric == "poly":
            gamma = self._kernel_params["gamma"]
            coef0 = self._kernel_params["coef0"]
            degree = self._kernel_params["degree"]
            return np.diag(
                gamma * degree * (coef0 + gamma * X.T.dot(y)) ** (degree - 1)
            ).dot(X.T)

        if self._kernel_metric == "rbf":
            gamma = self._kernel_params["gamma"]
            centered_X = X - y[:, None]
            return np.diag(
                -2
                * gamma
                * np.exp(-gamma * np.linalg.norm(centered_X, 2, axis=0) ** 2)
            ).dot(centered_X.T)

        return X.T

    @property
    def weights(self):
        if self._weights is None:
            raise ValueError("You need to call fit.")
        return self._weights

    @property
    def eigenvalues(self):
        if self._eigenvalues is None:
            raise ValueError("You need to call fit and analyze_fixed_point.")
        return self._eigenvalues

    @property
    def eigenvectors(self):
        if self._eigenvectors is None:
            raise ValueError("You need to call fit and analyze_fixed_point.")
        return self._eigenvectors

    @property
    def modes(self):
        if self._modes is None:
            raise ValueError("You need to call fit and analyze_fixed_point.")
        return self._modes

    @property
    def as_numpy_array(self):
        if self._Atilde is None:
            raise ValueError("You need to call fit and analyze_fixed_point.")
        return self._Atilde

    @property
    def A(self):
        if self._A is None:
            msg = (
                "Full linear operator currently not computed. You may need to "
                "call fit and analyze_fixed_point. If already done so, call "
                "analyze_fixed_point again using compute_A=True."
            )
            warnings.warn(msg)
        return self._A

    def compute_operator(self, X, Y, X_dict):
        self._weights = Y.dot(np.linalg.pinv(self.kernel_function(X_dict, X)))

    def compute_linear_operator(self, fixed_point, compute_A, X_dict):
        kernel_grad = self.kernel_gradient(X_dict, fixed_point)
        U, s, V = compute_svd(kernel_grad.T, self._svd_rank)
        if compute_A:
            self._A = self.weights.dot(kernel_grad)
            self._Atilde = np.linalg.multi_dot([U.conj().T, self._A, U])
        else:
            self._Atilde = np.linalg.multi_dot(
                [U.conj().T, self.weights.dot(kernel_grad), U]
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
        metrics are `"poly"`, `"rbf"`, and `"linear"`.
    :type kernel_metric: str
    :param kernel_params: additional parameters for the
        `sklearn.metrics.pairwise_kernels` function, including
        kernel-specific function parameters.
    :type kernel_params: dict
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
        dict_tol=0.1,
        permute=True,
    ):
        self._test_kernel_inputs(kernel_metric, kernel_params)

        # Set the default kernel parameterizations.
        if kernel_params is None:
            kernel_params = {}
        if kernel_metric != "linear" and "gamma" not in kernel_params.keys():
            kernel_params["gamma"] = DEFAULT_GAMMA
        if kernel_metric == "poly" and "degree" not in kernel_params.keys():
            kernel_params["degree"] = DEFAULT_DEGREE
        if kernel_metric == "poly" and "coef0" not in kernel_params.keys():
            kernel_params["coef0"] = DEFAULT_COEF0

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
        self._dict_tol = dict_tol
        self._permute = permute

        # Keep track of the results of the last fixed point analysis.
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
        if not self.partially_fitted:
            warnings.warn("You need to call fit.")
        return self._sparse_dictionary

    @property
    def fixed_point(self):
        if not self.fitted:
            warnings.warn("You need to call fit and analyze_fixed_point.")
        return self._fixed_point

    @property
    def bias(self):
        if not self.fitted:
            warnings.warn("You need to call fit and analyze_fixed_point.")
        return self._bias

    def f(self, x):
        """
        Prediction of the true system model F, where F(x) = y.

        :param x:
        :type x: numpy.ndarray
        :return:
        :rtype: numpy.ndarray
        """
        if not self.partially_fitted:
            raise RuntimeError("You need to call fit.")
        x = self._check_input_shape(x)
        return self.operator.weights.dot(
            self.operator.kernel_function(self._sparse_dictionary, x[:, None])
        )

    def nonlinear(self, x):
        if not self.fitted:
            raise RuntimeError("You need to call fit and analyze_fixed_point.")
        x = self._check_input_shape(x)
        return (
            self.f(self._fixed_point + x)
            - self._bias
            - np.linalg.multi_dot(
                [
                    self.operator.weights,
                    self.operator.kernel_gradient(
                        self._sparse_dictionary, self._fixed_point
                    ),
                    x,
                ]
            )
        )

    def fit(self, X, Y=None):
        """
        Compute the linear and nonlinear disambiguation optimization (LANDO).

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        :param Y: additional input snapshots such that F(x) = y.
            If not provided, snapshots from X are used.
        :type Y: numpy.ndarray or iterable
        """
        self._reset()
        self._snapshots_holder = Snapshots(X)
        n_samples = self.snapshots.shape[1]

        if Y is None:
            X = self.snapshots[:, :-1]
            Y = self.snapshots[:, 1:]
        else:
            self._compare_data_shapes(Snapshots(Y).snapshots)
            self._snapshots_holder_y = Snapshots(Y)
            X = self.snapshots
            Y = self.snapshots_y

        X, Y = compute_tlsq(X, Y, self._tlsq_rank)
        self._sparse_dictionary = self._learn_sparse_dictionary(X)
        self.operator.compute_operator(X, Y, self._sparse_dictionary)

        # Default timesteps
        self._set_initial_time_dictionary(
            {"t0": 0, "tend": n_samples - 1, "dt": 1}
        )

        return self

    def analyze_fixed_point(self, fixed_point, compute_A=False):
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
            fixed_point, compute_A, self._sparse_dictionary
        )
        self._b = self._compute_amplitudes()

    def predict(self, x0, tend, continuous=True, dt=1, solve_ivp_opts=None):
        if not self.partially_fitted:
            raise RuntimeError("You need to call fit.")

        x0 = self._check_input_shape(x0)

        if continuous:
            def ode_sys(xt, x):
                return self.f(x)

            if solve_ivp_opts is None:
                # Set integrator keywords to replicate odeint defaults.
                solve_ivp_opts = {}
                solve_ivp_opts["rtol"] = 1e-12
                solve_ivp_opts["method"] = "LSODA"
                solve_ivp_opts["atol"] = 1e-12

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

    # TODO: add a matrix scaling element to this
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
        x_0 = X[:, ind_0]
        k_00 = self._kernel_function(x_0[:, None], x_0[:, None])
        cholesky_factor = np.sqrt(k_00)
        dict_inds = [ind_0]

        for ind_t in parsing_inds[1:]:
            # Equation (3.11): Evaluate the kernel using the current dictionary
            # items and the next candidate addition to the dictionary.
            x_t = X[:, ind_t]
            k_tilde_next = self._kernel_function(X[:, dict_inds], x_t[:, None])

            # Equation (3.10): Use backsubstitution to compute the span of the
            # current dictionary.
            s_t = np.linalg.lstsq(cholesky_factor, k_tilde_next, rcond=None)[0]
            pi_t = np.linalg.lstsq(cholesky_factor.conj().T, s_t, rcond=None)[0]

            # Equation (3.9): Compute the minimum (squared) distance between
            # the current sample and the span of the current dictionary.
            k_tt = self._kernel_function(x_t[:, None], x_t[:, None])
            delta_t = k_tt - k_tilde_next.conj().T.dot(pi_t)

            if np.abs(delta_t) > self._dict_tol:
                dict_inds.append(ind_t)

                # Update the Cholesky factor
                m = len(cholesky_factor)
                c_t = max(0, np.sqrt(k_tt - np.sum(s_t**2)))
                cholesky_factor = np.hstack([cholesky_factor, np.zeros((m, 1))])
                cholesky_factor = np.vstack(
                    [cholesky_factor, np.append(s_t.conj().T, c_t)]
                )

                if k_tt < np.sum(s_t**2):
                    msg = (
                        "The Cholesky factor is ill-conditioned. "
                        "Consider increasing the sparsity parameter "
                        "or changing the kernel hyperparameters."
                    )
                    warnings.warn(msg)

        return X[:, dict_inds]

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
