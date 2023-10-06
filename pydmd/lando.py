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
from inspect import isfunction

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
    at a fixed point. See LANDO documentation for parameter descriptions.
    """

    def __init__(self, svd_rank, dict_tol, online, permute, lstsq):
        super().__init__(
            svd_rank=svd_rank,
            exact=True,
            forward_backward=False,
            rescale_mode=None,
            sorted_eigs=False,
            tikhonov_regularization=None,
        )

        # Keep track of the sparse dictionary.
        self._sparse_dictionary = None
        self._dict_tol = dict_tol
        self._online = online
        self._permute = permute
        self._lstsq = lstsq

        # Keep track of attributes for online learning.
        self._cholesky = None
        self._P = None

        # Keep track of operator attributes.
        self._weights = None
        self._eigenvalues = None
        self._eigenvectors = None
        self._modes = None
        self._Atilde = None
        self._A = None

    @property
    def sparse_dictionary(self):
        """
        Get the learned sparse feature dictionary.

        :return: the sparse feature dictionary.
        :rtype: numpy.ndarray
        """
        if self._sparse_dictionary is None:
            raise ValueError("You need to call fit().")
        return self._sparse_dictionary

    @property
    def weights(self):
        """
        Get the dictionary-based kernel model weights.
        Referred to as W_tilde in the original reference.

        :return: the kernel model weights.
        :rtype: numpy.ndarray
        """
        if self._weights is None:
            raise ValueError("You need to call fit().")
        return self._weights

    @property
    def eigenvalues(self):
        """
        Get the eigenvalues of the full linear operator evaluated at a fixed
        point. This operator is referred to as L in the original reference.

        :return: the eigenvalues of the full linear operator.
        :rtype: numpy.ndarray
        """
        if self._eigenvalues is None:
            msg = "You need to call fit() and analyze_fixed_point()."
            raise ValueError(msg)
        return self._eigenvalues

    @property
    def eigenvectors(self):
        """
        Get the eigenvectors of the reduced linear operator evaluated at a
        fixed point. The reduced operator is referred to as L_hat in the
        original reference.

        :return: the eigenvectors of the reduced linear operator.
        :rtype: numpy.ndarray
        """
        if self._eigenvectors is None:
            msg = "You need to call fit() and analyze_fixed_point()."
            raise ValueError(msg)
        return self._eigenvectors

    @property
    def modes(self):
        """
        Get the eigenvectors of the full linear operator evaluated at a
        fixed point. The full operator is referred to as L in the original
        reference.

        :return: the eigenvectors of the full linear operator.
        :rtype: numpy.ndarray
        """
        if self._modes is None:
            msg = "You need to call fit() and analyze_fixed_point()."
            raise ValueError(msg)
        return self._modes

    @property
    def as_numpy_array(self):
        """
        Get the reduced linear operator A_tilde.
        Referred to as L_hat in the original reference.

        :return: the reduced linear operator.
        :rtype: numpy.ndarray
        """
        if self._Atilde is None:
            msg = "You need to call fit() and analyze_fixed_point()."
            raise ValueError(msg)
        return self._Atilde

    @property
    def A(self):
        """
        Get the full linear operator A.
        Referred to as L in the original reference.

        :return: the full linear operator.
        :rtype: numpy.ndarray
        """
        if self._A is None:
            msg = (
                "Full linear operator currently not computed. You may need to "
                "call fit() and analyze_fixed_point(). If already done so, "
                "call analyze_fixed_point() using the flag compute_A=True."
            )
            warnings.warn(msg)
        return self._A

    def compute_operator(self, X, Y, kernel_function):
        """
        Compute the dictionary-based kernel model weights.

        :param X: the input snapshots x, rescaled.
        :type X: numpy.ndarray
        :param Y: input snapshots y such that F(x) = y.
        :type Y: numpy.ndarray
        :param kernel_function: kernel function to apply.
        :type kernel_function: function
        """
        # Determine the order in which to parse the data snapshots.
        parsing_inds = np.arange(X.shape[1])
        if self._permute:
            np.random.shuffle(parsing_inds)

        # Initialize the Cholesky factorization routine.
        ind_0 = parsing_inds[0]
        x_0 = np.expand_dims(X[:, ind_0], axis=-1)
        y_0 = np.expand_dims(Y[:, ind_0], axis=-1)
        C = np.sqrt(kernel_function(x_0, x_0))
        self._sparse_dictionary = np.copy(x_0)

        # Initialize the online learning routine, if applicable.
        if self._online:
            self._cholesky = C
            self._P = np.ones((1, 1))
            self._weights = y_0 / (self._cholesky[0][0] ** 2)

        for ind_t in parsing_inds[1:]:
            # Grab the next corresponding pair of snapshots.
            x_t = np.expand_dims(X[:, ind_t], axis=-1)
            y_t = np.expand_dims(Y[:, ind_t], axis=-1)

            # Get the results of this Cholesky factorization iteration.
            results = self._cholesky_step(x_t, kernel_function, C)
            _, s_t, _, k_tt, delta_t = results

            # NOT almost linearly dependent - add x to the dictionary.
            if np.abs(delta_t) > self._dict_tol:
                self._sparse_dictionary = np.hstack(
                    [self._sparse_dictionary, x_t]
                )

                # Update the Cholesky factor.
                C = np.vstack(
                    [
                        np.hstack([C, np.zeros((len(C), 1))]),
                        np.append(
                            s_t.conj().T,
                            max(0, np.abs(np.sqrt(k_tt - np.sum(s_t**2)))),
                        ),
                    ]
                )

                # Perform the online learning updates, if applicable.
                if self._online:
                    self._cholesky = C
                    self._update_online(y_t, results, cholesky_updated=True)

                if k_tt < np.sum(s_t**2):
                    msg = (
                        "The Cholesky factor is ill-conditioned. Consider "
                        "increasing dict_tol or changing the kernel function."
                    )
                    warnings.warn(msg)

            # Online learning updates for the almost linearly dependent case.
            elif self._online:
                self._update_online(y_t, results, cholesky_updated=False)

        if not self._online:
            K_mat = kernel_function(self._sparse_dictionary, X)
            if self._lstsq:  # use least squares
                self._weights = np.linalg.lstsq(K_mat.T, Y.T, rcond=None)[0].T
            else:  # use the pseudo-inverse
                self._weights = Y.dot(np.linalg.pinv(K_mat))

    def update_operator(self, X, Y, kernel_function):
        """
        Update the dictionary-based kernel model weights given more snapshots.

        :param X: the input snapshots x, rescaled.
        :type X: numpy.ndarray
        :param Y: input snapshots y such that F(x) = y.
        :type Y: numpy.ndarray
        :param kernel_function: kernel function to apply.
        :type kernel_function: function
        """
        if not self._online:
            msg = "Only LANDO models fitted with online=True can be updated."
            raise ValueError(msg)

        # Determine the order in which to parse the data snapshots.
        parsing_inds = np.arange(X.shape[1])
        if self._permute:
            np.random.shuffle(parsing_inds)

        for ind_t in parsing_inds:
            # Grab the next corresponding pair of snapshots.
            x_t = np.expand_dims(X[:, ind_t], axis=-1)
            y_t = np.expand_dims(Y[:, ind_t], axis=-1)

            # Get the results of this Cholesky factorization iteration.
            results = self._cholesky_step(x_t, kernel_function, self._cholesky)
            _, s_t, _, k_tt, delta_t = results

            # NOT almost linearly dependent - update dict and cholesky factor.
            if np.abs(delta_t) > self._dict_tol:
                self._sparse_dictionary = np.hstack(
                    [self._sparse_dictionary, x_t]
                )
                self._cholesky = np.vstack(
                    [
                        np.hstack(
                            [self._cholesky, np.zeros((len(self._cholesky), 1))]
                        ),
                        np.append(
                            s_t.conj().T,
                            max(0, np.abs(np.sqrt(k_tt - np.sum(s_t**2)))),
                        ),
                    ]
                )
                self._update_online(y_t, results, cholesky_updated=True)
                if k_tt < np.sum(s_t**2):
                    msg = (
                        "The Cholesky factor is ill-conditioned. Consider "
                        "increasing dict_tol or changing the kernel function."
                    )
                    warnings.warn(msg)
            else:
                self._update_online(y_t, results, cholesky_updated=False)

    def compute_linear_operator(
        self, fixed_point, compute_A, kernel_gradient, x_rescale
    ):
        """
        Compute the DMD diagnostics of the linear model at a fixed point.

        :param fixed_point: base state about which to perturb the system.
        :type fixed_point: numpy.ndarray
        :param compute_A: if True, the full linear operator is explicitly
            computed and stored. If False, the full linear operator isn't
            computed and stored explicitly.
        :type compute_A: bool
        :param kernel_gradient: gradient of the kernel function applied.
        :type kernel_gradient: function
        :param x_rescale: value or (n_features,) array of values for
            rescaling the features of the input data.
        :type x_rescale: float or numpy.ndarray
        """
        kernel_grad = kernel_gradient(self._sparse_dictionary, fixed_point)
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
        nonzero_inds = np.abs(eigenvalues) > 1e-16
        eigenvalues = eigenvalues[nonzero_inds]
        eigenvectors = eigenvectors[:, nonzero_inds]

        # Sort eigenvalues, descending based on modulus.
        sorted_inds = np.argsort(-np.abs(eigenvalues))
        self._eigenvalues = eigenvalues[sorted_inds]
        self._eigenvectors = eigenvectors[:, sorted_inds]

        self._modes = np.linalg.multi_dot(
            [
                self._weights,
                V,
                np.diag(s),
                self._eigenvectors,
                np.diag(1 / self._eigenvalues),
            ]
        )

    def _cholesky_step(self, x, kernel_function, cholesky):
        """
        Helper function that computes and returns all of the quantities used to
        decide the progression of the Cholesky factorization routine. See the
        original reference for specific equations and notations.
        """
        # Equation (3.11): Evaluate the kernel using the current dictionary
        # items and the next candidate addition to the dictionary.
        k_tilde = kernel_function(self._sparse_dictionary, x)

        # Equation (3.10): Use backsubstitution to compute the span of the
        # current dictionary.
        s = np.linalg.lstsq(cholesky, k_tilde, rcond=None)[0]
        pi = np.linalg.lstsq(cholesky.conj().T, s, rcond=None)[0]

        # Equation (3.9): Compute the minimum (squared) distance between
        # the current sample and the span of the current dictionary.
        kxx = kernel_function(x, x)[0][0]
        delta = kxx - k_tilde.conj().T.dot(pi)

        return k_tilde, s, pi, kxx, delta

    def _update_online(self, y, cholesky_results, cholesky_updated):
        """
        Helper function that performs a single online learning update for
        both the almost linearly dependent case and the not almost linearly
        dependent case. Updates the P matrix and the weight matrix. See the
        supplemental information of the original reference for more details.
        """
        k_tilde, _, pi, _, delta = cholesky_results

        # NOT almost linearly dependent case.
        if cholesky_updated:
            self._P = np.vstack(
                [
                    np.hstack([self._P, np.zeros((len(self._P), 1))]),
                    np.append(np.zeros((1, len(self._P))), 1.0),
                ]
            )
            weights_update = (y - self._weights.dot(k_tilde)) / delta
            self._weights = np.hstack(
                [
                    self._weights - weights_update.dot(pi.conj().T),
                    weights_update,
                ]
            )
        # Almost linearly dependent case.
        else:
            h = pi.conj().T.dot(self._P) / (
                1.0 + np.linalg.multi_dot([pi.conj().T, self._P, pi])
            )
            self._P = self._P.dot(np.eye(len(self._P)) - pi.dot(h))
            weights_update = np.linalg.lstsq(
                (self._cholesky.dot(self._cholesky.conj().T)).T,
                ((y - self._weights.dot(k_tilde)).dot(h)).T,
                rcond=None,
            )[0].T
            self._weights += weights_update


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
    :param kernel_function: custom kernel function to be used instead of the
        `sklearn.metrics.pairwise_kernels` built-ins. Must be able to take a
        (n_features, n_samples_X) numpy.ndarray and a (n_features, n_samples_Y)
        numpy.ndarray and return the (n_samples_X, n_samples_Y) kernel matrix
        as a numpy.ndarray. In other words, for the input matrices X, Y,
            kernel_function(X, Y) = K(X, Y),
        where K(X, Y) denotes the kernel matrix.
    :type kernel_function: function
    :param kernel_gradient: gradient of the kernel function evaluated at the
        given (n_features, n_samples_X) numpy.ndarray, and at the given
        (n_features,) numpy.ndarray. Must return a (n_samples_X, n_features)
        numpy.ndarray. This function must be defined in order to perform
        `analyze_fixed_point()` if a custom kernel function is given.
        In other words, for the input matrix X and input vector y,
            kernel_gradient(X, y) = \nabla K(X, y)
        for the kernel function defined via the kernel_function input.
    :type kernel_gradient: function
    :param x_rescale: value or `snapshots_shape` numpy array of values for
        rescaling the features of the input data. Can be used to improve
        conditioning.
    :type x_rescale: float or numpy.ndarray
    :param dict_tol: threshold at which delta_t, the degree to which a snapshot
        x_t can be represented by the snapshot dictionary in feature space, is
        considered high enough that x_t should be added to the snapshot
        dictionary. Referred to as \nu in the original paper. Note that
        increasing this tolerance will lead to sparser dictionaries.
    :type dict_tol: float
    :param online: whether or not to use the online learning variant of LANDO.
        Default is False, to not use the online learning algorithm.
    :type online: bool
    :param permute: whether or not to randomly permute the order of the
        snapshots prior to learning the snapshot dictionary. Default is True,
        to permute the snapshots, as doing so is generally recommended in order
        to avoid generating ill-conditioned dictionaries.
    :type permute: bool
    :param lstsq: method used for computing the weights of the dictionary-based
        kernel model. If True, least-squares is used to solve for the weights,
        otherwise the pseudo-inverse is used. Note that this parameter is
        ignored if online learning is enabled.
    :type lstsq: bool
    """

    def __init__(
        self,
        svd_rank=0,
        tlsq_rank=0,
        opt=False,
        kernel_metric="linear",
        kernel_params=None,
        kernel_function=None,
        kernel_gradient=None,
        x_rescale=1.0,
        dict_tol=1e-6,
        online=False,
        permute=True,
        lstsq=True,
    ):
        # Check the validity of the provided kernel functions.
        if kernel_params is None:
            kernel_params = {}
        self._test_kernel_inputs(kernel_metric, kernel_params)
        self._test_kernel_functions(kernel_function, kernel_gradient)

        # Initialize the basic DMD attributes.
        super().__init__(
            svd_rank=svd_rank,
            tlsq_rank=tlsq_rank,
            exact=True,
            opt=opt,
        )

        # Set the kernel functions, which must be valid.
        self._kernel_metric = kernel_metric
        self._kernel_params = kernel_params
        self._kernel_function = kernel_function
        self._kernel_gradient = kernel_gradient

        # Build the LANDO operator.
        self._Atilde = LANDOOperator(
            svd_rank=svd_rank,
            dict_tol=dict_tol,
            online=online,
            permute=permute,
            lstsq=lstsq,
        )

        # Keep track of the data rescaling factor.
        self._x_rescale = x_rescale

        # Keep track of the last fixed point analysis.
        self._compute_A = False
        self._fixed_point = None
        self._bias = None

        # Keep track of whether or not the model was ever updated.
        self._updated = False

    def kernel_function(self, X, Y):
        """
        Calls `sklearn.metrics.pairwise_kernels` to evaluate the kernel
        function at the given feature matrices X and Y. If a custom kernel
        function was provided, then that function will be used instead.

        :param X: feature matrix with shape (n_features, n_samples_X)
        :type X: numpy.ndarray
        :param Y: second feature matrix with shape (n_features, n_samples_Y)
        :type Y: numpy.ndarray
        :return: kernel matrix with shape (n_samples_X, n_samples_Y)
        :rtype: numpy.ndarray
        """
        if self._kernel_function is not None:
            return self._kernel_function(X, Y)
        return pairwise_kernels(
            X.T, Y.T, metric=self._kernel_metric, **self._kernel_params
        )

    def kernel_gradient(self, X, y):
        """
        Computes the gradient of the kernel with respect to the second feature
        vector x. Then evaluates the gradient at the feature matrix X and the
        feature vector y. Currently, this method is only compatible with
        the linear, polynomial, and RBF kernels, unless a custom kernel
        gradient function was provided.

        :param X: feature matrix with shape (n_features, n_samples_X)
        :type X: numpy.ndarray
        :param y: feature vector with shape (n_features,)
        :type y: numpy.ndarray
        :return: kernel gradient matrix with shape (n_samples_X, n_features)
        :rtype: numpy.ndarray
        """
        if self._kernel_function is not None:
            if self._kernel_gradient is None:
                msg = (
                    "Unable to evaluate the gradient of the kernel function. "
                    "If a custom kernel function was provided, please ensure "
                    "that the corresponding gradient function is also defined "
                    "by setting the kernel_gradient parameter the LANDO model."
                )
                raise ValueError(msg)
            return self._kernel_gradient(X, y)

        if self._kernel_metric == "linear":
            return X.T

        # Kernel metric must be polynomial or RBF.
        if "gamma" in self._kernel_params.keys():
            gamma = self._kernel_params["gamma"]
        else:  # set the pairwise_kernels gamma default
            gamma = 1.0 / X.shape[0]

        if self._kernel_metric == "poly":
            if "coef0" in self._kernel_params.keys():
                coef0 = self._kernel_params["coef0"]
            else:
                coef0 = 1
            if "degree" in self._kernel_params.keys():
                degree = self._kernel_params["degree"]
            else:
                degree = 3
            return np.diag(
                gamma * degree * (coef0 + gamma * X.T.dot(y)) ** (degree - 1)
            ).dot(X.T)

        # Kernel metric is RBF.
        centered_X = X - np.expand_dims(y, axis=-1)
        return np.diag(
            -2
            * gamma
            * np.exp(-gamma * np.linalg.norm(centered_X, 2, axis=0) ** 2)
        ).dot(centered_X.T)

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
        # Note: the attribute operator.sparse_dictionary is re-scaled.
        # Thus we undo this, but only when the user asks for the dictionary.
        if not self.partially_fitted:
            raise ValueError("You need to call fit().")
        if isinstance(self._x_rescale, np.ndarray):
            return np.divide(
                self.operator.sparse_dictionary,
                np.expand_dims(self._x_rescale, axis=-1),
            )
        return (1 / self._x_rescale) * self.operator.sparse_dictionary

    def f(self, X):
        """
        Prediction of the true system model F, where F(x) = y.

        :param X: feature vectors from the data domain.
        :type X: numpy.ndarray or iterable
        :return: f(X), an approximation of the true quantity F(X) = Y.
        :rtype: numpy.ndarray
        """
        if not self.partially_fitted:
            raise ValueError("You need to call fit().")

        # Ensure that the input is an array whose columns contain snapshots
        # with a dimension that matches that of the original input data.
        X = np.array(X)
        if X.shape[:-1] != self.snapshots_shape:
            msg = "Last dimension of X must contain data with shape {}."
            raise ValueError(msg.format(self.snapshots_shape))

        # Flatten and rescale the snapshots before applying the model.
        X = X.reshape(-1, X.shape[-1])
        X = self._rescale(X)

        return self.operator.weights.dot(
            self.kernel_function(self.operator.sparse_dictionary, X)
        )

    @property
    def fixed_point(self):
        """
        Get the fixed point from the last fixed point analysis.

        :return: the fixed point from the last fixed point analysis.
        :rtype: numpy.ndarray
        """
        if not self.fitted:
            msg = "You need to call fit() and analyze_fixed_point()."
            raise ValueError(msg)
        return self._fixed_point

    @property
    def bias(self):
        """
        Get the bias term from the last fixed point analysis. This is in
        reference to the term c in the expression
            f(x) = c + Ax' + N(x'), x = x_bar + x'
        for our system perturbed about x_bar. f is our model approximation,
        A is a linear operator, and N is a nonlinear operator.

        :return: the bias term from the last fixed point analysis.
        :rtype: numpy.ndarray
        """
        if not self.fitted:
            msg = "You need to call fit() and analyze_fixed_point()."
            raise ValueError(msg)
        return self._bias

    @property
    def linear(self):
        """
        Get the linear operator from the last fixed point analysis. This is in
        reference to the term A in the expression
            f(x) = c + Ax' + N(x'), x = x_bar + x'
        for our system perturbed about x_bar. f is our model approximation,
        c is a constant bias term, and N is a nonlinear operator.

        :return: the linear operator from the last fixed point analysis.
        :rtype: numpy.ndarray
        """
        if not self.fitted:
            msg = "You need to call fit() and analyze_fixed_point()."
            raise ValueError(msg)
        return self.operator.A

    def nonlinear(self, X):
        """
        Get the nonlinear term from the last fixed point analysis, evaluated
        at the input snapshots X. This is in reference to the term N(x') in
        the expression
            f(x) = c + Ax' + N(x'), x = x_bar + x'
        for our system perturbed about x_bar. f is our model approximation,
        c is a constant bias term, and A is a linear operator.

        :param X: feature vectors from the data domain.
        :type X: numpy.ndarray or iterable
        :return: the nonlinear terms from the last fixed point analysis,
            evaluated at the input vectors.
        :rtype: numpy.ndarray
        """
        if not self.fitted:
            msg = "You need to call fit() and analyze_fixed_point()."
            raise ValueError(msg)

        # Ensure that the input is an array whose columns contain snapshots
        # with a dimension that matches that of the original input data.
        X = np.array(X)
        if X.shape[:-1] != self.snapshots_shape:
            msg = "Last dimension of X must contain data with shape {}."
            raise ValueError(msg.format(self.snapshots_shape))

        # Re-create the kernel gradient matrix for the linear operator.
        kernel_grad = np.multiply(
            self.kernel_gradient(
                self.operator.sparse_dictionary,
                self._rescale(self._fixed_point.flatten()),
            ),
            self._x_rescale,
        )

        # Define the f(x) term.
        fX = self.f(X + np.expand_dims(self._fixed_point, axis=-1))

        return (
            fX
            - self._bias
            - np.linalg.multi_dot(
                [
                    self.operator.weights,
                    kernel_grad,
                    X.reshape(-1, X.shape[-1]),
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
        self.operator.compute_operator(X_rescaled, Y, self.kernel_function)

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
            f(x) = c + Ax' + N(x'),
        where f is our model approximation, c is a constant bias term, A is a
        linear operator, and N is a nonlinear operator. Additionally obtain the
        DMD diagnostics of the linear model A, along with the DMD amplitudes
        associated with the computed linear model.

        :param fixed_point: base state about which to analyze the system.
        :type fixed_point: numpy.ndarray or iterable
        :param compute_A: whether or not to explicitly compute the full linear
            operator A. If True, A is computed and stored explicitly, otherwise
            A is not stored and is only computed when needed.
        :type compute_A: bool
        """
        if not self.partially_fitted:
            msg = (
                "You need to call fit() before "
                "performing a fixed point analysis."
            )
            raise ValueError(msg)

        fixed_point = np.array(fixed_point)
        if fixed_point.shape != self.snapshots_shape:
            msg = "Input fixed point must have shape {}."
            raise ValueError(msg.format(self.snapshots_shape))

        self._compute_A = compute_A
        self._fixed_point = fixed_point
        self._bias = self.f(np.expand_dims(fixed_point, axis=-1))
        self.operator.compute_linear_operator(
            self._rescale(fixed_point.flatten()),
            self._compute_A,
            self.kernel_gradient,
            self._x_rescale,
        )

        self._b = self._compute_amplitudes()

    def update(self, X, Y=None):
        """
        Update a fitted LANDO model using new data.

        Note that this function only updates the LANDO model, and not the
        stored snapshots or DMDTimeDicts. Hence be aware that using this
        function may result in strange or unexpected behavior when used
        in conjunction with certain DMDBase functionalities.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        :param Y: additional input snapshots such that F(x) = y. If not given,
            snapshots from X are used to build a discrete-time model.
        :type Y: numpy.ndarray or iterable
        """
        if not self.partially_fitted:
            msg = "You need to call fit() before updating a LANDO model."
            raise ValueError(msg)

        X_newsnap = Snapshots(X).snapshots
        if Y is None:
            X = X_newsnap[:, :-1]
            Y = X_newsnap[:, 1:]
        else:
            Y_newsnap = Snapshots(Y).snapshots
            if X_newsnap.shape != Y_newsnap.shape:
                msg = "X {} and Y {} input data must be the same shape."
                raise ValueError(msg.format(X_newsnap.shape, Y_newsnap.shape))
            X = X_newsnap
            Y = Y_newsnap

        X, Y = compute_tlsq(X, Y, self._tlsq_rank)
        X_rescaled = self._rescale(X)
        self.operator.update_operator(X_rescaled, Y, self.kernel_function)

        # If a fixed point analysis was already done, redo it.
        if self.fitted:
            self._bias = self.f(np.expand_dims(self._fixed_point, axis=-1))
            self.operator.compute_linear_operator(
                self._rescale(self._fixed_point.flatten()),
                self._compute_A,
                self.kernel_gradient,
                self._x_rescale,
            )
            self._b = self._compute_amplitudes()

        # Flag this model as updated.
        self._updated = True

        return self

    def predict(self, x0, tend, continuous=True, dt=1.0, solve_ivp_opts=None):
        """
        Reconstruct or predict the state of the system using the fitted model.

        :param x0: initial condition from which to propagate.
        :type x0: numpy.ndarray or iterable
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
            raise ValueError("You need to call fit().")

        x0 = np.array(x0)
        if x0.shape != self.snapshots_shape:
            msg = "Input initial condition must have shape {}."
            raise ValueError(msg.format(self.snapshots_shape))
        x0 = x0.flatten()

        if continuous:

            def ode_sys(xt, x):
                return self.f(np.expand_dims(x, axis=-1)).flatten()

            if solve_ivp_opts is None:
                solve_ivp_opts = {}

            t_eval = np.arange(0, tend * dt, dt)

            sol = solve_ivp(
                ode_sys,
                [t_eval[0], t_eval[-1]],
                x0,
                t_eval=t_eval,
                **solve_ivp_opts,
            )

            return sol.y

        # Otherwise, assume a discrete-time system.
        Y = np.empty((len(x0), tend))
        Y[:, 0] = x0
        for i in range(tend - 1):
            Y[:, i + 1] = self.f(np.expand_dims(Y[:, i], axis=-1)).flatten()

        return Y

    def _rescale(self, X):
        """
        Helper function that rescales the given data according to
        the current `LANDO` instance's `x_rescale` value(s).
        """
        if isinstance(self._x_rescale, np.ndarray):
            if np.ndim(X) == 1:
                return np.multiply(X, self._x_rescale)
            return np.multiply(X, np.expand_dims(self._x_rescale, axis=-1))
        return X * self._x_rescale

    def _check_x_rescale(self):
        """
        Helper function that ensures that `x_rescale` is either an int, float,
        or a numpy array. Also ensures that if `x_rescale` is an array, then it
        must possess the same shape as the flattened input snapshots.
        """
        if not isinstance(self._x_rescale, (int, float, np.ndarray)):
            raise TypeError("x_rescale must be a float or a numpy array.")
        if isinstance(self._x_rescale, np.ndarray):
            if self._x_rescale.shape != self.snapshots_shape:
                msg = (
                    "If a numpy array, x_rescale must have the "
                    "same shape {} as the input features X."
                )
                raise ValueError(msg.format(self.snapshots_shape))
            self._x_rescale = self._x_rescale.flatten()

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

        if not isinstance(kernel_params, dict):
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

    @staticmethod
    def _test_kernel_functions(kernel_function, kernel_gradient):
        """
        Helper function that checks the validity of the provided kernel
        functions. Ensures that either both are None, or both inputs are valid.
        In short...
        - If provided, kernel_function and kernel_gradient must be functions
            that accept and return numpy arrays of the expected shape.
        - kernel_gradient function cannot be provided without kernel_function.
        - kernel_function can be provided without a kernel_gradient function,
            however a fixed point analysis cannot be performed as a result.
        """
        # Test moves on if both inputs are None.
        if kernel_function is not None or kernel_gradient is not None:
            # kernel gradient can't be provided alone.
            if kernel_function is None:
                msg = (
                    "kernel_gradient cannot be provided "
                    "without a corresponding kernel_function."
                )
                raise ValueError(msg)
            # kernel_function needs to be a function.
            if not isfunction(kernel_function):
                raise TypeError("kernel_function must be a function.")
            # Warn the user if kernel_gradient wasn't provided.
            if kernel_gradient is None:
                msg = (
                    "kernel_function given without kernel_gradient. Note that "
                    "fit() may be performed, but kernel_gradient must now be "
                    "provided in order to perform analyze_fixed_point()."
                )
                warnings.warn(msg)
            # kernel_gradient needs to be a function.
            elif not isfunction(kernel_gradient):
                raise TypeError("kernel_gradient must be a function")

            # Test that the given functions take and yield what is expected.
            X_dummy = np.empty((5, 10))
            Y_dummy = np.empty((5, 20))
            general_msg = (
                "Please check the LANDO class documentation "
                "for details on how to format the functions "
                "kernel_function and kernel_gradient."
            )
            # Test the kernel function.
            try:
                K_xy = kernel_function(X_dummy, Y_dummy)
            except Exception as e:
                msg = "Error calling kernel_function. "
                raise ValueError(msg + general_msg) from e
            if not isinstance(K_xy, np.ndarray) or K_xy.shape != (10, 20):
                msg = "kernel_function must return a {} numpy array. "
                raise ValueError((msg + general_msg).format((10, 20)))

            # Test the kernel gradient function.
            if kernel_gradient is not None:
                try:
                    grad = kernel_gradient(X_dummy, Y_dummy[:, 0])
                except Exception as e:
                    msg = "Error calling kernel_gradient. "
                    raise ValueError(msg + general_msg) from e
                if not isinstance(grad, np.ndarray) or grad.shape != (10, 5):
                    msg = "kernel_gradient must return a {} numpy array. "
                    raise ValueError((msg + general_msg).format((10, 5)))
