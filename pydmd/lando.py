"""
Derived module from dmdbase.py for linear and
nonlinear disambiguation optimization (LANDO).

References:
- 
"""
import warnings
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

from .dmdbase import DMDBase
from .dmdoperator import DMDOperator
from .utils import compute_svd, compute_rank
from .snapshots import Snapshots

SUPPORTED_KERNELS = ["poly", "rbf", "linear"]

class LANDOOperator(DMDOperator):
    """
    LANDO operator.
    """
    def __init__(
        self,
        svd_rank,
        kernel_function,
        sparsity_thres,
        permute,
    ):
        self._svd_rank = svd_rank
        self._kernel_function = kernel_function
        self._sparsity_thres = sparsity_thres
        self._permute = permute

        self._sparse_dictionary = None
        self._weights = None

        self._modes = None
        self._eigenvalues = None
        self._eigenvectors = None
        self._Atilde = None
        self._A = None

    def compute_operator(self, X, Y):
        self._sparse_dictionary = self._learn_sparse_dictionary(X)
        self._weights = Y.dot(
            np.linalg.pinv(self._kernel_function(self._sparse_dictionary, X))
        )

    # def fixed_point_analysis(self, x_fixed, compute_A):
    #     self._modes = 
    #     self._eigenvalues = 
    #     self._eigenvectors = 
    #     self._Atilde = 
    #     self._A = 

    @property
    def sparse_dictionary(self):
        return self._sparse_dictionary

    @property
    def weights(self):
        return self._weights

    @property
    def eigenvalues(self):
        if self._eigenvalues is None:
            raise RuntimeError("You need to call fit and analyze_fixed_point.")
        return self._eigenvalues

    @property
    def eigenvectors(self):
        if self._eigenvectors is None:
            raise RuntimeError("You need to call fit and analyze_fixed_point.")
        return self._eigenvectors

    @property
    def modes(self):
        if self._modes is None:
            raise RuntimeError("You need to call fit and analyze_fixed_point.")
        return self._modes

    @property
    def as_numpy_array(self):
        if self._Atilde is None:
            raise RuntimeError("You need to call fit and analyze_fixed_point.")
        return self._Atilde

    def _learn_sparse_dictionary(self, X):
        """
        Sparse dictionary learning with Cholesky updates.

        :param X: matrix containing the right-hand side snapshots by column.
        :type X: numpy.ndarray
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

            if np.abs(delta_t) > self._sparsity_thres:
                dict_inds.append(ind_t)

                # Update the Cholesky factor
                m = len(cholesky_factor)
                c_t = max(0, np.sqrt(k_tt - np.sum(s_t ** 2)))
                cholesky_factor = np.hstack(
                    [cholesky_factor, np.zeros((m, 1))]
                )
                cholesky_factor = np.vstack(
                    [cholesky_factor, np.append(s_t.conj().T, c_t)]
                )

                if k_tt < np.sum(s_t ** 2):
                    msg = (
                        "The Cholesky factor is ill-conditioned. "
                        "Consider increasing the sparsity parameter "
                        "or changing the kernel hyperparameters."
                    )
                    warnings.warn(msg)

        return X[:, dict_inds]




class LANDO(DMDBase):
    """
    Linear and nonlinear disambiguation optimization (LANDO).

    :param kernel_metric: the kernel function to apply. Supported kernel
        metrics are `"poly"`, `"rbf"`, and `"linear"`.
    :type kernel_metric: str
    :param kernel_params: additional parameters for the
        `sklearn.metrics.pairwise_kernels` function, including
        kernel-specific function parameters.
    :type kernel_params: dict
    :param sparsity_thres: threshold at which delta_t, the degree to which a
        snapshot x_t can be represented by the snapshot dictionary in feature
        space, is considered high enough that x_t should be added to the
        snapshot dictionary. Referred to as \nu in the original paper.
    :type sparsity_thres: float
    :param permute: whether or not to randomly permute the order of the
        snapshots prior to learning the snapshot dictionary. Default is True,
        to permute the snapshots, as doing so is generally recommended in order
        to avoid generating ill-conditioned dictionaries.
    :type permute: bool
    """

    def __init__(
        self,
        svd_rank=0,
        kernel_metric="poly",
        kernel_params=None,
        sparsity_thres=0.1,
        permute=True,
    ):
        kernel_params = self._test_kernel_inputs(kernel_metric, kernel_params)
        self._kernel_metric = kernel_metric
        self._kernel_params = kernel_params
        self._kernel_function = lambda X, Y: pairwise_kernels(
            X.T, Y.T, metric=self._kernel_metric, **self._kernel_params
        )

        self._Atilde = LANDOOperator(
            svd_rank,
            self._kernel_function,
            sparsity_thres,
            permute,
        )

    def fit(self, X, Y):
        """
        Compute the Optimized Dynamic Mode Decomposition.
        """
        self.operator.compute_operator(X, Y)

        return self

    def f(self, x):
        if self.operator.weights is None:
            raise RuntimeError("You need to call fit.")
        return self.operator.weights.dot(
            self._kernel_function(self.operator.sparse_dictionary, x[:, None])
        )

    def analyze_fixed_point(self, x_fixed, compute_A=False):
        c = self.f(x_fixed)
        self.operator.fixed_point_analysis(x_fixed, compute_A)

        return c, L, N


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

        # For simplicity, only gamma = 1 is supported for the poly kernel.
        if kernel_metric == "poly":
            if "gamma" in kernel_params.keys():
                msg = (
                    "Only gamma = 1 functionality is supported for the "
                    "polynomial kernel. Resetting gamma to 1..."
                )
                warnings.warn(msg)
            kernel_params["gamma"] = 1

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

        return kernel_params
