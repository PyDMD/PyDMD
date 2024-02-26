"""
Derived module from dmdbase.py for kernelized extended dmd.

References:
- M. O. Williams, C. W. Rowley, and I. G. Kevrekidis,
A kernel-based method for data-driven koopman spectral analysis,
J. Comput. Dynam., 2 (2015), pp. 247-265
- M. O. Williams, I. G. Kevrekidis, and C. W. Rowley,
A data-driven approximation of the koopman operator: extending
dynamic mode decomposition, J. Nonlinear Sci., 25 (2015), pp. 1307-1346
"""

import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

from .dmd import DMD
from .dmdoperator import DMDOperator


class EDMDOperator(DMDOperator):
    """
    DMD operator for kernelized extended DMD.

    :param svd_rank: the rank for the truncation; If positive integer, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation.
    :type svd_rank: int or float
    :param kernel_metric: the kernel function to apply. Supported kernel
        metrics include `"additive_chi2"`, `"chi2"`, `"linear"`, `"poly"`,
        `"polynomial"`, `"rbf"`, `"laplacian"`, `"sigmoid"`, and `"cosine"`.
    :type kernel_metric: str
    :param kernel_params: additional parameters for the
        `sklearn.metrics.pairwise_kernels` function, including
        kernel-specific function parameters.
    :type kernel_params: dict
    """

    def __init__(self, svd_rank, kernel_metric, kernel_params):
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
        self._eigenvalues = None
        self._Lambda = None
        self._modes = None
        self._svd_vals = None
        self._eigenvectors = None

    @property
    def svd_vals(self):
        """
        :return: the singular values of the extended feature matrix.
        :rtype: numpy.ndarray
        """
        return self._svd_vals

    @property
    def as_numpy_array(self):
        """
        EDMD does not explicitly compute the forward operator or its reduction.
        Doing so poses risks of prohibitively-large computations...
        """
        raise ValueError("Atilde is not computed explicitly during EDMD.")

    def compute_operator(self, X, Y):
        """
        Compute and store the kernelized EDMD diagnostics.

        :param X: matrix containing the right-hand side snapshots by column.
        :type X: numpy.ndarray
        :param Y: matrix containing the left-hand side snapshots by column.
        :type Y: numpy.ndarray
        :return: the (truncated) left-singular vectors matrix of the extended
            feature matrix, the (truncated) singular values matrix of the
            extended feature matrix, and the eigenvectors of the tranformed
            forward operator `K_hat`.
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
        """
        # Compute kernel matrices using given snapshots and kernel function.
        G_hat = pairwise_kernels(
            X.T, X.T, metric=self._kernel_metric, **self._kernel_params
        )
        A_hat = pairwise_kernels(
            Y.T, X.T, metric=self._kernel_metric, **self._kernel_params
        )

        # Use the Gramian to obtain the singular vectors Q and singular
        # values Sigma of the feature embedding matrix.
        Q, Sigma = self._compute_feature_matrix_svd(G_hat)

        # Construct the tranformed forward operator matrix K_hat.
        K_hat = np.linalg.multi_dot(
            [np.linalg.pinv(Sigma), Q.T, A_hat, Q, np.linalg.pinv(Sigma)]
        )

        # Obtain the eigenvalues and eigenvectors of K_hat.
        # Note: The nonzero eigenvalues of K_hat are the DMD eigenvalues.
        # DMD modes are derived via the left and right eigenvectors of K_hat.
        eigenvalues, V_hat = np.linalg.eig(K_hat)
        modes = np.linalg.multi_dot(
            [X, Q, np.linalg.pinv(Sigma), np.linalg.inv(V_hat).T]
        )

        # Discard eigenvalue, mode pairs that correspond with zero eigenvalues.
        nonzero_inds = np.abs(eigenvalues) > 1e-16
        eigenvalues = eigenvalues[nonzero_inds]
        modes = modes[:, nonzero_inds]

        # Set the DMD eigenvalues and modes.
        self._eigenvalues = eigenvalues
        self._Lambda = eigenvalues
        self._modes = modes

        # Store Sigma and V_hat for eigenfunction computations.
        self._svd_vals = np.diag(Sigma)
        self._eigenvectors = V_hat

        return Q, Sigma, V_hat

    def _compute_feature_matrix_svd(self, Gramian):
        """
        Helper function that computes the eigendecomposition of the Gramian in
        order to obtain the singular vectors Q and singular values Sigma of the
        corresponding feature embedding matrix.

        :param numpy.ndarray Gramian: the Gramian matrix.
        :return: the (truncated) left-singular vectors matrix of the extended
            feature matrix and the (truncated) singular values matrix of the
            extended feature matrix.
        :rtype: numpy.ndarray, numpy.ndarray
        """
        G_eigenvalues, G_eigenvectors = np.linalg.eig(Gramian)

        # Note: The eigenvalues of the Gramian are the singular values of the
        # feature embedding matrix, squared. Hence in theory, the eigenvalues
        # should be real and non-negative, so we discard imaginary components
        # and flip the sign of negative eigenvalues.
        G_eigenvalues = G_eigenvalues.real
        negative_inds = G_eigenvalues < 0
        G_eigenvectors[:, negative_inds] *= -1
        G_eigenvalues[negative_inds] *= -1
        s_vals = np.sqrt(G_eigenvalues)

        # Sort singular values in descending order and organize into truncated
        # eigenvector (left-singular vector) and singular value matrices.
        sorted_inds = np.argsort(-s_vals)
        Q = G_eigenvectors[:, sorted_inds]
        s_sorted = s_vals[sorted_inds]
        r = self._compute_rank_from_svals(s_sorted)
        Q = Q[:, :r]
        Sigma = np.diag(s_sorted[:r])

        return Q, Sigma

    def _compute_rank_from_svals(self, s):
        """
        Rank computation for the truncated Singular Value Decomposition given
        only singular values.

        :param numpy.ndarray s: the singular values.
        :return: the computed rank truncation.
        :rtype: int
        """
        if 0 < self._svd_rank < 1:
            cumulative_energy = np.cumsum(s**2 / (s**2).sum())
            return np.searchsorted(cumulative_energy, self._svd_rank) + 1
        elif self._svd_rank >= 1 and isinstance(self._svd_rank, int):
            return min(self._svd_rank, len(s))
        else:
            return len(s)


class EDMD(DMD):
    """
    Kernelized Extended DMD.

    :param svd_rank: the rank for the truncation; If positive integer, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation.
    :type svd_rank: int or float
    :param int tlsq_rank: rank truncation computing Total Least Square. Default
        is 0, meaning no truncation.
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
        metrics include `"additive_chi2"`, `"chi2"`, `"linear"`, `"poly"`,
        `"polynomial"`, `"rbf"`, `"laplacian"`, `"sigmoid"`, and `"cosine"`.
    :type kernel_metric: str
    :param kernel_params: additional parameters for the
        `sklearn.metrics.pairwise_kernels` function, including
        kernel-specific function parameters.
    :type kernel_params: dict
    """

    def __init__(
        self,
        svd_rank=-1,
        tlsq_rank=0,
        opt=False,
        kernel_metric="linear",
        kernel_params=None,
    ):
        if svd_rank == 0:
            raise ValueError("svd_rank = 0 functionality is not supported.")

        if kernel_params is None:
            kernel_params = {}
        elif not isinstance(kernel_params, dict):
            raise TypeError("kernel_params must be a dict.")

        self._test_kernel_inputs(kernel_metric, kernel_params)

        super().__init__(
            svd_rank=svd_rank,
            tlsq_rank=tlsq_rank,
            exact=True,
            opt=opt,
        )

        self._Atilde = EDMDOperator(svd_rank, kernel_metric, kernel_params)

        self._kernel_metric = kernel_metric
        self._kernel_params = kernel_params
        self._svd_modes = None

    def eigenfunctions(self, x):
        """
        :param x: array from the original snapshot domain denoting the point
            at which to compute the EDMD eigenfunctions.
        :type x: numpy.ndarray
        :return: array containing all r EDMD eigenfunctions evaluated at the
            given domain point, where r denotes the rank of the EDMD fit.
        :rtype: numpy.ndarray
        """
        if self._svd_modes is None:
            raise ValueError("You need to call fit before")

        if not isinstance(x, np.ndarray) or np.ndim(x) != 1:
            raise ValueError("Input x must be a 1-D numpy array.")

        K_xx = pairwise_kernels(
            x[None],
            self.snapshots.T,
            self._kernel_metric,
            **self._kernel_params,
        )

        return np.linalg.multi_dot(
            [
                K_xx,
                self._svd_modes,
                np.linalg.pinv(np.diag(self.operator.svd_vals)),
                self.operator.eigenvectors,
            ]
        )

    @staticmethod
    def _test_kernel_inputs(kernel_metric, kernel_params):
        """
        Helper function that uses a dummy array of data in order to
        call `sklearn.metrics.pairwise_kernels` using the user-given
        kernel parameters. Ensures that the given kernel parameters
        produce valid kernel matrices.
        """
        x_dummy = np.arange(4).reshape(2, 2)
        try:
            pairwise_kernels(
                x_dummy, x_dummy, metric=kernel_metric, **kernel_params
            )
        except Exception as e:
            msg = (
                "Invalid kernel parameters given to the EDMD model. "
                "Please verify that kernel_metric and kernel_params "
                "are passable to sklearn.metrics.pairwise_kernels."
            )
            raise ValueError(msg) from e
