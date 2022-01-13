import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

from .utils import compute_svd


class DMDOperator():
    """
    Dynamic Mode Decomposition standard operator class. Non-standard ways of
    computing the low-rank Atilde operator should be coded into subclasses of
    this class.

    :param svd_rank: the rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive interger, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation.
    :type svd_rank: int or float
    :param bool exact: flag to compute either exact DMD or projected DMD.
        Default is False.
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
    :param tikhonov: tikhonov parameter for regularization
        If 0, no regularization is applied, if float, it is used as the lambda tikhonov 
        parameter
    :type tikhonov: int or float
    """

    def __init__(self, svd_rank, exact, forward_backward, rescale_mode,
                 sorted_eigs, tikhonov):
        self._exact = exact
        self._rescale_mode = rescale_mode
        self._svd_rank = svd_rank
        self._forward_backward = forward_backward
        self._sorted_eigs = sorted_eigs
        self._tikhonov = tikhonov

    def compute_operator(self, X, Y):
        """
        Compute the low-rank operator.

        :param numpy.ndarray X: matrix containing the snapshots x0,..x{n-1} by
            column.
        :param numpy.ndarray Y: matrix containing the snapshots x1,..x{n} by
            column.
        :return: the (truncated) left-singular vectors matrix, the (truncated)
            singular values array, the (truncated) right-singular vectors
            matrix of X.
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
        """

        U, s, V = compute_svd(X, self._svd_rank)

        if self._tikhonov:
            self._norm_X = np.linalg.norm(X)
        atilde = self._least_square_operator(U, s, V, Y)

        if self._forward_backward:
            # b stands for "backward"
            bU, bs, bV = compute_svd(Y, svd_rank=len(s))
            atilde_back = self._least_square_operator(bU, bs, bV, X)
            atilde = sqrtm(atilde.dot(np.linalg.inv(atilde_back)))

        if self._rescale_mode == 'auto':
            self._rescale_mode = s

        self._Atilde = atilde
        self._compute_eigenquantities()
        self._compute_modes(Y, U, s, V)

        return U, s, V

    @property
    def shape(self):
        """Shape of the operator"""
        return self.as_numpy_array.shape

    def __call__(self, snapshot_lowrank_modal_coefficients):
        """
        Apply the low-rank operator to a vector of the modal coefficients of a
        snapshot(s).

        :param numpy.ndarray snapshot_lowrank_modal_coefficients: low-rank
            representation (in modal coefficients) of a snapshot x{n}.
        :return: low-rank representation (in modal coefficients) of x{n+1}.
        :rtype: numpy.ndarray
        """

        return self._Atilde.dot(snapshot_lowrank_modal_coefficients)

    @property
    def eigenvalues(self):
        if not hasattr(self, '_eigenvalues'):
            raise ValueError('You need to call fit before')
        return self._eigenvalues

    @property
    def eigenvectors(self):
        if not hasattr(self, '_eigenvectors'):
            raise ValueError('You need to call fit before')
        return self._eigenvectors

    @property
    def modes(self):
        if not hasattr(self, '_modes'):
            raise ValueError('You need to call fit before')
        return self._modes

    @property
    def Lambda(self):
        if not hasattr(self, '_Lambda'):
            raise ValueError('You need to call fit before')
        return self._Lambda

    @property
    def as_numpy_array(self):
        if not hasattr(self, '_Atilde') or self._Atilde is None:
            raise ValueError('You need to call fit before')
        else:
            return self._Atilde

    def _least_square_operator(self, U, s, V, Y):
        """
        Private method that computes the lowrank operator from the singular
        value decomposition of matrix X and the matrix Y.

        .. math::

            \\mathbf{\\tilde{A}} =
            \\mathbf{U}^* \\mathbf{Y} \\mathbf{X}^\\dagger \\mathbf{U} =
            \\mathbf{U}^* \\mathbf{Y} \\mathbf{V} \\mathbf{S}^{-1}

        :param numpy.ndarray U: 2D matrix that contains the left-singular
            vectors of X, stored by column.
        :param numpy.ndarray s: 1D array that contains the singular values of
            X.
        :param numpy.ndarray V: 2D matrix that contains the right-singular
            vectors of X, stored by row.
        :param numpy.ndarray Y: input matrix Y.
        :return: the lowrank operator
        :rtype: numpy.ndarray
        """
        if self._tikhonov:
            s = (s**2 + self._tikhonov * self._norm_X) * np.reciprocal(s)
        return np.linalg.multi_dot([U.T.conj(), Y, V]) * np.reciprocal(s)

    def _compute_eigenquantities(self):
        """
        Private method that computes eigenvalues and eigenvectors of the
        low-dimensional operator, scaled according to self._rescale_mode.
        """

        if self._rescale_mode is None:
            # scaling isn't required
            Ahat = self._Atilde
        elif isinstance(self._rescale_mode, np.ndarray):
            if len(self._rescale_mode) != self.as_numpy_array.shape[0]:
                raise ValueError('''Scaling by an invalid number of
                        coefficients''')
            scaling_factors_array = self._rescale_mode

            factors_inv_sqrt = np.diag(np.power(scaling_factors_array, -0.5))
            factors_sqrt = np.diag(np.power(scaling_factors_array, 0.5))

            # if an index is 0, we get inf when taking the reciprocal
            for idx, item in enumerate(scaling_factors_array):
                if item == 0:
                    factors_inv_sqrt[idx] = 0

            Ahat = np.linalg.multi_dot([factors_inv_sqrt, self.as_numpy_array,
                                        factors_sqrt])
        else:
            raise ValueError('Invalid value for rescale_mode: {} of type {}'
                             .format(self._rescale_mode,
                                     type(self._rescale_mode)))

        self._eigenvalues, self._eigenvectors = np.linalg.eig(Ahat)

        if self._sorted_eigs is not False and self._sorted_eigs is not None:
            if self._sorted_eigs == 'abs':
                def k(tp):
                    return abs(tp[0])
            elif self._sorted_eigs == 'real':
                def k(tp):
                    eig = tp[0]
                    if isinstance(eig, complex):
                        return (eig.real, eig.imag)
                    return (eig.real, 0)
            else:
                raise ValueError('Invalid value for sorted_eigs: {}'.format(
                    self._sorted_eigs))

            # each column is an eigenvector, therefore we take the
            # transpose to associate each row (former column) to an
            # eigenvalue before sorting
            a, b = zip(*sorted(zip(self._eigenvalues, self._eigenvectors.T),
                               key=k))
            self._eigenvalues = np.array([eig for eig in a])
            # we restore the original condition (eigenvectors in columns)
            self._eigenvectors = np.array([vec for vec in b]).T

    def _compute_modes(self, Y, U, Sigma, V):
        """
        Private method that computes eigenvalues and eigenvectors of the
        high-dimensional operator (stored in self.modes and self.Lambda).

        :param numpy.ndarray Y: matrix containing the snapshots x1,..x{n} by
            column.
        :param numpy.ndarray U: (truncated) left singular vectors of X
        :param numpy.ndarray Sigma: (truncated) singular values of X
        :param numpy.ndarray V: (truncated) right singular vectors of X
        """

        if self._rescale_mode is None:
            W = self.eigenvectors
        else:
            # compute W as shown in arXiv:1409.5496 (section 2.4)
            factors_sqrt = np.diag(np.power(self._rescale_mode, 0.5))
            W = factors_sqrt.dot(self.eigenvectors)

        # compute the eigenvectors of the high-dimensional operator
        if self._exact:
            if self._tikhonov:
                Sigma = (Sigma**2 + self._tikhonov * self._norm_X) * np.reciprocal(Sigma)     
            high_dimensional_eigenvectors = ((Y.dot(V) *
                                              np.reciprocal(Sigma)).dot(W))
        else:
            high_dimensional_eigenvectors = U.dot(W)

        # eigenvalues are the same of lowrank
        high_dimensional_eigenvalues = self.eigenvalues

        self._modes = high_dimensional_eigenvectors
        self._Lambda = high_dimensional_eigenvalues

    def plot_operator(self):
        """
        Plot the low-rank Atilde operator
        """

        matrix = self.as_numpy_array
        cmatrix = matrix.real
        rmatrix = matrix.imag

        if np.linalg.norm(cmatrix) > 1.e-12:
            _, axes = plt.subplots(nrows=1, ncols=2)

            axes[0].set_title('Real')
            axes[0].matshow(rmatrix, cmap='jet')
            axes[1].set_title('Complex')
            axes[1].matshow(cmatrix, cmap='jet')
        else:
            plt.title('Real')
            plt.matshow(rmatrix)
        plt.show()
