import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

class DMDOperator(object):
    def __init__(self, svd_rank, exact, forward_backward, rescale_mode):
        self._exact = exact
        self._rescale_mode = rescale_mode
        self._svd_rank = svd_rank
        self._forward_backward = forward_backward

    def compute_operator(self, X, Y):
        U, s, V = self._compute_svd(X)

        atilde = self._least_square_operator(U, s, V, Y)

        if self._forward_backward:
            # b stands for "backward"
            bU, bs, bV = self._compute_svd(Y, svd_rank=self._svd_rank)
            atilde_back = self._least_square_operator(bU, bs, bV, X)
            atilde = sqrtm(atilde.dot(np.linalg.inv(atilde_back)))

        if self._rescale_mode == 'auto':
            self._rescale_mode = s

        self._Atilde = atilde
        self._compute_eigenquantities()
        self._compute_modes(Y, U, s, V)

        self._svd_modes = U

        return U, s, V

    @property
    def shape(self):
        return self.as_numpy_array.shape

    def __call__(self, snapshot_lowrank_modal_coefficients):
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
        return self._Atilde

    def _compute_svd(self, X, svd_rank=None):
        """
        Truncated Singular Value Decomposition.

        :param numpy.ndarray X: the matrix to decompose.
        :param svd_rank: the rank for the truncation; If 0, the method computes
            the optimal rank and uses it for truncation; if positive interger,
            the method uses the argument for the truncation; if float between 0
            and 1, the rank is the number of the biggest singular values that
            are needed to reach the 'energy' specified by `svd_rank`; if -1,
            the method does not compute truncation.
        :type svd_rank: int or float
        :return: the truncated left-singular vectors matrix, the truncated
            singular values array, the truncated right-singular vectors matrix.
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray

        References:
        Gavish, Matan, and David L. Donoho, The optimal hard threshold for
        singular values is, IEEE Transactions on Information Theory 60.8
        (2014): 5040-5053.
        """
        U, s, V = np.linalg.svd(X, full_matrices=False)
        V = V.conj().T

        if svd_rank is None:
            svd_rank = self._svd_rank

        if svd_rank == 0:
            omega = lambda x: 0.56 * x**3 - 0.95 * x**2 + 1.82 * x + 1.43
            beta = np.divide(*sorted(X.shape))
            tau = np.median(s) * omega(beta)
            rank = np.sum(s > tau)
        elif svd_rank > 0 and svd_rank < 1:
            cumulative_energy = np.cumsum(s**2 / (s**2).sum())
            rank = np.searchsorted(cumulative_energy, svd_rank) + 1
        elif svd_rank >= 1 and isinstance(svd_rank, int):
            rank = min(svd_rank, U.shape[1])
        else:
            rank = X.shape[1]

        U = U[:, :rank]
        V = V[:, :rank]
        s = s[:rank]

        return U, s, V

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
        :param numpy.ndarray s: 1D array that contains the singular values of X.
        :param numpy.ndarray V: 2D matrix that contains the right-singular
            vectors of X, stored by row.
        :param numpy.ndarray Y: input matrix Y.
        :return: the lowrank operator
        :rtype: numpy.ndarray
        """
        return U.T.conj().dot(Y).dot(V) * np.reciprocal(s)

    def _compute_eigenquantities(self):
        """
        Private method that computes eigenvalues and eigenvectors of the
        low-dimensional operator, scaled according to self._rescale_mode.
        """

        if self._rescale_mode is None:
            # scaling isn't required
            Ahat = self._Atilde
        else:
            print('upakjiwecbdsicsdiucdbsidbcsdicubisudbc')
            if len(self._rescale_mode) != self.as_numpy_array.shape[0]:
                raise ValueError('''Scaling by an invalid number of
                        coefficients''')
            scaling_factors_array = self._rescale_mode

            factors_inv_sqrt = np.diag(np.power(scaling_factors_array, -0.5))
            factors_sqrt = np.diag(np.power(scaling_factors_array, 0.5))
            Ahat = np.linalg.multi_dot([factors_inv_sqrt, self.as_numpy_array,
                factors_sqrt])

        self._eigenvalues, self._eigenvectors = np.linalg.eig(Ahat)

    def _compute_modes(self, Y, U, Sigma, V):
        """
        Private method that computes eigenvalues and eigenvectors of the
        high-dimensional operator.
        """

        if self._rescale_mode is None:
            W = self.eigenvectors
        else:
            # compute W as shown in arXiv:1409.5496 (section 2.4)
            factors_sqrt = np.diag(np.power(self._rescale_mode, 0.5))
            W = factors_sqrt.dot(self.eigenvectors)

        # compute the eigenvectors of the high-dimensional operator
        if self._exact:
            high_dimensional_eigenvectors = ((Y.dot(V) *
                             np.reciprocal(Sigma)).dot(W))
        else:
            high_dimensional_eigenvectors = U.dot(W)

        # eigenvalues are the same of lowrank
        high_dimensional_eigenvalues = self.eigenvalues

        self._modes = high_dimensional_eigenvectors
        self._Lambda = high_dimensional_eigenvalues

    def plot_operator(self):
        matrix = self.as_numpy_array
        cmatrix = matrix.real
        rmatrix = matrix.imag

        if np.linalg.norm(cmatrix) > 1.e-12:
            fig, axes = plt.subplots(nrows=1, ncols=2)

            axes[0].set_title('Real')
            axes[0].matshow(rmatrix, cmap='jet')
            axes[1].set_title('Complex')
            axes[1].matshow(cmatrix, cmap='jet')
        else:
            plt.title('Real')
            plt.matshow(rmatrix)
        plt.show()
