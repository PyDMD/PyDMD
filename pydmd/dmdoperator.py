# Provides a basic implementation of an operator (most likely used to represent
# Atilde in DMD implementations).
# This class was subclassed following this reference:
# https://numpy.org/doc/stable/user/basics.subclassing.html
import numpy as np

class DMDOperator(np.ndarray):
    """
    Represents an operator, packed with some convenient methods, as a subclass
    of np.ndarray.
    """

    def __new__(cls, U, s=None, V=None, Y=None):
        if s is None or V is None or Y is None:
            array = U
        else:
            # use the default definition
            array = U.T.conj().dot(Y).dot(V) * np.reciprocal(s)
        return array.view(cls)

    # this is only for documentation purposes
    def __init__(self, U, s=None, V=None, Y=None):
        """
        Builds the lowrank operator from the singular value decomposition of
        matrix X and the matrix Y, or from a given matrix. If s, V or Y is None,
        the operator is equal to U.

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
        """
        super().__init__()

    def eigenvalues_eigenvectors(self, Y, U, s, V, exact, rescale_mode=None):
        """
        Private method that computes eigenvalues and eigenvectors of the
        high-dimensional operator from the low-dimensional operator and the
        input matrix.

        :param numpy.ndarray Y: input matrix Y.
        :param numpy.ndarray U: 2D matrix that contains the left-singular
            vectors of X, stored by column.
        :param numpy.ndarray s: 1D array that contains the singular values of X.
        :param numpy.ndarray V: 2D matrix that contains the right-singular
            vectors of X, stored by row.
        :param bool exact: if True, the exact modes are computed; otherwise,
            the projected ones are computed.
        :param rescale_mode: Scale Atilde as shown in
            10.1016/j.jneumeth.2015.10.010 (section 2.4) before computing its
            eigendecomposition. None means no rescaling, 'auto' means automatic
            rescaling using singular values, otherwise the scaling factors.
        :type rescale_mode: {'auto'} or None or numpy.ndarray
        :return: eigenvalues, eigenvectors
        :rtype: numpy.ndarray, numpy.ndarray
        """

        if rescale_mode is None:
            # scaling isn't required
            Ahat = self
        else:
            # rescale using the singular values (as done in the paper)
            if rescale_mode == 'auto':
                scaling_factors_array = s.copy()
            # rescale using custom values
            else:
                scaling_factors_array = rescale_mode

            factors_inv_sqrt = np.diag(np.power(scaling_factors_array, -0.5))
            factors_sqrt = np.diag(np.power(scaling_factors_array, 0.5))
            Ahat = factors_inv_sqrt.dot(self).dot(factors_sqrt)

        lowrank_hat_eigenvalues, lowrank_hat_eigenvectors = np.linalg.eig(Ahat)

        # eigenvalues are invariant wrt scaling
        lowrank_eigenvalues = lowrank_hat_eigenvalues

        if rescale_mode is None:
            lowrank_eigenvectors = lowrank_hat_eigenvectors
        else:
            # compute eigenvalues after scaling
            lowrank_eigenvectors = factors_sqrt.dot(lowrank_hat_eigenvectors)

        # Compute the eigenvectors of the high-dimensional operator
        if exact:
            eigenvectors = ((Y.dot(V) *
                             np.reciprocal(s)).dot(lowrank_eigenvectors))
        else:
            eigenvectors = U.dot(lowrank_eigenvectors)

        # The eigenvalues are the same
        eigenvalues = lowrank_eigenvalues

        return eigenvalues, eigenvectors
