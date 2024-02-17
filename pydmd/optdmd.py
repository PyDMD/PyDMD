"""
Derived module from :meth:`pydmd.dmdbase` for the optimal closed-form solution
to dmd.

.. note::

    P. Heas & C. Herzet. Low-rank dynamic mode decomposition: optimal
    solution in polynomial time. arXiv:1610.02962. 2016.

"""

import numpy as np
from scipy.linalg import eig

from .dmdbase import DMDBase
from .dmdoperator import DMDOperator
from .snapshots import Snapshots
from .utils import compute_svd, compute_tlsq


def pinv_diag(x):
    """
    Utility function to compute the pseudo-inverse of a diagonal matrix.

    :param array_like x: diagonal of the matrix to be pseudo-inversed.
    :return: the computed pseudo-inverse
    :rtype: numpy.ndarray
    """
    t = x.dtype.char.lower()
    factor = {"f": 1e2, "d": 1e4}
    rcond = factor[t] * np.finfo(t).eps

    y = np.zeros(*x.shape)

    y[x > rcond] = np.reciprocal(x[x > rcond])

    return np.diag(y)


class DMDOptOperator(DMDOperator):
    """
    DMD operator for OptDMD.

    :param svd_rank: the rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive interger, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation.
    :type svd_rank: int or float
    :param str factorization: compute either the eigenvalue decomposition of
        the unknown high-dimensional DMD operator (factorization="evd") or
        its singular value decomposition (factorization="svd"). Default is
        "evd".
    """

    def __init__(self, svd_rank, factorization):
        super().__init__(
            svd_rank=svd_rank,
            exact=True,
            forward_backward=False,
            rescale_mode=None,
            sorted_eigs=False,
            tikhonov_regularization=None,
        )
        self._factorization = factorization

    @property
    def right_eigenvectors(self):
        if self._factorization == "evd":
            return self._right_eigenvectors
        raise ValueError("Eigenquantities haven't been computed yet.")

    def compute_operator(self, X, Y):
        """
        Compute the low-rank operator.

        :param numpy.ndarray X: matrix containing the snapshots x0,..x{n-1} by
            column.
        :param numpy.ndarray Y: matrix containing the snapshots x1,..x{n} by
            column.
        :return: Left singular vectors of Z, and Q.
        :rtype: numpy.ndarray, numpy.ndarray
        """

        Ux, Sx, Vx = compute_svd(X, -1)

        Z = np.linalg.multi_dot(
            [Y, Vx, np.diag(Sx), pinv_diag(Sx), Vx.T.conj()]
        )

        Uz, _, _ = compute_svd(Z, self._svd_rank)

        Q = np.linalg.multi_dot(
            [Uz.T.conj(), Y, Vx, pinv_diag(Sx), Ux.T.conj()]
        ).T.conj()

        self._Atilde = Q.T.conj().dot(Uz)
        if self._factorization == "evd":
            self._compute_eigenquantities(Uz, Q)

        return Uz, Q

    """
        Private method that computes eigenvalues and eigenvectors of the
        low-dimensional operator.

        :param numpy.ndarray P: Left singular vectors of Z.
        :param numpy.ndarray Q: The matrix Q.
    """

    def _compute_eigenquantities(self, P, Q):
        Atilde = self.as_numpy_array

        vals, vecs_left, vecs_right = eig(Atilde, left=True, right=True)

        # --> Build the matrix of right eigenvectors.
        right_vecs = np.linalg.multi_dot([P, Atilde, vecs_right])
        right_vecs = right_vecs.dot(pinv_diag(vals))

        # --> Build the matrix of left eigenvectors.
        left_vecs = Q.dot(vecs_left)
        left_vecs = left_vecs.dot(pinv_diag(vals))

        # --> Rescale the left eigenvectors.
        m = np.diag(left_vecs.T.conj().dot(right_vecs))
        left_vecs = left_vecs.dot(pinv_diag(m))

        self._eigenvalues = vals
        self._eigenvectors = left_vecs
        self._right_eigenvectors = right_vecs

    def _compute_modes(self, Y, U, Sigma, V):
        raise NotImplementedError("This function has not been implemented yet.")


class OptDMD(DMDBase):
    """
    Dynamic Mode Decomposition

    This class implements the closed-form solution to the DMD minimization
    problem. It relies on the optimal solution given by [HeasHerzet16]_.

    .. [HeasHerzet16] P. Heas & C. Herzet. Low-rank dynamic mode decomposition:
        optimal solution in polynomial time. arXiv:1610.02962. 2016.

    :param str factorization: compute either the eigenvalue decomposition of
        the unknown high-dimensional DMD operator (factorization="evd") or
        its singular value decomposition (factorization="svd"). Default is
        "evd".
    :param svd_rank: the rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive interger, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation.
    :type svd_rank: int or float
    :param int tlsq_rank: rank truncation computing Total Least Square. Default
        is 0, that means TLSQ is not applied.
    :param opt: argument to control the computation of DMD modes amplitudes. See
        :class:`DMDBase`. Default is False.
    :type opt: bool or int
    """

    def __init__(self, factorization="evd", svd_rank=0, tlsq_rank=0, opt=False):
        self._factorization = factorization
        self._tlsq_rank = tlsq_rank

        self._Atilde = DMDOptOperator(
            svd_rank=svd_rank, factorization=factorization
        )

        self._svds = None
        self._input_space = None
        self._output_space = None
        self._input_holder = None
        self._output_holder = None

    @property
    def factorization(self):
        return self._factorization

    @property
    def modes(self):
        return self._output_space

    @property
    def eigs(self):
        return self.operator.eigenvalues

    @property
    def amplitudes(self):
        return self._b

    def fit(self, X, Y=None):
        """
        Compute the Dynamic Modes Decomposition to the input data.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        :param Y: the input snapshots at sequential timestep, if passed. Default
            is None.
        :type Y: numpy.ndarray or iterable
        """
        self._reset()

        if Y is None:
            self._snapshots_holder = Snapshots(X)

            X = self.snapshots[:, :-1]  # x = x[k]
            Y = self.snapshots[:, 1:]  # y = x[k+1]
        else:
            self._input_holder = Snapshots(X)
            X = self._input_holder.snapshots

            self._output_holder = Snapshots(Y)
            Y = self._output_holder.snapshots

        X, Y = compute_tlsq(X, Y, self._tlsq_rank)
        Uz, Q = self.operator.compute_operator(X, Y)

        if self.factorization == "svd":
            # --> DMD basis for the input space.
            self._input_space = Q

            # --> DMD basis for the output space.
            self._output_space = Uz

        elif self.factorization == "evd":
            # --> Compute DMD eigenvalues and right/left eigenvectors
            self._input_space = self.eigs
            self._output_space = self.operator.right_eigenvectors

        return self

    def predict(self, X):
        """
        Predict the output Y given the input X using the fitted DMD model.

        :param numpy.ndarray X: the input vector.
        :return: one time-step ahead predicted output.
        :rtype: numpy.ndarray
        """
        if self.factorization == "svd":
            Y = np.linalg.multi_dot(
                [self._output_space, self._input_space.T.conj(), X]
            )
        elif self.factorization == "evd":
            Y = np.linalg.multi_dot(
                [
                    self._output_space,
                    np.diag(self._eigs),
                    self._input_space.T.conj(),
                    X,
                ]
            )

        return Y

    @property
    def modes_activation_bitmask(self):
        raise RuntimeError("This feature has not been implemented yet.")

    @modes_activation_bitmask.setter
    def modes_activation_bitmask(self, value):
        raise RuntimeError("This feature has not been implemented yet.")

    def _compute_amplitudes(self, modes, snapshots, eigs, opt):
        raise NotImplementedError("This function has not been implemented yet.")

    @property
    def dynamics(self):
        raise NotImplementedError("This function has not been implemented yet.")

    @property
    def fitted(self):
        raise NotImplementedError("This function has not been implemented yet.")

    @property
    def modes_activation_bitmask(self):
        raise NotImplementedError("This function has not been implemented yet.")

    @modes_activation_bitmask.setter
    def modes_activation_bitmask(self, value):
        raise NotImplementedError("This function has not been implemented yet.")
