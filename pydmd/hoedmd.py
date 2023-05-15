"""
Derived module from dmdbase.py for higher order edmd.
As a reference consult this work by Weiyang and Jie:
https://doi.org/10.1137/21M1463665
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv
import scipy
from .dmdbase import DMDBase
from .dmdoperator import DMDOperator
from .snapshots import Snapshots
from .utils import compute_svd


class HOEDMDOperator(DMDOperator):
    """
    DMD operator for HigherOrder-EDMD.

    :param svd_rank: the rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive interger, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation.
    :type svd_rank: int or float
    :param sorted_eigs: Sort eigenvalues (and modes/dynamics accordingly) by
        magnitude if `sorted_eigs='abs'`, by real part (and then by imaginary
        part to break ties) if `sorted_eigs='real'`. Default: False.
    :type sorted_eigs: {'real', 'abs'} or False
    """

    def __init__(
        self,
        svd_rank,
        rescale_mode,
        forward_backward,
        sorted_eigs,
        tikhonov_regularization,
    ):
        super().__init__(
            svd_rank=svd_rank,
            exact=True,
            rescale_mode=rescale_mode,
            forward_backward=forward_backward,
            sorted_eigs=sorted_eigs,
            tikhonov_regularization=tikhonov_regularization,
        )
        self._Atilde = None

    def compute_operator(self, Psi, d, alg_type):
        """
        Compute the low-rank operator.

        :param numpy.ndarray Psi: the time-delay data matrix containing the snapshots x0,..x{n} by column.
        :param int d: order in HOEDMD.
        :param string alg_type: Algorithm type in HOEDMD. Default is 'stls'.
        :return: the (truncated) left-singular vectors matrix, the (truncated)
            singular values array, the (truncated) right-singular vectors
            matrix of Px.
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
        """
        m = Psi.shape[0] // (d + 1)
        if alg_type == "stls":
            P, _, _ = compute_svd(
                Psi, svd_rank=self._svd_rank
            )  # solve for tlsq
            Px = P[:-d, :]
            Py = P[d:, :]
            M, omega, N = compute_svd(
                Px, svd_rank=self._svd_rank
            )  # solve for dmd
            o_inv = np.reciprocal(omega)
            atilde = np.linalg.multi_dot([N * o_inv, M.T.conj(), Py])

            self._Atilde = atilde
            self._compute_eigenquantities()
            U = self.eigenvectors_r
            V = self.eigenvectors_l
            inprodUV = self.eigenvalues.conj() * ((U.conj() * V).sum(0))
            U = P[m : 2 * m, :].dot(U)
            V = np.linalg.multi_dot([(M * o_inv), N.T.conj(), V])
            self._compute_modes(U, V, inprodUV)

        elif alg_type == "nosvd":
            Gxx = np.dot(Psi[:-d, :].T.conj(), Psi[:-d, :])
            Gxy = np.dot(Psi[:-d, :].T.conj(), Psi[d:, :])
            Gpp = np.dot(Psi[m * d :, :].T.conj(), Psi[m * d :, :])
            sigma, Q = np.linalg.eig(Gxx + Gpp)
            ind = np.sort(-sigma)
            sigma = sigma(ind)
            Q = Q[:, ind[: self._svd_rank]]
            L = scipy.linalg.cholesky(Q.T.conj() * Gxx * Q, lower=True)
            atilde = np.linalg.solve(
                L.T.conj(), np.linalg.solve(L, Q.T.conj() * Gxy * Q)
            )
            self._Atilde = atilde
            self._compute_eigenquantities()
            U = self.eigenvectors_r
            V = self.eigenvectors_l
            inprodUV = self.eigenvalues.conj() * ((U.conj() * V).sum(0))
            U = Psi[m : 2 * m, :] * (Q * U)
            V = Psi[: m * d, :] * (
                Q * np.linalg.solve(L.T.conj(), np.linalg.solve(L, V))
            )
            self._compute_modes(U, V, inprodUV)

        else:
            raise ValueError("Invalid value for alg_type: {}".format(alg_type))

    @property
    def eigenvectors_r(self):
        if not hasattr(self, "_eigenvectors_r"):
            raise ValueError("You need to call fit before")
        return self._eigenvectors_r

    @property
    def eigenvectors_l(self):
        if not hasattr(self, "_eigenvectors_l"):
            raise ValueError("You need to call fit before")
        return self._eigenvectors_l

    @property
    def Vhat(self):
        if not hasattr(self, "_Vhat"):
            raise ValueError("You need to call fit before")
        return self._Vhat

    def _compute_eigenquantities(self):
        """
        Private method that computes eigenvalues and eigenvectors(both right and left) of the low-dimensional operator
        """

        if self._rescale_mode is None:
            Ahat = self._Atilde

        (
            self._eigenvalues,
            self._eigenvectors_r,
            self._eigenvectors_l,
        ) = scipy.linalg.eig(Ahat, left=True)

        if self._sorted_eigs is not False and self._sorted_eigs is not None:
            if self._sorted_eigs == "abs":

                def k(tp):
                    return abs(tp[0])

            elif self._sorted_eigs == "real":

                def k(tp):
                    eig = tp[0]
                    if isinstance(eig, complex):
                        return (eig.real, eig.imag)
                    return (eig.real, 0)

            else:
                raise ValueError(
                    "Invalid value for sorted_eigs: {}".format(
                        self._sorted_eigs
                    )
                )

            # each column is an eigenvector, therefore we take the
            # transpose to associate each row (former column) to an
            # eigenvalue before sorting
            a, b, c = zip(
                *sorted(
                    zip(
                        self._eigenvalues,
                        self._eigenvectors_r.T,
                        self._eigenvectors_l.T,
                    ),
                    key=k,
                )
            )
            self._eigenvalues = np.array([eig for eig in a])
            # we restore the original condition (eigenvectors in columns)
            self._eigenvectors_r = np.array([vec for vec in b]).T
            self._eigenvectors_l = np.array([vec for vec in c]).T

    def _compute_modes(self, U, V, inprodUV):
        """
        Private method that computes eigenvalues and eigenvectors of the
        high-dimensional operator (stored in self.modes and self.Lambda).

        :param numpy.ndarray U: right eigenvectors from high-dimensional to original space.
        :param numpy.ndarray V: left eigenvectors from high-dimensional to original space.
        :param numpy.ndarray inprodUV: inner product of right eigenvectors and left eigenvectors from high-dimensional to original space.
        """

        # normalization
        unorms = np.linalg.norm(U, axis=0)
        U = U / unorms
        V = V * unorms / inprodUV  # need to check

        self._modes = U
        self._Lambda = self.eigenvalues
        self._Vhat = V


class HOEDMD(DMDBase):
    """
    Higher Order Extended Dynamic Mode Decomposition.

    :param svd_rank: the rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive interger, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation.
    :type svd_rank: int or float
    :param int tlsq_rank: rank truncation computing Total Least Square. Default
        is 0, that means TLSQ is not applied.
    :param int d: the new order for spatial dimension of the input snapshots.
        Default is 1.
    :param sorted_eigs: Sort eigenvalues (and modes/dynamics accordingly) by
        magnitude if `sorted_eigs='abs'`, by real part (and then by imaginary
        part to break ties) if `sorted_eigs='real'`.
    :type sorted_eigs: {'real', 'abs'} or False
    :param sorted_eigs: Sort eigenvalues (and modes/dynamics accordingly) by
        magnitude if `sorted_eigs='abs'`, by real part (and then by imaginary
        part to break ties) if `sorted_eigs='real'`. Default: False.
    :type sorted_eigs: {'real', 'abs'} or False
    """

    def __init__(
        self,
        svd_rank=0,
        alg_type="stls",
        tlsq_rank=0,
        exact=False,
        opt=False,
        rescale_mode=None,
        forward_backward=False,
        d=1,
        sorted_eigs=False,
        tikhonov_regularization=None,
    ):
        super().__init__(
            svd_rank=svd_rank,
            tlsq_rank=tlsq_rank,
            exact=exact,
            opt=opt,
            rescale_mode=rescale_mode,
            sorted_eigs=sorted_eigs,
        )
        self._d = d
        self._ho_snapshots = None
        self._alg_type = alg_type
        self._Atilde = HOEDMDOperator(
            svd_rank=svd_rank,
            rescale_mode=rescale_mode,
            forward_backward=forward_backward,
            sorted_eigs=sorted_eigs,
            tikhonov_regularization=tikhonov_regularization,
        )

    @property
    def d(self):
        """The new order for spatial dimension of the input snapshots."""
        return self._d

    @property
    def ho_snapshots(self):
        """
        Get the time-delay data matrix.

        :return: the matrix that contains the time-delayed data.
        :rtype: numpy.ndarray
        """
        return self._ho_snapshots

    def _stacked_matrix(self, X):
        """
        Arrange the snapshot in the matrix `X` into the stacked
        matrix. The attribute `d` controls the number of snapshot from `X` in
        each snapshot of the stacked matrix.

        :Example:

            >>> from pydmd import HOEDMD
            >>> import numpy as np

            >>> dmd = HOEDMD(d=2)
            >>> a = np.array([[1, 2, 3, 4, 5]])
            >>> dmd._stacked_matrix(a)
                array([[1, 2, 3],
                       [2, 3, 4],
                       [3, 4, 5]])
            >>> dmd = HOEDMD(d=4)
            >>> dmd._stacked_matrix(a)
            array([[1],
                   [2],
                   [3],
                   [4],
                   [5]])

            >>> dmd = HOEDMD(d=2)
            >>> a = np.array([1,2,3,4,5,6]).reshape(2,3)
            >>> a
            array([[1, 2, 3],
                   [4, 5, 6]])
            >>> dmd._stacked_matrix(a)
            array([[1],
                   [4],
                   [2],
                   [5],
                   [3],
                   [6]])
        """
        return (swv(X, (X.shape[0], self.d + 1))).T.reshape(
            -1, X.shape[1] - self.d
        )

    def fit(self, X):
        """
        Compute the Dynamic Modes Decomposition to the input data.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        """
        self._reset()

        # generate stacked matrix from snapshots
        self._snapshots_holder = Snapshots(X)
        n_samples = self.snapshots.shape[-1]
        if n_samples < self._d:
            msg = """The number of snapshots provided is not enough for d={}.
Expected at least d."""
            raise ValueError(msg.format(self._d))
        self._ho_snapshots = Snapshots(
            self._stacked_matrix(self.snapshots)
        ).snapshots

        # compute HOEDMD
        self.operator.compute_operator(
            self._ho_snapshots, self.d, self._alg_type
        )

        # Default timesteps
        self._set_initial_time_dictionary(
            {"t0": 0, "tend": n_samples - 1, "dt": 1}
        )

        # compute amplitudes
        self._b = self.operator.Vhat.T.conj() * self._ho_snapshots[: -self.d, 1]

        return self
