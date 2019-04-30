"""
Derived module from dmdbase.py for the optimal closed-form solution to dmd.
"""

# --> Import PyDMD base class for DMD.
from .dmdbase import DMDBase

# --> Import standard python packages
import numpy as np
from scipy.linalg import eig, eigvals
from scipy.linalg import svdvals
from scipy.linalg import pinv2


def pinv(x): return pinv2(x, rcond=10*np.finfo(float).eps)


class OptDMD(DMDBase):
    """
    Dynamic Mode Decomposition

    This class implements the closed-form solution to the DMD minimization
    problem. It relies on the optimal solution given by Heas & Herzet [1].

    Parameters
    ----------
    factorization : string
        Compute either the eigenvalue decomposition of the unknown high-
        -dimensional DMD operator (factorization="evd") or its singular value
        decomposition, i.e. factorization="svd". (the default is "evd").

    svd_rank : int or float
        The rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive interger, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation.

    References
    ----------
    [1] P. Héas & C. Herzet. Low-rank dynamic mode decomposition: optimal
    solution in polynomial time. arXiv:1610.02962. 2016.

    """

    def __init__(self, factorization="evd", svd_rank=0):

        # --> Compute either the SVD or EVD factorization of the DMD problem.
        self.factorization = factorization

        # --> Rank of the DMD model.
        self.svd_rank = svd_rank

    def fit(self, X, Y=None):
        """
        Fit the DMD model using the input X and output Y (if provided).

        Parameters
        ----------
        X : array-like, shape (n_features, n_samples)
            Description of parameter `X`.
        Y : array-like, shape (n_features, n_samples)
            Description of parameter `Y` (the default is None).

        Returns
        -------
        self : optDMD object
            The fitted DMD model.

        """

        # --> Check whether Y is given or not.
        if Y is None:
            # --> Assume X includes a single realization of the dynamic process.
            Y = X[:, 1:]    # y = x[k+1]
            X = X[:, :-1]   # x = x[k]

        # --> Compute the singular value decomposition of X.
        Ux, Sx, Vx = self._compute_svd(X, -1)

        # --> Compute the matrix Z.
        Z = np.linalg.multi_dot(
            [Y, Vx, np.diag(Sx), pinv(np.diag(Sx)), Vx.conj().T]
        )

        # --> Compute the singular value decomposition of Z.
        Uz, _, _ = self._compute_svd(Z, self.svd_rank)

        # --> Compute the Q matrix.
        Q = np.linalg.multi_dot(
            [Uz.T, Y, Vx, pinv(np.diag(Sx)), Ux.T.conj()]
        ).T

        # --> Compute the low-dimensional DMD operator.
        S = Q.T.dot(Uz)

        if self.factorization == "svd":
            # --> Input/Output spaces.
            self._input_space = Q
            self._output_space = Uz

            # --> DMD-SVD modes.
            self._modes = Uz

            # --> Low-dimensional DMD operator.
            self._Atilde = S

            # --> Eigenvalues of Atilde.
            self._eigs = eigvals(S)

            # --> Singular values of Atilde.
            self._svds = svdvals(S)

        elif self.factorization == "evd":
            # --> Compute the eigentriplets of the low-dimensional DMD matrix.
            vals, vecs_left, vecs_right = eig(S, left=True, right=True)

            # --> Build the matrix of right eigenvectors.
            right_vecs = np.linalg.multi_dot(
                [Uz, Q.T, Uz, vecs_right]
            )
            right_vecs[:, abs(vals) > 1e-8] /= vals[abs(vals) > 1e-8]

            # --> Build the matrix of left eigenvectors.
            left_vecs = Q.dot(vecs_left)
            left_vecs[:, abs(vals) > 1e-8] /= vals[abs(vals) > 1e-8]

            # --> Rescale the left eigenvectors.
            m = np.diag(left_vecs.T.conj().dot(right_vecs))
            left_vecs[:, abs(vals) > 1e-8] /= m[abs(vals) > 1e-8].conj()

            # --> Input/Output modes.
            self._input_space = left_vecs
            self._output_space = right_vecs

            # --> DMD-EVD modes.
            self._modes = right_vecs

            # --> Low-dimensional DMD operator.
            self._Atilde = S

            # --> Eigenvalues of Atilde.
            self._eigs = vals

            # --> Singular values of Atilde.
            self._svds = svdvals(S)

        return self

    def predict(self, X):
        """
        Predict the output Y given the input X using the fitted DMD model.

        Parameters
        ----------
        X : numpy array
            Input data.

        Returns
        -------
        Y : numpy array
            Predicted output.

        """

        # --> Factorization : SVD.
        if self.factorization == "svd":
            Y = np.linalg.multi_dot(
                [self._output_space, self._input_space.T.conj(), X]
            )
        # --> Factorization : EVD.
        elif self.factorization == "evd":
            Y = np.linalg.multi_dot(
                [self._output_space, np.diag(self._eigs), self._input_space.T.conj(), X]
            )

        return Y
