#
#
#
#
#

# --> Import standard python packages
import numpy as np
from scipy.linalg import eig, eigvals
from scipy.linalg import pinv2

pinv = lambda x: pinv2(x, rcond=-1)

# --> Import PyDMD base class for DMD.
from .dmdbase import DMDBase


class optDMD(DMDBase):

    """
    Dynamic Mode Decomposition

    This class implements the closed-form solution to the DMD minimization
    problem. It relies on the optimal solution and the demonstration given by
    P. Héas & C. Herzet [1].

    [1] P. Héas & C. Herzet. Low-rank dynamic mode decomposition: optimal
    solution in polynomial time. arXiv:1610.02962. 2016

    """

    def __init__(self, factorization="svd", svd_rank=0):
        """Initialize the DMD model.

        Parameters
        ----------
        factorization : string
            Compute either the eigenvalue decomposition of the unknown high-
            -dimensional DMD operator (factorization="evd") or its singular value
            decomposition, i.e. factorization="svd". (the default is "svd").

        Returns
        -------
        self
            The instantiate DMD model.

        """

        # --> Compute either the SVD or EVD factorization of the DMD problem.
        self.factorization = factorization
        self.svd_rank = svd_rank

    def fit(self, X, Y=None):
        """Fit the DMD model using the input X and output Y.

        Parameters
        ----------
        X : array-like, shape (n_features, n_samples)
            Description of parameter `X`.
        Y : array-like, shape (n_features, n_samples)
            Description of parameter `Y` (the default is None).

        Returns
        -------
        type
            Description of returned object.

        """

        # --> Check whether Y is given or not.
        if Y is None:
            # --> Assume X includes a single realization of the dynamic process.
            Y = X[:, 1:]    # y = x[k+1]
            X = X[:, :-1]   # x = x[k]

        # --> Compute the singular value decomposition of X.
        Ux, Sx, Vx = self._compute_svd(X, X.shape[1])

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
            # --> DMD-SVD modes.
            self._modes = Uz
            # --> Low-dimensional DMD operator.
            self._Atilde = S
            # --> Eigenvalues of S.
            self._eigs = eigvals(S)

        elif self.factorization == "evd":
            # --> Compute the eigentriplets of the low-dimensional DMD matrix.
            vals, vecs_left, vecs_right = eig(S, left=True, right=True)

            # --> Build the matrix of right eigenvectors.
            right_vecs = np.linalg.multi_dot(
                [Uz, Q.T, Uz, vecs_right]
            )
            right_vecs[:, abs(vals)>1e-6] /= vals[abs(vals)>1e-6]

            # --> Build the matrix of left eigenvectors.
            left_vecs = np.linalg.multi_dot(
                [Q, vecs_left]
            )
            left_vecs[:, abs(vals)>1e-6] /= vals[abs(vals)>1e-6]

            # --> Rescale the left eigenvectors.
            m = np.diag(left_vecs.T.conj().dot(right_vecs))
            left_vecs[:, abs(vals)>1e-6] /= m[abs(vals)>1e-6].conj()

            self._input_space = left_vecs
            self._output_space = right_vecs
            self._Atilde = np.diag(vals)
            self._eigs = vals

        return self

    def predict(self, X):
        """Short summary.

        Parameters
        ----------
        X : numpy array
            Description of parameter `X`.

        Returns
        -------
        Y : numpy array
            Description of returned object `Y`.

        """

        # --> Factorization : SVD.
        if self.factorization == "svd":
            Y = np.linalg.multi_dot(
                [self._modes, self._Atilde, self._modes.T.conj(), X]
            )
        elif self.factorization == "evd":
            raise NotImplementedError(
                "Prediction using the EVD factorization is not implemented yet."
                )

        return Y
