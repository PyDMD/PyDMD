#
#
#
#
#

# --> Import standard python packages
import numpy as np
from scipy.linalg import eig

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

    def __init__(self, factorization="svd", rank=0):
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
        self.rank = rank

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
        Ux, Sx, Vx = self._compute_svd(X, 0)

        # --> Compute the matrix Z.
        Z = np.linalg.multi_dot([Y, Vx, Vx.conj().T])

        # --> Compute the singular value decomposition of Z.
        Uz, _, _ = self._compute_svd(Z, self.rank)

        # --> Compute the P and Q matrices.
        P = Uz
        Q = np.linalg.multi_dot(
            [P.T, Y, Vx, np.diag(np.reciprocal(Sx)), Ux.T.conj()]
        ).T

        S = Q.T.dot(P)

        if self.factorization == "svd":
            return P, S
        elif self.factorization == "evd":
            vals, vecs_left, vecs_right = eig(S, left=True, right=True)
            right_vecs = np.linalg.multi_dot(
                [P, Q.T, P, vecs_right]
            )
            left_vecs = np.linalg.multi_dot(
                [Q, vecs_left]
            )

            return left_vecs, right_vecs, np.diag(vals)
