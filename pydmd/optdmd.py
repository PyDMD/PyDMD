"""
Derived module from dmdbase.py for the optimal closed-form solution to dmd.
"""

# --> Import standard python packages
import numpy as np
from scipy.linalg import eig, eigvals, svdvals

# --> Import PyDMD base class for DMD.
from .dmdbase import DMDBase


def pinv_diag(x):

    """
    Utility function to compute the pseudo-inverse of a diagonal matrix.

    Parameters
    ----------

    x : array-like, shape (n,)
        Diagonal of the matrix to be pseudo-inversed.

    Returns
    -------

    y : array-like, shape (n, n)
        The computed pseudo-inverse.

    """

    # --> Set the tolerance to zero-out small values.
    t = x.dtype.char.lower()
    factor = {'f': 1E2, 'd': 1E4}
    rcond = factor[t] * np.finfo(t).eps

    # --> Initialize array.
    y = np.zeros(*x.shape)

    # --> Compute the pseudo-inverse.
    y[x > rcond] = np.reciprocal(x[x > rcond])

    return np.diag(y)


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
    [1] P. Heas & C. Herzet. Low-rank dynamic mode decomposition: optimal
    solution in polynomial time. arXiv:1610.02962. 2016.

    """

    def __init__(self,
                 factorization="evd",
                 svd_rank=-1,
                 tlsq_rank=0,
                 exact=False,
                 opt=False
                 ):

        # --> Initialize the super-seeded class
        super().__init__()

        # --> Compute either the SVD or EVD factorization of the DMD problem.
        self.factorization = factorization

        # --> Rank of the DMD model.
        self.svd_rank = svd_rank

        # --> Singular values of the DMD operator.
        self._svds = None

        # --> DMD basis for the input space.
        self._input_space = None

        # --> DMD basis for the output space.
        self._output_space = None

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
        self : OptDMD object
            The fitted DMD model.

        """

        # --> Check whether Y is given or not.
        if Y is None:
            # --> Assume X includes a single realization of the dynamic process.
            Y = X[:, 1:]    # y = x[k+1]
            X = X[:, :-1]   # x = x[k]

        # --> Total Least-Squares Denoising of the snapshots matrices.
        X, Y = self._compute_tlsq(X, Y, self.tlsq_rank)

        # --> Compute the singular value decomposition of X.
        Ux, Sx, Vx = self._compute_svd(X, -1)

        # --> Compute the matrix Z.
        Z = np.linalg.multi_dot([Y, Vx, np.diag(Sx), pinv_diag(Sx), Vx.T])

        # --> Compute the singular value decomposition of Z.
        Uz, _, _ = self._compute_svd(Z, self.svd_rank)

        # --> Compute the Q matrix.
        Q = np.linalg.multi_dot([Uz.T, Y, Vx, pinv_diag(Sx), Ux.T]).T

        # --> Low-dimensional DMD operator.
        self._Atilde = self._build_lowrank_op(Uz, Q)

        # --> Eigenvalues of Atilde.
        self._eigs = eigvals(self._Atilde)

        # --> Singular values of Atilde.
        self._svds = svdvals(self._Atilde)

        if self.factorization == "svd":
            # --> DMD basis for the input space.
            self._input_space = Q

            # --> DMD basis for the output space.
            self._output_space = Uz

            # --> DMD-SVD modes.
            self._modes = Uz

        elif self.factorization == "evd":
            # --> Compute DMD eigenvalues and right/left eigenvectors
            _, self._input_space, self._output_space = self._eig_from_lowrank_op(self._Atilde, Uz, Q)

            # --> DMD-EVD modes.
            self._modes = self._output_space

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
            One time-step ahead predicted output.

        """

        # --> Factorization : SVD.
        if self.factorization == "svd":
            Y = np.linalg.multi_dot(
                [self._output_space, self._input_space.T.conj(), X]
            )
        # --> Factorization : EVD.
        elif self.factorization == "evd":
            Y = np.linalg.multi_dot(
                [self._output_space, np.diag(self._eigs),
                 self._input_space.T.conj(), X]
            )

        return Y

    @staticmethod
    def _build_lowrank_op(P, Q):
        """
        Utility function to build the low-dimension DMD operator.

        Parameters
        ----------
        P : array-like, shape (m, n)
            SVD-DMD basis for the output space.
        Q : array-like, shape (m, n)
            SVD-DMD basis for the input space.

        Returns
        -------
        array-like, shape (n , n)
            Low-dimensional DMD operator.

        """
        return Q.T.dot(P)

    @staticmethod
    def _eig_from_lowrank_op(Atilde, P, Q):
        """
        Utility function to compute the eigenvalues of the low-dimensional
        DMD operator and the high-dimensional left and right eigenvectors.

        Parameters
        ----------
        Atilde : array-like, shape (n, n)
            Low-dimensional DMD operator.
        P : array-like, shape (m, n)
            Right DMD-SVD vectors.
        Q : array-like, shape (m, n)
            Left DMD-SVD vectors.

        Returns
        -------
        vals : array-like, shape (n,)
            Eigenvalues of the low-dimensional DMD operator.

        left_vecs : array-like, shape (m, n)
            Left eigenvectors of the DMD operator.

        right_vecs : array-like, shape (m, n)
            Right eigenvectors of the DMD operator.

        """

        # --> Compute the eigentriplets of the low-dimensional DMD matrix.
        vals, vecs_left, vecs_right = eig(Atilde, left=True, right=True)

        # --> Build the matrix of right eigenvectors.
        right_vecs = np.linalg.multi_dot([P, Atilde, vecs_right])
        #right_vecs[:, abs(vals) > 1e-8] /= vals[abs(vals) > 1e-8]
        right_vecs = right_vecs.dot(pinv_diag(vals))

        # --> Build the matrix of left eigenvectors.
        left_vecs = Q.dot(vecs_left)
        #left_vecs[:, abs(vals) > 1e-8] /= vals[abs(vals) > 1e-8]
        left_vecs = left_vecs.dot(pinv_diag(vals))

        # --> Rescale the left eigenvectors.
        m = np.diag(left_vecs.T.conj().dot(right_vecs))
        #left_vecs[:, abs(vals) > 1e-8] /= m[abs(vals) > 1e-8].conj()
        left_vecs = left_vecs.dot(pinv_diag(m))

        return vals, left_vecs, right_vecs

    def _compute_amplitudes(self, modes, snapshots, eigs, opt):
        """Short summary.

        Parameters
        ----------
        modes : type
            Description of parameter `modes`.
        snapshots : type
            Description of parameter `snapshots`.
        eigs : type
            Description of parameter `eigs`.
        opt : type
            Description of parameter `opt`.

        Returns
        -------
        type
            Description of returned object.

        """
        raise NotImplementedError("This function has not been implemented yet.")


    @property
    def dynamics(self):
        """Short summary.

        Parameters
        ----------


        Returns
        -------
        type
            Description of returned object.

        """
        raise NotImplementedError("This function has not been implemented yet.")
