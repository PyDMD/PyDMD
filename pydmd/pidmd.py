"""
Derived module from dmd.py for Physics-informed DMD.

References:
- Peter J. Baddoo, Benjamin Herrmann, Beverley J. McKeon, J. Nathan Kutz, and
Steven L. Brunton. Physics-informed dynamic mode decomposition (pidmd). 2021.
arXiv:2112.04307.
"""
import numpy as np
from numpy.fft import fft, ifft, fft2
from scipy import sparse
from scipy.linalg import block_diag, rq

from .dmd import DMD
from .dmdoperator import DMDOperator
from .utils import compute_svd
from .rdmd import compute_rank


class PiDMDOperator(DMDOperator):
    """
    DMD operator for Physics-informed DMD.

    :param manifold: the matrix manifold for the full DMD operator A.
    :type manifold: str
    :param manifold_opt: option used to specify certain manifolds. If manifold
        is "diagonal", manifold_opt may be used to specify the width of the
        diagonal of A. If manifold starts with "BC", manifold_opt must be a
        2D tuple that specifies the desired dimensions of the blocks of A.
    :type manifold_opt: int, tuple(int,int), or numpy.ndarray
    :param compute_A: Flag that determines whether or not to compute the full
        Koopman operator A.
    :type compute_A: bool
    :param svd_rank: the rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive integer, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation.
    :type svd_rank: int or float
    """
    def __init__(
        self,
        manifold,
        manifold_opt,
        compute_A,
        svd_rank,
    ):
        self._manifold = manifold
        self._manifold_opt = manifold_opt
        self._compute_A = compute_A
        self._svd_rank = svd_rank

        self._A = None
        self._Atilde = None
        self._eigenvalues = None
        self._eigenvectors = None
        self._modes = None


    @property
    def A(self):
        """
        Get the full Koopman operator A.

        :return: the full Koopman operator A.
        :rtype: numpy.ndarray
        """
        if not self._compute_A:
            msg = "A not computed during fit. " \
                  "Set parameter compute_A = True to compute A."
            raise ValueError(msg)
        if self._A is None:
            raise ValueError("You need to call fit before")
        return self._A


    def _check_compute_A(self):
        """
        Helper method that checks that compute_A is True.
        Throws an error if compute_A is False.
        """
        if not self._compute_A:
            raise ValueError(
                "A must be computed for the chosen manifold."
                "Set compute_A = True to compute A."
            )


    def _compute_unitary(self, X, Y):
        """
        Given the data matrices X and Y, solves for the best-fit unitary
        operator A that solves the relationship Y=AX. Stores the corresponding
        reduced operator atilde.
        """
        Ux = compute_svd(X, self._svd_rank)[0]
        Yproj = Ux.conj().T.dot(Y)
        Xproj = Ux.conj().T.dot(X)
        Uyx, _, Vyx = compute_svd(Yproj.dot(Xproj.conj().T), -1)
        self._Atilde = Uyx.dot(Vyx.conj().T)


    def _compute_uppertriangular(self, X, Y):
        """
        Given the data matrices X and Y, solves for and stores the best-fit
        uppertriangular matrix A that solves the relationship Y=AX.
        """
        self._check_compute_A()
        R, Q = rq(X, mode="economic")
        Ut = np.triu(Y.dot(Q.conj().T))
        self._A = np.linalg.lstsq(R.T, Ut.T, rcond=None)[0].T


    def _compute_diagonal(self, X, Y, manifold_opt):
        """
        Given the data matrices X and Y and specification for the diagonals of
        the matrix A, solves for and stores the best-fit matrix A that solves
        the relationship Y=AX and has diagonal entries specified by
        manifold_opt. Only the eigenvalues and eigenvectors of A are stored
        if the compute_A flag is not True.
        """
        # Specify the index matrix for the diagonals of A.
        nx = len(X)
        if manifold_opt is None:
            ind_mat = np.ones((nx, 2))
        elif isinstance(manifold_opt, int) and manifold_opt > 0:
            ind_mat = manifold_opt * np.ones((nx, 2))
        elif (isinstance(manifold_opt, tuple)
                and len(manifold_opt) == 2
                and np.all(np.array(manifold_opt) > 0)):
            ind_mat = np.ones((nx, 2))
            ind_mat[:, 0] *= manifold_opt[0]
            ind_mat[:, 1] *= manifold_opt[1]
        elif (isinstance(manifold_opt, np.ndarray)
                and manifold_opt.shape == (nx, 2)
                and np.all(manifold_opt > 0)):
            ind_mat = manifold_opt
        else:
            raise ValueError("manifold_opt is not in an allowable format.")

        # Keep track of info for building A as a sparse coordinate matrix.
        I = []
        J = []
        R = []

        # Solve min||Cx-b|| along each row.
        for j in range(nx):
            l1 = int(max(j-(ind_mat[j,0]-1), 0))
            l2 = int(min(j+(ind_mat[j,1]-1), nx-1) + 1)
            C = X[l1:l2].T
            b = Y[j].T
            I.append(j * np.ones(l2 - l1))
            J.append(np.arange(l1, l2))
            R.append(np.linalg.lstsq(C, b, rcond=None)[0].T)

        # Build A as a sparse matrix.
        A_sparse = sparse.coo_array(
            (np.hstack(R), (np.hstack(I), np.hstack(J))), shape=(nx,nx)
        )
        if self._compute_A:
            self._A = A_sparse.toarray()
        else:
            self._eigenvalues, self._modes = sparse.linalg.eigs(
                A_sparse, k=compute_rank(X, self._svd_rank)
            )


    def _compute_symmetric(self, X, Y, skewsymmetric=False):
        """
        Given the data matrices X and Y, solves for the best-fit symmetric
        (or skewsymmetric) operator A that solves the relationship Y=AX.
        Stores the corresponding reduced operator atilde.
        """
        U, s, V = compute_svd(X, -1)
        C = np.linalg.multi_dot([U.conj().T, Y, V])
        r = compute_rank(X, self._svd_rank)
        atilde = np.zeros((r, r), dtype="complex")

        if skewsymmetric:
            for i in range(r):
                atilde[i, i] = 1j * (C[i, i].imag / s[i])
                for j in range(i + 1, r):
                    atilde[i, j] = -s[i] * np.conj(C[j,i]) + s[j] * C[i,j]
                    atilde[i, j] /= (s[i] ** 2 + s[j] ** 2)
            atilde += -atilde.conj().T - 1j * np.diag(np.diag(atilde.imag))
        else: # symmetric
            for i in range(r):
                atilde[i, i] = C[i, i].real / s[i]
                for j in range(i + 1, r):
                    atilde[i, j] = s[i] * np.conj(C[j,i]) + s[j] * C[i,j]
                    atilde[i, j] /= (s[i] ** 2 + s[j] ** 2)
            atilde += atilde.conj().T - np.diag(np.diag(atilde.real))

        self._Atilde = atilde


    def _compute_toeplitz(self, X, Y, flipped=False):
        """
        Given the data matrices X and Y, solves for and stores the
        best-fit toeplitz (or hankel if flipped=True) operator A that
        solves the relationship Y=AX.
        """
        self._check_compute_A()

        nx, nt = X.shape
        if flipped: # hankel
            J = np.fliplr(np.eye(nx))
        else: # toeplitz
            J = np.eye(nx)

        # Define left and right matrices.
        Am = fft(np.hstack([np.eye(nx), np.zeros((nx, nx))]).T, axis=0)
        Am = Am.conj().T / np.sqrt(2 * nx)
        B = fft(np.hstack([J.dot(X).conj().T, np.zeros((nt, nx))]).T, axis=0)
        B = B.conj().T / np.sqrt(2 * nx)

        # Compute AA* and B*B (fast computation of AA*).
        AAt = ifft(fft(np.vstack([
            np.hstack([np.eye(nx), np.zeros((nx, nx))]),
            np.zeros((nx, 2 * nx))
        ]), axis=0).T, axis=0).T

        # Solve linear system y = dL.
        y = np.diag(np.linalg.multi_dot([Am.conj().T, Y.conj(), B])).conj().T
        L = np.multiply(AAt, B.conj().T.dot(B).T).conj().T
        d = np.linalg.lstsq(L[:-1, :-1].T, y[:-1], rcond=None)[0]
        d = np.append(d, 0)

        # Convert eigenvalues into circulant matrix.
        new_A = ifft(fft(np.diag(d), axis=0).T, axis=0).T

        # Extract toeplitz matrix from the circulant matrix.
        self._A = new_A[:nx, :nx].dot(J)


    def _compute_circulant(self, X, Y, circulant_opt):
        """
        Given the data matrices X and Y, solves for and stores the best-fit
        circulant matrix A that solves the relationship Y=AX and satisfies any
        additional conditions set by circulant_opt. Only the eigenvalues and
        eigenvectors of A are stored if the compute_A flag is not True.
        """
        nx = len(X)
        fX = fft(X, axis=0)
        fY = fft(Y.conj(), axis=0)
        fX_norm = np.linalg.norm(fX, axis=1)
        d = np.divide(np.diag(fX.dot(fY.conj().T)), fX_norm ** 2)

        if circulant_opt == "unitary":
            d = np.exp(1j * np.angle(d))
        elif circulant_opt == "symmetric":
            d = d.real
        elif circulant_opt == "skewsymmetric":
            d = 1j * d.imag

        # Remove the least important eigenvalues.
        r = compute_rank(X, self._svd_rank)
        res = np.divide(np.diag(np.abs(fX.dot(fY.conj().T))),
                        np.linalg.norm(fX.conj().T, 2, axis=0).T)
        ind_exclude = np.argpartition(res, nx-r)[:nx-r]
        d[ind_exclude] = 0

        if self._compute_A:
            self._A = fft(np.multiply(d, ifft(np.eye(nx), axis=0).T).T, axis=0)
        else:
            self._eigenvalues = np.delete(d, ind_exclude)
            self._modes = np.delete(fft(np.eye(nx),axis=0),ind_exclude, axis=1)


    def _compute_symtridiagonal(self, X, Y):
        """
        Given the data matrices X and Y, solves for and stores the best-fit
        symmetric tridiagonal matrix A that solves the relationship Y=AX.
        Only the eigenvalues and eigenvectors of A are stored if the
        compute_A flag is not True.
        """
        # Form the leading block.
        nx = len(X)
        T1e = np.linalg.norm(X, axis=1) ** 2
        T1 = sparse.diags(T1e)

        # Form the second and third blocks.
        T2e = np.diag(X[1:].dot(X[:-1].T))
        T2 = sparse.spdiags([T2e, T2e], diags=[-1, 0], m=nx, n=nx-1)

        # Form the final block.
        T3e = np.insert(np.diag(X[2:].dot(X[:-2].T)), 0, 0)
        T3_offdiag = sparse.spdiags(T3e, diags=1, m=nx-1, n=nx-1)
        T3 = sparse.spdiags(T1e[:-1] + T1e[1:], diags=0, m=nx-1, n=nx-1) \
                + T3_offdiag + T3_offdiag.conj().T

        # Form the symmetric block-tridiagonal matrix T = [T1  T2; T2* T3].
        T = sparse.vstack(
            [sparse.hstack([T1, T2]), sparse.hstack([T2.conj().T, T3])]
        )

        # Solve for c in the system Tc = d.
        d = np.concatenate(
            (np.diag(X.dot(Y.T)),
             np.diag(X[:-1].dot(Y[1:].T)) + np.diag(X[1:].dot(Y[:-1].T))),
             axis=None
        )
        c = sparse.linalg.lsqr(T.real, d.real)[0]

        # Form the solution matrix A.
        A_sparse = sparse.diags(c[:nx])
        A_sparse += sparse.spdiags(np.insert(c[nx:],0,0), diags=1, m=nx, n=nx)
        A_sparse += sparse.spdiags(np.append(c[nx:],0), diags=-1, m=nx, n=nx)

        if self._compute_A:
            self._A = A_sparse.toarray()
        else:
            self._eigenvalues, self._modes = sparse.linalg.eigs(
                A_sparse, k=compute_rank(X, self._svd_rank)
            )


    def _compute_BCCB(self, X, Y, block_shape, bccb_opt):
        """
        Given the data matrices X and Y, solves for and stores the best-fit
        BCCB (block circulant with circulant blocks) matrix A that solves the
        relationship Y=AX. The blocks of A will have the shape block_shape and
        will have the additional matrix property given by bccb_opt.
        """
        def fft_block2d(X):
            """
            Helper method that, given a 2D numpy array X, reshapes the
            columns of X into arrays of shape block_shape, computes the
            2D discrete Fourier Transform, and returns the resulting
            FFT matrix restored to the original X shape and scaled
            according to the length of the columns of X.
            """
            m, n = X.shape
            Y = X.reshape(*block_shape, n, order="F")
            Y_fft2 = fft2(Y, axes=(0, 1))
            return Y_fft2.reshape(m, n, order="F") / np.sqrt(m)

        nx = len(X)
        fX = fft_block2d(X.conj())
        fY = fft_block2d(Y.conj())
        d = np.zeros(nx, dtype="complex")

        if not bccb_opt:
            for j in range(nx):
                d[j] = np.dot(fX[j], fY[j].conj()).conj()
                d[j] /= np.linalg.norm(fX[j].conj(), 2)**2
        else:
            for j in range(nx):
                dp = np.linalg.lstsq(
                    fX[j,:,None], fY[j,:,None], rcond=None)[0][0][0]
                if bccb_opt == "unitary":
                    d[j] = np.exp(1j * np.angle(dp))
                elif bccb_opt == "symmetric":
                    d[j] = dp.real
                else: # skewsymmetric
                    d[j] = 1j * dp.imag

        # Remove the least important eigenvalues.
        r = compute_rank(X, self._svd_rank)
        res = np.divide(np.diag(np.abs(fX.dot(fY.conj().T))),
                        np.linalg.norm(fX.conj().T, 2, axis=0).T)
        d[np.argpartition(res, nx-r)[:nx-r]] = 0
        self._A = fft_block2d(
            np.multiply(d.conj(), fft_block2d(np.eye(nx)).conj().T).T
        )


    def _compute_BC(self, X, Y, block_shape, tridiagonal_blocks=False):
        """
        Given the data matrices X and Y, solves for and stores the best-fit
        BC (block circulant) matrix A that solves the relationship Y=AX. The
        blocks of A will have the shape block_shape and will be tridiagonal
        if tridiagonal_blocks is True.
        """
        def fft_block(X):
            """
            Helper method that, given a 2D numpy array X, reshapes the
            columns of X into arrays of shape block_shape, computes the
            1D discrete Fourier Transform, and returns the resulting
            FFT matrix restored to the original X shape and scaled
            according to the number of columns per block.
            """
            m, n = X.shape
            Y = X.reshape(*block_shape, n, order="F")
            Y_fft = fft(Y, axis=1)
            return Y_fft.reshape(m, n, order="F") / np.sqrt(block_shape[1])

        nx = len(X)
        fX = fft_block(X)
        fY = fft_block(Y)
        d = []

        for j in range(block_shape[1]):
            ls = (j * block_shape[0]) + np.arange(block_shape[0])
            if tridiagonal_blocks:
                # Updates self._A to be the current tridiagonal block.
                self._compute_diagonal(fX[ls], fY[ls], 2)
                d.append(np.copy(self._A))
            else:
                d.append(np.linalg.lstsq(fX[ls].T, fY[ls].T, rcond=None)[0].T)

        # Update self._A to be the full block circulant matrix.
        BD = block_diag(*d)
        fI = fft_block(np.eye(nx))
        self._A = fft_block(BD.dot(fI).conj()).conj()


    def _compute_procrustes(self, X, Y, manifold, manifold_opt=None):
        """
        Private method that computes the best-fit linear operator A in the
        relationship Y=AX such that A is restricted to the family of matrices
        defined by the given manifold (and manifold option if applicable).
        Computes and stores either...
        (1) the reduced operator atilde,
        (2) the full operator A, or
        (3) the eigenvalues and eigenvectors of A,
        depending on the chosen manifold and the compute_A parameter.
        """
        if manifold == "unitary":
            self._compute_unitary(X, Y)
        elif manifold == "uppertriangular":
            self._compute_uppertriangular(X, Y)
        elif manifold == "lowertriangular":
            self._compute_uppertriangular(np.flipud(X), np.flipud(Y))
            self._A = np.rot90(self._A, 2)
        elif manifold == "diagonal":
            self._compute_diagonal(X, Y, manifold_opt)
        elif manifold == "symmetric":
            self._compute_symmetric(X, Y)
        elif manifold == "skewsymmetric":
            self._compute_symmetric(X, Y, skewsymmetric=True)
        elif manifold == "toeplitz":
            self._compute_toeplitz(X, Y)
        elif manifold == "hankel":
            self._compute_toeplitz(X, Y, flipped=True)
        elif manifold in [
            "circulant",
            "circulant_unitary",
            "circulant_symmetric",
            "circulant_skewsymmetric"
        ]:
            circulant_opt = manifold.replace("circulant_", "")
            self._compute_circulant(X, Y, circulant_opt)
        elif manifold == "symmetric_tridiagonal":
            self._compute_symtridiagonal(X, Y)
        elif manifold in [
            "BC",
            "BCTB",
            "BCCB",
            "BCCBunitary",
            "BCCBsymmetric",
            "BCCBskewsymmetric",
        ]:
            self._check_compute_A()
            # Specify the shape of the blocks in the output matrix A.
            if manifold_opt is None:
                raise ValueError("manifold_opt must be specified.")
            if (not isinstance(manifold_opt,tuple) or len(manifold_opt) != 2):
                raise ValueError("manifold_opt is not in an allowable format.")
            block_shape = np.array(manifold_opt)

            if manifold.startswith("BCCB"):
                bccb_opt = manifold.replace("BCCB", "")
                self._compute_BCCB(X, Y, block_shape, bccb_opt)
            elif manifold == "BC":
                self._compute_BC(X, Y, block_shape)
            else:
                self._compute_BC(X, Y, block_shape, tridiagonal_blocks=True)
        else:
            raise ValueError("The selected manifold is not implemented.")


    def compute_operator(self, X, Y):
        """
        Compute and store the low-rank operator and the full operator A.

        :param X: matrix containing the snapshots x0,..x{n-1} by column.
        :type X: numpy.ndarray
        :param Y: matrix containing the snapshots x1,..x{n} by column.
        :type Y: numpy.ndarray
        :return: the (truncated) left-singular vectors matrix, the (truncated)
            singular values array, and the (truncated) right-singular vectors
            matrix of X.
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
        """
        U, s, V = compute_svd(X, self._svd_rank)

        # Compute the corresponding Procrustes problem.
        self._compute_procrustes(X, Y, self._manifold, self._manifold_opt)

        # Case 1: atilde was computed.
        if self._Atilde is not None:
            self._eigenvalues, self._eigenvectors = np.linalg.eig(self._Atilde)
            self._modes = U.dot(self._eigenvectors)
            if self._compute_A:
                self._A = np.linalg.multi_dot([self._modes,
                                               np.diag(self._eigenvalues),
                                               np.linalg.pinv(self._modes)])
        else:
            # Case 2: A was computed.
            if self._eigenvalues is None or self._modes is None:
                self._eigenvalues, self._modes = np.linalg.eig(self._A)
            # Case 3: eigenvalues and modes were computed.
            self._eigenvectors = U.conj().T.dot(self._modes)
            self._Atilde = np.linalg.multi_dot(
                [self._eigenvectors,
                 np.diag(self._eigenvalues),
                 np.linalg.pinv(self._eigenvectors)]
            )

        return U, s, V


class PiDMD(DMD):
    """
    Physics-informed Dynamic Mode Decomposition.

    :param manifold: the matrix manifold to restrict the full operator A to.
        The following matrix manifolds are permissible:
        - "unitary"
        - "uppertriangular"
        - "lowertriangular"
        - "diagonal"
        - "symmetric",
        - "skewsymmetric"
        - "toeplitz"
        - "hankel"
        - "circulant"
        - "circulant_unitary"
        - "circulant_symmetric"
        - "circulant_skewsymmetric"
        - "symmetric_tridiagonal"
        - "BC" (block circulant)
        - "BCTB" (BC with tridiagonal blocks)
        - "BCCB" (BC with circulant blocks)
        - "BCCBunitary" (BCCB and unitary)
        - "BCCBsymmetric" (BCCB and symmetric)
        - "BCCBskewsymmetric" (BCCB and skewsymmetric)
    :type manifold: str
    :param manifold_opt: option used to specify certain manifolds.
        If manifold is "diagonal", manifold_opt may be used to specify the
        width of the diagonal of A. If manifold_opt is an integer k, A is
        banded, with a lower and upper bandwidth of k-1. If manifold_opt is
        a tuple containing two integers k1 and k2, A is banded with a lower
        bandwidth of k1-1 and an upper bandwidth of k2-1. Finally, if
        manifold_opt is a numpy array of size (len(X), 2), the entries of
        manifold_opt are used to explicitly define the the upper and lower
        bounds of the indices of the non-zero elements of A.
        If manifold starts with "BC", manifold_opt must be a 2D tuple that
        specifies the desired dimensions of the blocks of A.
        Note that all other manifolds do not use manifold_opt.
    :type manifold_opt: int, tuple(int,int), or numpy.ndarray
    :param compute_A: Flag that determines whether or not to compute the full
        Koopman operator A.
    :type compute_A: bool
    :param svd_rank: the rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive integer, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation. Default is -1, meaning no truncation.
    :type svd_rank: int or float
    :param tlsq_rank: rank truncation computing Total Least Square. Default is
        0, meaning no truncation.
    :type tlsq_rank: int
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
    """
    def __init__(
        self,
        manifold,
        manifold_opt=None,
        compute_A=False,
        svd_rank=-1,
        tlsq_rank=0,
        opt=False,
    ):
        super().__init__(
            svd_rank=svd_rank,
            tlsq_rank=tlsq_rank,
            opt=opt,
        )
        self._Atilde = PiDMDOperator(
            manifold=manifold,
            manifold_opt=manifold_opt,
            compute_A=compute_A,
            svd_rank=svd_rank,
        )

    @property
    def A(self):
        """
        Get the full Koopman operator A.

        :return: the full Koopman operator A.
        :rtype: numpy.ndarray
        """
        return self.operator.A
