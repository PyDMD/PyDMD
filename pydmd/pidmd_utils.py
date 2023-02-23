"""
PiDMD utilities module.

References:
- Peter J. Baddoo, Benjamin Herrmann, Beverley J. McKeon, J. Nathan Kutz, and
Steven L. Brunton. Physics-informed dynamic mode decomposition (pidmd). 2021.
arXiv:2112.04307.
"""
import numpy as np
from numpy.fft import fft, ifft, fft2
from scipy import sparse
from scipy.linalg import block_diag, rq

from .utils import compute_svd
from .rdmd import compute_rank

def compute_unitary(X, Y, svd_rank):
    """
    Given the data matrices X and Y and the rank truncation svd_rank, solves
    for the best-fit unitary operator A that solves the relationship Y = AX.
    Returns a dictionary containing the corresponding reduced operator atilde.
    """
    Ux = compute_svd(X, svd_rank)[0]
    Yproj = Ux.conj().T.dot(Y)
    Xproj = Ux.conj().T.dot(X)
    Uyx, _, Vyx = compute_svd(Yproj.dot(Xproj.conj().T), -1)
    atilde = Uyx.dot(Vyx.conj().T)
    return {"atilde": atilde}

def compute_uppertriangular(X, Y):
    """
    Given the data matrices X and Y, solves for the best-fit
    uppertriangular matrix A that solves the relationship Y = AX.
    Returns a dictionary containing A.
    """
    R, Q = rq(X, mode="economic")
    Ut = np.triu(Y.dot(Q.conj().T))
    A = np.linalg.lstsq(R.T, Ut.T, rcond=None)[0].T
    return {"A": A}

def compute_diagonal(X, Y, svd_rank, manifold_opt, compute_A):
    """
    Given the data matrices X and Y and the rank truncation svd_rank, solves
    for the best-fit matrix A that solves the relationship Y = AX and has
    diagonal entries specified by manifold_opt. Only the eigenvalues and
    eigenvectors of A are computed if the compute_A flag is not True.
    Returns a dictionary of computed values.
    """
    # Specify the index matrix for the diagonals of A.
    nx = len(X)
    if manifold_opt is None:
        ind_mat = np.ones((nx, 2), dtype=int)
    elif isinstance(manifold_opt, int) and manifold_opt > 0:
        ind_mat = manifold_opt * np.ones((nx, 2), dtype=int)
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
    nxs = np.arange(nx)
    l1s = (nxs - ind_mat[:, 0] + 1).clip(min=0).astype(int)
    l2s = (nxs + ind_mat[:, 1]).clip(max=nx).astype(int)
    for j in range(nx):
        l1, l2 = l1s[j], l2s[j]
        C = X[l1:l2].T
        b = Y[j].T
        I.append(j * np.ones(l2 - l1))
        J.append(np.arange(l1, l2))
        R.append(np.linalg.lstsq(C, b, rcond=None)[0].T)

    # Build A as a sparse matrix.
    A_sparse = sparse.coo_matrix(
        (np.hstack(R), (np.hstack(I), np.hstack(J))), shape=(nx,nx))
    if compute_A:
        return {"A": A_sparse.toarray()}
    r = compute_rank(X, svd_rank)
    eigenvalues, modes = sparse.linalg.eigs(A_sparse, k=r)
    return {"eigenvalues": eigenvalues, "modes": modes}

def compute_symmetric(X, Y, svd_rank, skew_symmetric=False):
    """
    Given the data matrices X and Y and the rank truncation svd_rank, solves
    for the best-fit symmetric (or skew-symmetric) operator A that solves the
    relationship Y = AX. Returns a dictionary containing the corresponding
    reduced operator atilde.
    """
    U, s, V = compute_svd(X, -1)
    C = np.linalg.multi_dot([U.conj().T, Y, V])
    r = compute_rank(X, svd_rank)
    if skew_symmetric:
        atilde = 1j * np.diag(np.diagonal(C).imag / s)
        for i in range(r):
            for j in range(i + 1, r):
                atilde[i, j] = -s[i] * np.conj(C[j,i]) + s[j] * C[i,j]
                atilde[i, j] /= (s[i] ** 2 + s[j] ** 2)
        atilde += -atilde.conj().T - 1j * np.diag(np.diag(atilde.imag))
    else: # symmetric
        atilde = np.diag(np.diagonal(C).real / s)
        for i in range(r):
            for j in range(i + 1, r):
                atilde[i, j] = s[i] * np.conj(C[j,i]) + s[j] * C[i,j]
                atilde[i, j] /= (s[i] ** 2 + s[j] ** 2)
        atilde += atilde.conj().T - np.diag(np.diag(atilde.real))
    return {"atilde": atilde}

def compute_toeplitz(X, Y, flipped=False):
    """
    Given the data matrices X and Y, solves for the best-fit toeplitz operator
    A (or hankel operator if flipped = True) that solves the relationship
    Y = AX. Returns a dictionary containing A.
    """
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
    eigenvalues = np.linalg.lstsq(L[:-1, :-1].T, y[:-1], rcond=None)[0]
    eigenvalues = np.append(eigenvalues, 0)

    # Convert eigenvalues into circulant matrix.
    new_A = ifft(fft(np.diag(eigenvalues), axis=0).T, axis=0).T

    # Extract toeplitz matrix from the circulant matrix.
    A = new_A[:nx, :nx].dot(J)

    return {"A": A}

def compute_circulant(X, Y, circulant_opt, svd_rank, compute_A):
    """
    Given the data matrices X and Y and the rank truncation svd_rank, solves
    for the best-fit circulant matrix A that solves the relationship Y = AX and
    satisfies any additional conditions set by circulant_opt. Only the
    eigenvalues and eigenvectors of A are computed if the compute_A
    flag is not True. Returns a dictionary of computed values.
    """
    nx = len(X)
    fX = fft(X, axis=0)
    fY = fft(Y.conj(), axis=0)
    fX_norm = np.linalg.norm(fX, axis=1)
    eigenvalues = np.divide(np.diag(fX.dot(fY.conj().T)), fX_norm ** 2)

    if circulant_opt == "unitary":
        eigenvalues = np.exp(1j * np.angle(eigenvalues))
    elif circulant_opt == "symmetric":
        eigenvalues = eigenvalues.real
    elif circulant_opt == "skewsymmetric":
        eigenvalues = 1j * eigenvalues.imag

    # Remove the least important eigenvalues.
    r = compute_rank(X, svd_rank)
    res = np.divide(np.diag(np.abs(fX.dot(fY.conj().T))),
                    np.linalg.norm(fX.conj().T, 2, axis=0).T)
    ind_exclude = np.argpartition(res, nx-r)[:nx-r]
    eigenvalues[ind_exclude] = 0

    if compute_A:
        A = fft(np.multiply(eigenvalues, ifft(np.eye(nx), axis=0).T).T, axis=0)
        return {"A": A}

    eigenvalues = np.delete(eigenvalues, ind_exclude)
    modes = np.delete(fft(np.eye(nx), axis=0), ind_exclude, axis=1)
    return {"eigenvalues": eigenvalues, "modes": modes}

def compute_symtridiagonal(X, Y, svd_rank, compute_A):
    """
    Given the data matrices X and Y and the rank truncation svd_rank, solves
    for the best-fit symmetric tridiagonal matrix A that solves Y = AX.
    Only the eigenvalues and eigenvectors of A are computed if the compute_A
    flag is not True. Returns a dictionary of computed values.
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
    T = sparse.vstack([sparse.hstack([T1, T2]),
                       sparse.hstack([T2.conj().T, T3])])

    # Solve for c in the system Tc = d.
    d = np.concatenate(
        (np.diag(X.dot(Y.T)),
         np.diag(X[:-1].dot(Y[1:].T)) + np.diag(X[1:].dot(Y[:-1].T))),
         axis=None
    )
    c = sparse.linalg.lsqr(T.real, d.real)[0]

    # Form the solution matrix A.
    A_sparse = sparse.diags(c[:nx]) \
        + sparse.spdiags(np.insert(c[nx:], 0, 0), diags=1, m=nx, n=nx) \
        + sparse.spdiags(np.append(c[nx:], 0), diags=-1, m=nx, n=nx)

    if compute_A:
        return {"A": A_sparse.toarray()}

    r = compute_rank(X, svd_rank)
    eigenvalues, modes = sparse.linalg.eigs(A_sparse, k=r)
    return {"eigenvalues": eigenvalues, "modes": modes}

def compute_BCCB(X, Y, block_shape, bccb_opt, svd_rank):
    """
    Given the data matrices X and Y and the rank truncation svd_rank, solves
    for the best-fit BCCB (block circulant with circulant blocks) matrix A that
    solves the relationship Y = AX. The blocks of A will have the shape
    block_shape and A will have the additional matrix property given by
    bccb_opt. Returns a dictionary containing A.
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
    r = compute_rank(X, svd_rank)
    res = np.divide(np.diag(np.abs(fX.dot(fY.conj().T))),
                    np.linalg.norm(fX.conj().T, 2, axis=0).T)
    d[np.argpartition(res, nx-r)[:nx-r]] = 0
    A = fft_block2d(np.multiply(d.conj(), fft_block2d(np.eye(nx)).conj().T).T)
    return {"A": A}

def compute_BC(X, Y, block_shape, tridiagonal_blocks=False):
    """
    Given the data matrices X and Y, solves for the best-fit block circulant
    matrix A that solves the relationship Y = AX. The blocks of A will have the
    shape block_shape and will be tridiagonal if tridiagonal_blocks is True.
    Returns a dictionary containing A.
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
            d.append(compute_diagonal(fX[ls], fY[ls], svd_rank=-1,
                                      manifold_opt=2, compute_A=True)["A"])
        else:
            d.append(np.linalg.lstsq(fX[ls].T, fY[ls].T, rcond=None)[0].T)

    # Update self._A to be the full block circulant matrix.
    BD = block_diag(*d)
    fI = fft_block(np.eye(nx))
    A = fft_block(BD.dot(fI).conj()).conj()
    return {"A": A}
