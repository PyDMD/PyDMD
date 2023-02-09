import numpy as np
from pytest import raises
from numpy.testing import assert_allclose
from pydmd.pidmd import PiDMD

def error(true, comp):
    """Helper function that computes and returns relative error."""
    return np.linalg.norm(comp - true) / np.linalg.norm(true)

# Use random matrix as dummy data.
rng = np.random.default_rng(seed=42)
X = rng.standard_normal((25, 1000))
X1 = X[:, :-1]
X2 = X[:, 1:]

# Compute A via pseudoinverse as benchmark.
A_dmd = X2.dot(np.linalg.pinv(X1))
dmd_error = error(X2, A_dmd.dot(X1))
error_tol = 0.45 * dmd_error

def assert_accurate(A):
    """
    Helper method that, given a computed A matrix, tests that A is an
    accurate operator relative to that obtained via standard DMD.
    """
    computed_error = error(X2, A.dot(X1))
    assert_allclose(computed_error, dmd_error, atol=error_tol)

def assert_all_zero(A):
    """
    Helper method that, given a matrix A, tests that A is
    approximately a matrix of only zero entries.
    """
    assert_allclose(np.linalg.norm(A), 0, atol=5*1e-15)

def assert_circulant(A):
    """
    Helper method that, given a matrix A, tests that A is a circulant matrix.
    """
    for i in range(1, len(A)):
        assert_allclose(np.roll(A[i, :], -i), A[0, :], atol=5*1e-15)

def assert_block_circulant(A):
    """
    Helper method that, given a matrix A, tests that A is block circulant.
    Assumes A is a square matrix composed of equally-sized square blocks.
    """
    block_n = int(np.sqrt(len(A)))
    for i in range(1, block_n):
        block_row_i = A[i*block_n:(i+1)*block_n, :]
        rolled_block_row_i = np.roll(block_row_i, -i*block_n, axis=1)
        assert_allclose(rolled_block_row_i, A[:block_n, :], atol=5*1e-15)

def test_invalid_manifold():
    # Test that an error is thrown if an invalid manifold is given.
    with raises(ValueError):
        PiDMD("some_manifold").fit(X)

def test_unitary():
    pidmd = PiDMD("unitary", compute_A=True).fit(X)
    # Ensure that the A matrix is unitary and accurate.
    assert_allclose(pidmd.A.conj().T.dot(pidmd.A), np.eye(25), atol=5*1e-15)
    assert_accurate(pidmd.A)

def test_uppertriangular():
    # Throw error if model is fitted without computing A.
    with raises(ValueError):
        pidmd = PiDMD("uppertriangular").fit(X)
    pidmd = PiDMD("uppertriangular", compute_A=True).fit(X)
    # Ensure that the A matrix is uppertriangular and accurate.
    assert_all_zero(np.tril(pidmd.A, k=-1))
    assert_accurate(pidmd.A)

def test_lowertriangular():
    # Throw error if model is fitted without computing A.
    with raises(ValueError):
        pidmd = PiDMD("lowertriangular").fit(X)
    pidmd = PiDMD("lowertriangular", compute_A=True).fit(X)
    # Ensure that the A matrix is lowertriangular and accurate.
    assert_all_zero(np.triu(pidmd.A, k=1))
    assert_accurate(pidmd.A)

def test_diagonal():
    # Throw error if manifold_opt is formatted incorrectly.
    with raises(ValueError):
        pidmd = PiDMD("diagonal", manifold_opt=-1, compute_A=True).fit(X)

    # Test diagonal model with integer manifold_opt.
    pidmd = PiDMD("diagonal", manifold_opt=1, compute_A=True).fit(X)
    assert_all_zero(pidmd.A - np.diag(np.diag(pidmd.A)))
    assert_accurate(pidmd.A)

    # Test tridiagonal model with tuple manifold_opt.
    pidmd = PiDMD("diagonal", manifold_opt=(2,2), compute_A=True).fit(X)
    assert_all_zero(np.triu(pidmd.A, k=2) + np.tril(pidmd.A, k=-2))
    assert_accurate(pidmd.A)

    # Test model with custom (2, nx) manifold_opt.
    custom_opt = np.hstack((np.ones((25,1)), 4*np.ones((25,1))))
    pidmd = PiDMD("diagonal", manifold_opt=custom_opt, compute_A=True).fit(X)
    assert_all_zero(np.triu(pidmd.A, k=4) + np.tril(pidmd.A, k=-1))
    assert_accurate(pidmd.A)

def test_symmetric():
    pidmd = PiDMD("symmetric", compute_A=True).fit(X)
    # Ensure that the A matrix is symmetric and accurate.
    assert_allclose(pidmd.A, pidmd.A.T)
    assert_accurate(pidmd.A)

def test_skewsymmetric():
    pidmd = PiDMD("skewsymmetric", compute_A=True).fit(X)
    # Ensure that the A matrix is skewsymmetric and accurate.
    assert_allclose(pidmd.A, -pidmd.A.T, atol=5*1e-15)
    assert_accurate(pidmd.A)

def test_toeplitz():
    # Throw error if model is fitted without computing A.
    with raises(ValueError):
        pidmd = PiDMD("toeplitz").fit(X)
    pidmd = PiDMD("toeplitz", compute_A=True).fit(X)
    # Ensure that the A matrix is toeplitz and accurate.
    for k in range(-24, 25):
        k_diag = np.diag(pidmd.A, k=k)
        assert_allclose(k_diag, k_diag[0]*np.ones(k_diag.shape))
    assert_accurate(pidmd.A)

def test_hankel():
    # Throw error if model is fitted without computing A.
    with raises(ValueError):
        pidmd = PiDMD("hankel").fit(X)
    pidmd = PiDMD("hankel", compute_A=True).fit(X)
    # Ensure that the A matrix is hankel and accurate.
    for k in range(-24, 25):
        k_diag = np.diag(np.fliplr(pidmd.A), k=k)
        assert_allclose(k_diag, k_diag[0]*np.ones(k_diag.shape))
    assert_accurate(pidmd.A)

def test_circulant():
    # Test circulant case.
    pidmd = PiDMD("circulant", compute_A=True).fit(X)
    assert_circulant(pidmd.A)
    assert_accurate(pidmd.A)

    # Test circulant_unitary case.
    pidmd = PiDMD("circulant_unitary", compute_A=True).fit(X)
    assert_circulant(pidmd.A)
    assert_accurate(pidmd.A)
    assert_allclose(pidmd.A.conj().T.dot(pidmd.A), np.eye(25), atol=5*1e-15)

    # Test circulant_symmetric case.
    pidmd = PiDMD("circulant_symmetric", compute_A=True).fit(X)
    assert_circulant(pidmd.A)
    assert_accurate(pidmd.A)
    assert_allclose(pidmd.A, pidmd.A.T)

    # Test circulant_skewsymmetric case.
    pidmd = PiDMD("circulant_skewsymmetric", compute_A=True).fit(X)
    assert_circulant(pidmd.A)
    assert_accurate(pidmd.A)
    assert_allclose(pidmd.A, -pidmd.A.T, atol=5*1e-15)

def test_symmetric_tridiagonal():
    pidmd = PiDMD("symmetric_tridiagonal", compute_A=True).fit(X)
    # Ensure that the A matrix is symmetric, tridiagonal, and accurate.
    assert_allclose(pidmd.A, pidmd.A.T)
    assert_all_zero(np.triu(pidmd.A, k=2) + np.tril(pidmd.A, k=-2))
    assert_accurate(pidmd.A)

def test_BC():
    # Throw error if model is fitted without computing A.
    with raises(ValueError):
        pidmd = PiDMD("BC", manifold_opt=(5,5)).fit(X)

    # Throw error if model is fitted without providing block size.
    with raises(ValueError):
        pidmd = PiDMD("BC", compute_A=True).fit(X)

    # BC (block circulant)
    pidmd = PiDMD("BC", manifold_opt=(5,5), compute_A=True).fit(X)
    assert_block_circulant(pidmd.A)
    assert_accurate(pidmd.A)

    # BCTB (block circulant with tridiagonal blocks)
    pidmd = PiDMD("BCTB", manifold_opt=(5,5), compute_A=True).fit(X)
    assert_block_circulant(pidmd.A)
    assert_accurate(pidmd.A)
    for i in range(5):
        for j in range(5):
            block = pidmd.A[i*5:(i+1)*5, j*5:(j+1)*5]
            assert_all_zero(np.triu(block, k=2) + np.tril(block, k=-2))

    # BCCB (block circulant with circulant blocks)
    pidmd = PiDMD("BCCB", manifold_opt=(5,5), compute_A=True).fit(X)
    assert_block_circulant(pidmd.A)
    assert_accurate(pidmd.A)
    for i in range(5):
        for j in range(5):
            block = pidmd.A[i*5:(i+1)*5, j*5:(j+1)*5]
            assert_circulant(block)

    # BCCB and unitary
    pidmd = PiDMD("BCCBunitary", manifold_opt=(5,5), compute_A=True).fit(X)
    assert_block_circulant(pidmd.A)
    assert_accurate(pidmd.A)
    for i in range(5):
        for j in range(5):
            block = pidmd.A[i*5:(i+1)*5, j*5:(j+1)*5]
            assert_circulant(block)
    assert_allclose(pidmd.A.conj().T.dot(pidmd.A), np.eye(25), atol=5*1e-15)

    # BCCB and symmetric
    pidmd = PiDMD("BCCBsymmetric", manifold_opt=(5,5), compute_A=True).fit(X)
    assert_block_circulant(pidmd.A)
    assert_accurate(pidmd.A)
    for i in range(5):
        for j in range(5):
            block = pidmd.A[i*5:(i+1)*5, j*5:(j+1)*5]
            assert_circulant(block)
    assert_allclose(pidmd.A, pidmd.A.T)

    # BCCB and skewsymmetric
    pidmd = PiDMD("BCCBskewsymmetric",manifold_opt=(5,5),compute_A=True).fit(X)
    assert_block_circulant(pidmd.A)
    assert_accurate(pidmd.A)
    for i in range(5):
        for j in range(5):
            block = pidmd.A[i*5:(i+1)*5, j*5:(j+1)*5]
            assert_circulant(block)
    assert_allclose(pidmd.A, -pidmd.A.T, atol=5*1e-15)
