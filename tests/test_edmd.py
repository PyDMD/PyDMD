import numpy as np
from numpy.testing import assert_allclose
from pytest import raises

from pydmd.edmd import EDMD

# Simulate testing data using 500 random initial conditions.
J = np.array([[0.9, -0.1], [0.0, 0.8]])
rng = np.random.default_rng(seed=42)
X = rng.standard_normal((2, 500))
Y = J.dot(X)

# Additionally use model to propagate a single initial condition forward.
X2 = np.empty(X.shape)
X2[:, 0] = X[:, 0]
for i in range(X.shape[1] - 1):
    X2[:, i + 1] = J.dot(X2[:, i])

# Compute the first 8 ground truth eigenvalues and eigenfunctions along
# the defined xy-grid (excluding the eigenfunction, eigenvalue pair 1,1).
x_vals = np.linspace(-5, 5, 11)
y_vals = np.linspace(5, -5, 11)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
eigenfunctions_true = np.empty((8, *X_grid.shape))
eigenvalues_true = np.empty(8)
for idx, (i, j) in enumerate(
    zip([1, 2, 0, 3, 1, 4, 2, 0], [0, 0, 1, 0, 1, 0, 1, 2])
):
    eigenfunc = np.multiply(((X_grid - Y_grid) / np.sqrt(2)) ** i, Y_grid**j)
    eigenfunc /= np.linalg.norm(eigenfunc, np.inf)
    eigenfunctions_true[idx] = eigenfunc
    eigenvalues_true[idx] = (0.9**i) * (0.8**j)


def relative_error(mat, mat_true):
    """
    Computes and returns the relative error between two matrices.
    """
    return np.linalg.norm(mat - mat_true) / np.linalg.norm(mat_true)


def test_eigs():
    """
    Test that EDMD can sucessfully recover the true eigenvalues
    of the toy system from the original EDMD paper.
    """
    edmd = EDMD(
        svd_rank=15,
        kernel_metric="poly",
        kernel_params={"gamma": 1, "coef0": 1, "degree": 4},
    )
    edmd.fit(X, Y)
    sorted_inds = np.argsort(-np.abs(edmd.eigs))
    edmd_eigs = edmd.eigs[sorted_inds][1:9]
    assert_allclose(edmd_eigs, eigenvalues_true)


def test_eigenfunctions():
    """
    Test that EDMD can sucessfully recover the true eigenfunctions
    of the toy system from the original EDMD paper.
    """
    edmd = EDMD(
        svd_rank=15,
        kernel_metric="poly",
        kernel_params={"gamma": 1, "coef0": 1, "degree": 4},
    )
    edmd.fit(X, Y)

    # Evaluate eigenfunctions from EDMD along the grid.
    eigenfunctions = np.empty((15, *X_grid.shape))
    for y_idx, y in enumerate(Y_grid[:, 0]):
        for x_idx, x in enumerate(X_grid[0, :]):
            xy = np.array([x, y])
            eigenfunctions[:, y_idx, x_idx] = edmd.eigenfunctions(xy).real

    # Scale eigenfunctions to have infinity norm 1.
    for eigenfunction in eigenfunctions:
        eigenfunction /= np.linalg.norm(eigenfunction, np.inf)

    # Sort the eigenfunctions based on eigenvalue magnitude.
    sorted_inds = np.argsort(-np.abs(edmd.eigs))
    edmd_eigenfunctions = eigenfunctions[sorted_inds][1:9]

    # Realign eigenfunction signs for numerical comparison.
    for func, func_true in zip(edmd_eigenfunctions, eigenfunctions_true):
        if np.sign(func[0, 0]) != np.sign(func_true[0, 0]):
            func *= -1
        assert relative_error(func, func_true) < 1e-6


def test_operator():
    """
    Test that EDMD can sucessfully recover the true linear operator.
    Essentially tests that the eigenvalues and modes are correct.
    """
    edmd = EDMD(
        svd_rank=15,
        kernel_metric="poly",
        kernel_params={"gamma": 1, "coef0": 1, "degree": 4},
    )
    edmd.fit(X, Y)
    J_est = np.linalg.multi_dot(
        [edmd.modes, np.diag(edmd.eigs), np.linalg.pinv(edmd.modes)]
    )
    assert_allclose(J_est, J, atol=1e-13)


def test_reconstruction():
    """
    Test that EDMD can sucessfully reconstruct the forward-time system.
    """
    edmd = EDMD(
        svd_rank=15,
        kernel_metric="poly",
        kernel_params={"gamma": 1, "coef0": 1, "degree": 4},
    )
    edmd.fit(X, Y)
    assert_allclose(edmd.reconstructed_data, X2, atol=1e-13)


def test_kernel_errors():
    """
    Test that EDMD throws error upon initialization in the following scenarios:
        - kernel_metric is invalid
        - kernel_params isn't a dictionary
        - kernel_params contains invalid value types
        - kernel_params contains entries incompatible with the kernel
    """
    with raises(ValueError):
        EDMD(kernel_metric="rbf!")

    with raises(TypeError):
        EDMD(kernel_metric="rbf", kernel_params=3)

    with raises(ValueError):
        EDMD(kernel_metric="rbf", kernel_params={"gamma": "three"})

    with raises(ValueError):
        EDMD(kernel_metric="rbf", kernel_params={"gamma": 3, "degree": 5})


def test_svd_error():
    """
    Test that EDMD throws error upon initialization if svd_rank is zero.
    """
    with raises(ValueError):
        EDMD(svd_rank=0)


def test_atilde_error():
    """
    Test that EDMD throws error if a user asks for the operator matrix.
    """
    edmd = EDMD().fit(X, Y)

    with raises(ValueError):
        _ = edmd.operator.as_numpy_array

    with raises(ValueError):
        _ = edmd.operator.shape


def test_eigenfunction_error():
    """
    Test that EDMD throws an error if a user attempts to either
    - compute eigenfunctions prior to calling fit, or
    - compute eigenfunctions using input other than a 1-D numpy array.
    """
    edmd = EDMD()
    x_dummy = np.array([1, 1])

    with raises(ValueError):
        edmd.eigenfunctions(x_dummy)

    edmd.fit(X, Y)

    with raises(ValueError):
        edmd.eigenfunctions([1, 1])

    with raises(ValueError):
        edmd.eigenfunctions(x_dummy[None])
