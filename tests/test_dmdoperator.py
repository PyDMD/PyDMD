import numpy as np
import pytest
from pytest import raises

from pydmd.dmdoperator import DMDOperator
from pydmd.utils import compute_tlsq

from .linalg.utils import assert_allclose, setup_backends

data_backends = setup_backends()


def test_constructor():
    operator = DMDOperator(
        svd_rank=2,
        exact=True,
        forward_backward=False,
        rescale_mode="auto",
        sorted_eigs=False,
        tikhonov_regularization=None,
    )

    assert operator._svd_rank == 2
    assert operator._exact == True
    assert operator._forward_backward == False
    assert operator._rescale_mode == "auto"


def test_noncompute_error():
    operator = DMDOperator(
        svd_rank=2,
        exact=True,
        forward_backward=False,
        rescale_mode="auto",
        sorted_eigs=False,
        tikhonov_regularization=None,
    )

    with raises(ValueError):
        operator.shape

    with raises(ValueError):
        operator.Lambda

    with raises(ValueError):
        operator.modes

    with raises(ValueError):
        operator.eigenvalues

    with raises(ValueError):
        operator.eigenvectors

    with raises(ValueError):
        operator.as_array


def test_compute_operator():
    operator = DMDOperator(
        svd_rank=0,
        exact=True,
        forward_backward=False,
        rescale_mode="auto",
        sorted_eigs=False,
        tikhonov_regularization=None,
    )
    operator.compute_operator(np.ones((3, 3)), np.ones((3, 3)))

    assert operator.as_array is not None
    assert operator.eigenvalues is not None
    assert operator.eigenvectors is not None
    assert operator.modes is not None
    assert operator.Lambda is not None


# test that a value of 'auto' in rescale_mode is replaced by the singular
# values of X
def test_rescalemode_auto_singular_values():
    operator = DMDOperator(
        svd_rank=0,
        exact=True,
        forward_backward=False,
        rescale_mode="auto",
        sorted_eigs=False,
        tikhonov_regularization=None,
    )
    operator.compute_operator(np.ones((3, 3)), np.ones((3, 3)))
    assert_allclose(operator._rescale_mode, [3.0])


@pytest.mark.parametrize("X", data_backends)
def test_call(X):
    operator = DMDOperator(
        svd_rank=2,
        exact=True,
        forward_backward=False,
        rescale_mode=None,
        sorted_eigs=False,
        tikhonov_regularization=None,
    )

    a = X[:, :-1]
    b = X[:, 1:]
    a, b = compute_tlsq(a, b, 0)

    operator.compute_operator(a, b)

    expected = [-0.47643628 + 0.87835227j, -0.47270971 + 0.88160808j]
    assert_allclose(operator(np.ones(2)), expected, atol=1.0e-6)


def test_compute_eigenquantities_wrong_rescalemode():
    operator = DMDOperator(
        svd_rank=0,
        exact=True,
        forward_backward=False,
        rescale_mode=4,
        sorted_eigs=False,
        tikhonov_regularization=None,
    )
    with raises(ValueError):
        operator.compute_operator(np.ones((3, 3)), np.ones((3, 3)))

    operator = DMDOperator(
        svd_rank=0,
        exact=True,
        forward_backward=False,
        rescale_mode=np.ones((4,)),
        sorted_eigs=False,
        tikhonov_regularization=None,
    )
    with raises(ValueError):
        operator.compute_operator(np.ones((3, 3)), np.ones((3, 3)))
