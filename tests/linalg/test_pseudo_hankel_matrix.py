from functools import partial

import pytest

from pydmd.linalg import build_linalg_module

from .utils import Backend, assert_allclose, setup_backends

torch_backends = setup_backends(exclude=Backend.NUMPY)


@pytest.mark.parametrize("X", torch_backends)
def test_tensorized_pseudo_hankel_matrix(X):
    linalg_module = build_linalg_module(X)
    X = linalg_module.new_array([X * i for i in range(1, 11)])
    phm = partial(linalg_module.pseudo_hankel_matrix, d=5)
    assert_allclose(
        phm(X),
        list(map(phm, X)),
    )
