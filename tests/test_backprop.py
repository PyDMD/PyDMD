from functools import partial

import pytest
import torch

from pydmd import CDMD, DMD, HODMD, DMDc, FbDMD, HankelDMD

from .utils import setup_backends

torch_backends = setup_backends(filters=("NumPy",))

dmds = [
    pytest.param(CDMD(svd_rank=-1), id="CDMD"),
    pytest.param(DMD(svd_rank=-1), id="DMD"),
    # TODO
    # pytest.param(DMDc(svd_rank=-1), id="DMDc"),
    pytest.param(FbDMD(svd_rank=-1), id="FbDMD"),
    pytest.param(HankelDMD(svd_rank=-1, d=3), id="HankelDMD"),
    pytest.param(HODMD(svd_rank=-1, d=3, svd_rank_extra=-1), id="HODMD"),
]


@pytest.mark.parametrize("dmd", dmds)
@pytest.mark.parametrize("X", torch_backends)
def test_backprop(dmd, X):
    X = X.clone()
    X.requires_grad = True
    dmd.fit(X=X)
    dmd.reconstructed_data.sum().backward()
    X.requires_grad = False


@pytest.mark.parametrize("dmd", dmds)
@pytest.mark.parametrize("X", torch_backends)
def test_second_fit_backprop(dmd, X):
    X = X.clone()
    X.requires_grad = True
    dmd = DMD(svd_rank=-1)
    dmd.fit(X=X)
    dmd.reconstructed_data.sum().backward()

    dmd.fit(X=X.clone())
    dmd.reconstructed_data.sum().backward()
    X.requires_grad = False
