import pytest
import torch
from pydmd import DMD, FbDMD, HankelDMD, HODMD, DMDc, CDMD

from .utils import assert_allclose, setup_backends

torch_backends = setup_backends(filters=("NumPy",))

dmds = [
    pytest.param(CDMD(svd_rank=-1), id="CDMD"),
    pytest.param(DMD(svd_rank=-1), id="DMD"),
    pytest.param(DMD(svd_rank=-1, opt=True), id="DMD opt=True"),
    pytest.param(DMD(svd_rank=-1, exact=True), id="DMD exact=True"),
    pytest.param(DMD(svd_rank=-1, opt=True, exact=True), id="DMD exact+opt=True"),
    # TODO
    # pytest.param(DMDc(svd_rank=-1), id="DMDc"),
    pytest.param(FbDMD(svd_rank=-1), id="FbDMD"),
    pytest.param(HankelDMD(svd_rank=-1, d=3), id="HankelDMD"),
    pytest.param(HODMD(svd_rank=-1, d=3), id="HODMD"),
]

@pytest.mark.parametrize("dmd", dmds)
@pytest.mark.parametrize("X", torch_backends)
def test_tensorizexd_snapshots(dmd, X):
    X = torch.stack([X*i for i in range(1,11)])
    dmd.fit(X=X)
    assert dmd._snapshots.shape == X.shape
    assert_allclose(dmd._snapshots[0], X[0])

@pytest.mark.parametrize("dmd", dmds)
@pytest.mark.parametrize("X", torch_backends)
def test_tensorized_reconstructed_data(dmd, X):
    X = torch.stack([X*i for i in range(1,11)])
    dmd.fit(X=X)
    assert_allclose(dmd.reconstructed_data, X)

@pytest.mark.parametrize("dmd", dmds)
@pytest.mark.parametrize("X", torch_backends)
def test_tensorized_backprop(dmd, X):
    X = torch.stack([X*i for i in range(1,11)])
    X.requires_grad = True
    dmd.fit(X=X)
    dmd.reconstructed_data.sum().backward()
    X.requires_grad = False

@pytest.mark.parametrize("dmd", dmds)
@pytest.mark.parametrize("X", torch_backends)
def test_tensorized_second_fit_backprop(dmd, X):
    X = torch.stack([X*i for i in range(1,11)])
    X.requires_grad = True
    dmd.fit(X=X)
    dmd.reconstructed_data.sum().backward()
    
    dmd.fit(X=X.clone())
    dmd.reconstructed_data.sum().backward()
    X.requires_grad = False