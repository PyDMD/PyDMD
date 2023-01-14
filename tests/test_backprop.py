import pytest
from torch.autograd import gradcheck

from pydmd import CDMD, DMD, HODMD, DMDc, FbDMD, HankelDMD, SubspaceDMD, RDMD

from .utils import setup_backends, sample_data

torch_backends = setup_backends(data=sample_data()[:100], filters=("NumPy",))

dmds = [
    pytest.param(CDMD(svd_rank=-1), id="CDMD"),
    pytest.param(DMD(svd_rank=-1), id="DMD"),
    # TODO
    # pytest.param(DMDc(svd_rank=-1), id="DMDc"),
    pytest.param(FbDMD(svd_rank=-1), id="FbDMD"),
    pytest.param(HankelDMD(svd_rank=-1, d=3), id="HankelDMD"),
    pytest.param(HODMD(svd_rank=-1, d=3, svd_rank_extra=-1), id="HODMD"),
    pytest.param(SubspaceDMD(svd_rank=-1), id="SubspaceDMD"),
    pytest.param(RDMD(svd_rank=-1), id="RDMD"),
]


def fit_reconstruct(dmd):
    def func(X):
        batch = X.ndim == 3
        return dmd.fit(X, batch=batch).reconstructed_data

    return func


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


@pytest.mark.gradcheck
@pytest.mark.parametrize("dmd", dmds)
@pytest.mark.parametrize("X", torch_backends)
def test_backprop_gradcheck(dmd, X):
    X = X.clone()
    X.requires_grad = True
    assert gradcheck(fit_reconstruct(dmd), X)
    X.requires_grad = False


@pytest.mark.gradcheck
@pytest.mark.parametrize("dmd", dmds)
@pytest.mark.parametrize("X", torch_backends)
def test_second_fit_backprop_gradcheck(dmd, X):
    X = X.clone()
    X.requires_grad = True
    dmd.fit(X=X)
    assert gradcheck(fit_reconstruct(dmd), X)
    X.requires_grad = False
