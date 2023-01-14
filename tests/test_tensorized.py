import pytest
import torch
from torch.autograd import gradcheck
from pydmd import DMD, FbDMD, HankelDMD, HODMD, DMDc, CDMD, SubspaceDMD, RDMD
import numpy as np

from .utils import assert_allclose, setup_backends, noisy_data
from .test_backprop import fit_reconstruct

torch_backends = setup_backends(filters=("NumPy",))
noisy_backends = setup_backends(data=noisy_data(), filters=("NumPy",))

dmds = [
    pytest.param(CDMD(svd_rank=-1), id="CDMD"),
    pytest.param(DMD(svd_rank=-1), id="DMD"),
    pytest.param(DMD(svd_rank=-1, opt=True), id="DMD opt=True"),
    pytest.param(DMD(svd_rank=-1, exact=True), id="DMD exact=True"),
    pytest.param(
        DMD(svd_rank=-1, opt=True, exact=True), id="DMD exact+opt=True"
    ),
    pytest.param(FbDMD(svd_rank=-1), id="FbDMD"),
    pytest.param(HankelDMD(svd_rank=-1, d=3), id="HankelDMD"),
    pytest.param(HODMD(svd_rank=-1, d=3, svd_rank_extra=-1), id="HODMD"),
    pytest.param(SubspaceDMD(svd_rank=-1), id="SubspaceDMD"),
    pytest.param(RDMD(svd_rank=-1), id="RDMD"),
]


@pytest.mark.parametrize("dmd", dmds)
@pytest.mark.parametrize("X", torch_backends)
def test_tensorized_snapshots(dmd, X):
    X = torch.stack([X * i for i in range(1, 11)])
    dmd.fit(X=X, batch=True)
    assert dmd.snapshots.shape == X.shape
    assert_allclose(dmd.snapshots[0], X[0])


@pytest.mark.parametrize(
    "dmd",
    [
        pytest.param(CDMD(svd_rank=0), id="CDMD-0"),
        pytest.param(CDMD(svd_rank=0.1), id="CDMD-0.1"),
        pytest.param(DMD(svd_rank=0), id="DMD-0"),
        pytest.param(DMD(svd_rank=0.2), id="DMD-0.1"),
        pytest.param(FbDMD(svd_rank=0), id="FbDMD-0"),
        pytest.param(FbDMD(svd_rank=0.1), id="FbDMD-0.1"),
        pytest.param(HankelDMD(svd_rank=0, d=3), id="HankelDMD-0"),
        pytest.param(HankelDMD(svd_rank=0.1, d=3), id="HankelDMD-0.1"),
        pytest.param(HODMD(svd_rank=0, d=3, svd_rank_extra=-1), id="HODMD-0"),
        pytest.param(
            HODMD(svd_rank=0.1, d=3, svd_rank_extra=-1), id="HODMD-0.1"
        ),
        pytest.param(
            HODMD(svd_rank=-1, d=3, svd_rank_extra=0), id="HODMD-extra-0"
        ),
        pytest.param(
            HODMD(svd_rank=-1, d=3, svd_rank_extra=0.1), id="HODMD-extra-0.1"
        ),
    ],
)
@pytest.mark.parametrize("X", torch_backends)
def test_tensorized_fit_rejects_auto_svd_rank(dmd, X):
    X = torch.stack([X * i for i in range(1, 11)])
    with pytest.raises(
        ValueError,
        match="Automatic SVD rank selection not available in tensorized DMD",
    ):
        dmd.fit(X=X, batch=True)


@pytest.mark.parametrize("dmd", dmds)
@pytest.mark.parametrize("X", torch_backends)
def test_tensorized_reconstructed_data(dmd, X):
    if isinstance(dmd, (FbDMD, SubspaceDMD)):
        pytest.skip()
    X = torch.stack([X * i for i in range(1, 11)])
    dmd.fit(X=X, batch=True)
    assert_allclose(dmd.reconstructed_data, X)


@pytest.mark.parametrize("X", noisy_backends)
def test_tensorized_reconstructed_data_fbdmd(X):
    X = torch.stack([X * i for i in range(1, 11)])
    dmd = FbDMD(exact=True, svd_rank=-1)
    dmd.fit(X=X, batch=True)
    dmd_data_correct = np.stack(
        [
            np.load("tests/test_datasets/fbdmd_data.npy") * i
            for i in range(1, 11)
        ]
    )
    assert_allclose(dmd.reconstructed_data, dmd_data_correct)


@pytest.mark.parametrize("dmd", dmds)
@pytest.mark.parametrize("X", torch_backends)
def test_tensorized_backprop(dmd, X):
    X = torch.stack([X * i for i in range(1, 11)])
    X.requires_grad = True
    dmd.fit(X=X, batch=True)
    dmd.reconstructed_data.sum().backward()
    X.requires_grad = False


@pytest.mark.parametrize("dmd", dmds)
@pytest.mark.parametrize("X", torch_backends)
def test_tensorized_second_fit_backprop(dmd, X):
    X = torch.stack([X * i for i in range(1, 11)])
    X.requires_grad = True
    dmd.fit(X=X, batch=True)
    dmd.reconstructed_data.sum().backward()

    dmd.fit(X=X.clone(), batch=True)
    dmd.reconstructed_data.sum().backward()
    X.requires_grad = False


@pytest.mark.gradcheck
@pytest.mark.parametrize("dmd", dmds)
@pytest.mark.parametrize("X", torch_backends)
def test_tensorized_backprop_gradcheck(dmd, X):
    X = torch.stack([X * i for i in range(1, 11)])
    X.requires_grad = True
    assert gradcheck(fit_reconstruct(dmd), X)
    X.requires_grad = False


@pytest.mark.gradcheck
@pytest.mark.parametrize("dmd", dmds)
@pytest.mark.parametrize("X", torch_backends)
def test_tensorized_second_fit_backprop_gradcheck(dmd, X):
    X = torch.stack([X * i for i in range(1, 11)])
    X.requires_grad = True
    dmd.fit(X=X)
    assert gradcheck(fit_reconstruct(dmd), X)
    X.requires_grad = False
