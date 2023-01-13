import numpy as np

from pydmd import SubspaceDMD
from pydmd.subspacedmd import SubspaceDMDOperator, reducedsvd


def test_smoke():
    dmd = SubspaceDMD()
    x = np.random.rand(200, 100)
    assert dmd.fit(x) is not None


def test_modes_shape():
    dmd = SubspaceDMD()
    x = np.random.rand(200, 100)
    assert dmd.fit(x).modes.shape[0] == 200


def test_fixed_parameters():
    dmd = SubspaceDMD()

    assert dmd._tlsq_rank == 0
    assert not dmd.operator._forward_backward
    assert dmd.operator._tikhonov_regularization is None
    assert dmd.operator._exact


def test_default_constructor():
    dmd = SubspaceDMD()

    assert not dmd._opt
    assert dmd.operator._svd_rank == -1
    assert dmd.operator._rescale_mode is None
    assert not dmd.operator._sorted_eigs

    assert dmd._original_time is None
    assert dmd._dmd_time is None
    assert dmd._b is None
    assert dmd._snapshots_holder is None
    assert dmd._modes_activation_bitmask_proxy is None


def test_constructor():
    dmd = SubspaceDMD(
        svd_rank=20,
        opt=True,
        rescale_mode="pippo",
        sorted_eigs="pluto",
    )

    assert dmd._opt
    assert dmd.operator._svd_rank == 20
    assert dmd.operator._rescale_mode == "pippo"
    assert dmd.operator._sorted_eigs == "pluto"


def test_operator():
    dmd = SubspaceDMD()
    assert isinstance(dmd._Atilde, SubspaceDMDOperator)


def test_time():
    X = np.random.rand(10, 100)
    dmd = SubspaceDMD().fit(X)

    assert dmd.dmd_time["t0"] == 0
    assert dmd.dmd_time["tend"] == 99
    assert dmd.dmd_time["dt"] == 1


def test_initial_time():
    X = np.random.rand(10, 100)
    dmd = SubspaceDMD().fit(X)

    assert dmd.original_time["t0"] == 0
    assert dmd.original_time["tend"] == 99
    assert dmd.original_time["dt"] == 1


def test_amplitudes():
    X = np.random.rand(10, 100)
    dmd = SubspaceDMD().fit(X)

    assert dmd.amplitudes is not None


def test_modes():
    X = np.random.rand(10, 100)
    dmd = SubspaceDMD().fit(X)

    assert dmd.modes is not None
    assert dmd.modes.shape[0] == 10


def test_eigs():
    X = np.random.rand(10, 100)
    dmd = SubspaceDMD().fit(X)

    assert dmd.eigs is not None


def test_snapshots():
    X = np.random.rand(10, 100)
    dmd = SubspaceDMD().fit(X)

    assert dmd.snapshots.shape == (10, 100)


def test_reducedsvd():
    X = np.eye(4)
    X = np.hstack((X, X))
    X = np.vstack((X, np.zeros((5, X.shape[1]))))

    U, s, V = reducedsvd(X)
    assert U.shape[1] == 4
    assert V.shape[1] == 4
    assert len(s) == 4


def test_reducedsvd_with_r():
    X = np.eye(4)
    X = np.hstack((X, X))
    X = np.vstack((X, np.zeros((5, X.shape[1]))))

    U, s, V = reducedsvd(X, 2)
    assert U.shape[1] == 2
    assert V.shape[1] == 2
    assert len(s) == 2


def test_svd_rank_positive():
    X = np.random.rand(10, 100)
    dmd = SubspaceDMD(svd_rank=2).fit(X)

    assert len(dmd.eigs) == 2
    assert dmd.modes.shape[1] == 2
    assert len(dmd.amplitudes) == 2
