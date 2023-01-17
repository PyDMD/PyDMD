import numpy as np
import pytest
from past.utils import old_div
from pytest import raises

from pydmd import DMD, FbDMD, MrDMD

from .utils import assert_allclose, setup_backends


def create_data(t_size=1600):
    x = np.linspace(-10, 10, 80)
    t = np.linspace(0, 20, t_size)
    Xm, Tm = np.meshgrid(x, t)

    D = np.exp(-np.power(old_div(Xm, 2), 2)) * np.exp(0.8j * Tm)
    D += np.sin(0.9 * Xm) * np.exp(1j * Tm)
    D += np.cos(1.1 * Xm) * np.exp(2j * Tm)
    D += 0.6 * np.sin(1.2 * Xm) * np.exp(3j * Tm)
    D += 0.6 * np.cos(1.3 * Xm) * np.exp(4j * Tm)
    D += 0.2 * np.sin(2.0 * Xm) * np.exp(6j * Tm)
    D += 0.2 * np.cos(2.1 * Xm) * np.exp(8j * Tm)
    D += 0.1 * np.sin(5.7 * Xm) * np.exp(10j * Tm)
    D += 0.1 * np.cos(5.9 * Xm) * np.exp(12j * Tm)
    D += 0.1 * np.random.randn(*Xm.shape)
    D += 0.03 * np.random.randn(*Xm.shape)
    D += 5 * np.exp(-np.power(old_div((Xm + 5), 5), 2)) * np.exp(-np.power(
        old_div((Tm - 5), 5), 2))
    D[:800, 40:] += 2
    D[200:600, 50:70] -= 3
    D[800:, :40] -= 2
    D[1000:1400, 10:30] += 3
    D[1000:1080, 50:70] += 2
    D[1160:1240, 50:70] += 2
    D[1320:1400, 50:70] += 2
    return D.T


data_backends = setup_backends(create_data())

@pytest.mark.parametrize("X", data_backends)
def test_max_level_threshold(X):
    level = 10
    dmd = DMD()
    mrdmd = MrDMD(dmd, max_level=level, max_cycles=2)
    mrdmd.fit(X=X)
    lvl_threshold = int(np.log(X.shape[1]/4.)/np.log(2.)) + 1
    assert lvl_threshold == mrdmd.max_level

@pytest.mark.parametrize("X", data_backends)
def test_partial_time_interval(X):
    level = 4
    dmd = DMD()
    mrdmd = MrDMD(dmd, max_level=level, max_cycles=2)
    mrdmd.fit(X=X)
    ans = {'t0': 1200, 'tend': 1400.0, 'delta': 200.0}
    assert mrdmd.partial_time_interval(3, 6) == ans

@pytest.mark.parametrize("X", data_backends)
def test_partial_time_interval2(X):
    level = 4
    dmd = DMD()
    mrdmd = MrDMD(dmd, max_level=level, max_cycles=2)
    mrdmd.fit(X=X)
    with raises(ValueError):
        mrdmd.partial_time_interval(5, 0)

@pytest.mark.parametrize("X", data_backends)
def test_partial_time_interval3(X):
    level = 4
    dmd = DMD()
    mrdmd = MrDMD(dmd, max_level=level, max_cycles=2)
    mrdmd.fit(X=X)
    with raises(ValueError):
        mrdmd.partial_time_interval(3, 8)

@pytest.mark.parametrize("X", data_backends)
def test_time_window_bins(X):
    level = 4
    dmd = DMD()
    mrdmd = MrDMD(dmd, max_level=level, max_cycles=2)
    mrdmd.fit(X=X)
    assert len(mrdmd.time_window_bins(0, 1600)) == 2**5-1

@pytest.mark.parametrize("X", data_backends)
def test_time_window_bins2(X):
    level = 3
    dmd = DMD()
    mrdmd = MrDMD(dmd, max_level=level, max_cycles=2)
    mrdmd.fit(X=X)
    expected_bins = np.array([
        [0, 0],
        [1, 0],
        [2, 0],
        [2, 1],
        [3, 1],
        [3, 2]])
    assert_allclose(mrdmd.time_window_bins(200, 600), expected_bins)

@pytest.mark.parametrize("X", data_backends)
def test_time_window_eigs(X):
    level = 2
    dmd = DMD()
    mrdmd = MrDMD(dmd, max_level=level, max_cycles=1)
    mrdmd.fit(X=X)
    exp =  sum([len(dmd.eigs) for dmd in mrdmd])
    assert len(mrdmd.time_window_eigs(0, 1600)) == exp

@pytest.mark.parametrize("X", data_backends)
def test_time_window_frequency(X):
    level = 2
    dmd = DMD()
    mrdmd = MrDMD(dmd, max_level=level, max_cycles=1)
    mrdmd.fit(X=X)
    exp =  sum([len(dmd.frequency) for dmd in mrdmd])
    assert len(mrdmd.time_window_frequency(0, 1600)) == exp

@pytest.mark.parametrize("X", data_backends)
def test_time_window_growth_rate(X):
    level = 2
    dmd = DMD()
    mrdmd = MrDMD(dmd, max_level=level, max_cycles=1)
    mrdmd.fit(X=X)
    exp =  sum([len(dmd.growth_rate) for dmd in mrdmd])
    assert len(mrdmd.time_window_growth_rate(0, 1600)) == exp

@pytest.mark.parametrize("X", data_backends)
def test_time_window_amplitudes(X):
    level = 2
    dmd = DMD()
    mrdmd = MrDMD(dmd, max_level=level, max_cycles=1)
    mrdmd.fit(X=X)
    exp =  sum([len(dmd.amplitudes) for dmd in mrdmd])
    assert len(mrdmd.time_window_amplitudes(0, 1600)) == exp

@pytest.mark.parametrize("X", data_backends)
def test_shape_modes(X):
    level = 2
    dmd = DMD(svd_rank=1)
    mrdmd = MrDMD(dmd, max_level=level, max_cycles=1)
    mrdmd.fit(X=X)
    assert mrdmd.modes.shape == (X.shape[0], 2**(level+1) - 1)

@pytest.mark.parametrize("X", data_backends)
def test_shape_dynamics(X):
    level = 2
    dmd = DMD(svd_rank=1)
    mrdmd = MrDMD(dmd, max_level=level, max_cycles=1)
    mrdmd.fit(X=X)
    assert mrdmd.dynamics.shape == (2**(level+1) - 1, X.shape[1])

@pytest.mark.parametrize("X", data_backends)
def test_reconstructed_data(X):
    dmd = DMD(svd_rank=1)
    mrdmd = MrDMD(dmd, max_level=6, max_cycles=2)
    mrdmd.fit(X=X)
    dmd_data = mrdmd.reconstructed_data
    norm_err = (old_div(
        np.linalg.norm(X - dmd_data),
        np.linalg.norm(X)))
    assert norm_err < 1

@pytest.mark.parametrize("X", data_backends)
def test_partial_modes1(X):
    level = 2
    rank = 2
    dmd = DMD(svd_rank=rank)
    mrdmd = MrDMD(dmd, max_level=6, max_cycles=2)
    mrdmd.fit(X=X)
    pmodes = mrdmd.partial_modes(level)
    assert pmodes.shape == (X.shape[0], 2**level * rank)

@pytest.mark.parametrize("X", data_backends)
def test_partial_modes2(X):
    level = 2
    rank = 2
    dmd = DMD(svd_rank=rank)
    mrdmd = MrDMD(dmd, max_level=6, max_cycles=2)
    mrdmd.fit(X=X)
    pmodes = mrdmd.partial_modes(level, 3)
    assert pmodes.shape == (X.shape[0], rank)

@pytest.mark.parametrize("X", data_backends)
def test_partial_dynamics1(X):
    level = 2
    rank = 2
    dmd = DMD(svd_rank=rank)
    mrdmd = MrDMD(dmd, max_level=6, max_cycles=2)
    mrdmd.fit(X=X)
    pdynamics = mrdmd.partial_dynamics(level)
    assert pdynamics.shape == (2**level * rank, X.shape[1])

@pytest.mark.parametrize("X", data_backends)
def test_partial_dynamics2(X):
    level = 2
    rank = 2
    dmd = DMD(svd_rank=rank)
    mrdmd = MrDMD(dmd, max_level=6, max_cycles=2)
    mrdmd.fit(X=X)
    pdynamics = mrdmd.partial_dynamics(level, 3)
    assert pdynamics.shape == (rank, X.shape[1] // 2**level)

@pytest.mark.parametrize("X", data_backends)
def test_eigs2(X):
    rank = -1
    dmd = DMD(svd_rank=rank)
    mrdmd = MrDMD(dmd, max_level=6, max_cycles=2)
    mrdmd.fit(X=X)
    assert mrdmd.eigs.ndim == 1

@pytest.mark.parametrize("X", data_backends)
def test_partial_eigs1(X):
    max_level = 5
    level = 2
    rank = 2
    dmd = DMD(svd_rank=rank)
    mrdmd = MrDMD(dmd, max_level=max_level, max_cycles=2)
    mrdmd.fit(X=X)
    peigs = mrdmd.partial_eigs(level)
    assert peigs.shape == (rank * 2**level, )

@pytest.mark.parametrize("X", data_backends)
def test_partial_eigs2(X):
    level = 2
    rank = 2
    dmd = DMD(svd_rank=rank)
    mrdmd = MrDMD(dmd, max_level=6, max_cycles=2)
    mrdmd.fit(X=X)
    peigs = mrdmd.partial_eigs(level, 3)
    assert peigs.shape == (rank, )

@pytest.mark.parametrize("X", data_backends)
def test_partial_reconstructed1(X):
    level = 2
    rank = 2
    dmd = DMD(svd_rank=rank)
    mrdmd = MrDMD(dmd, max_level=6, max_cycles=2)
    mrdmd.fit(X=X)
    pdata = mrdmd.partial_reconstructed_data(level)
    assert pdata.shape == X.shape

@pytest.mark.parametrize("X", data_backends)
def test_partial_reconstructed2(X):
    level = 2
    rank = 2
    dmd = DMD(svd_rank=rank)
    mrdmd = MrDMD(dmd, max_level=5, max_cycles=2)
    mrdmd.fit(X=X)
    pdata = mrdmd.partial_reconstructed_data(level, 3)
    assert pdata.shape == (X.shape[0], X.shape[1] // 2**level)

@pytest.mark.parametrize("X", data_backends)
def test_wrong_partial_reconstructed(X):
    max_level = 5
    rank = 2
    dmd = DMD(svd_rank=rank)
    mrdmd = MrDMD(dmd, max_level=max_level, max_cycles=2)
    mrdmd.fit(X=X)
    with raises(ValueError):
        mrdmd.partial_reconstructed_data(max_level+1, 2)

@pytest.mark.parametrize("X", data_backends)
def test_wrong_level(X):
    max_level = 5
    rank = 2
    dmd = DMD(svd_rank=rank)
    mrdmd = MrDMD(dmd, max_level=max_level, max_cycles=2)
    mrdmd.fit(X=X)
    with raises(ValueError):
        mrdmd.partial_modes(max_level + 1)

@pytest.mark.parametrize("X", data_backends)
def test_wrong_bin(X):
    level = 2
    dmd = DMD()
    mrdmd = MrDMD(dmd, max_level=5, max_cycles=2)
    mrdmd.fit(X=X)
    with raises(ValueError):
        mrdmd.partial_modes(level=level, node=2**level)

@pytest.mark.parametrize("X", data_backends)
def test_consistency(X):
    level = 5
    dmd = DMD()
    mrdmd = MrDMD(dmd, max_level=level, max_cycles=1)
    mrdmd.fit(X=X)
    assert_allclose(mrdmd.reconstructed_data, mrdmd.modes @ mrdmd.dynamics)

@pytest.mark.parametrize("X", data_backends)
def test_consistency2(X):
    import sys

    import numpy
    numpy.set_printoptions(threshold=sys.maxsize)

    mrdmd = MrDMD(DMD(), max_level=5, max_cycles=1)
    mrdmd.fit(X=X)

    assert_allclose(mrdmd.reconstructed_data, mrdmd.modes @ mrdmd.dynamics)

@pytest.mark.parametrize("X", data_backends)
def test_bitmask_not_implemented(X):
    with raises(RuntimeError):
        mrdmd = MrDMD(DMD(), max_level=5, max_cycles=1)
        mrdmd.fit(X=X)
        mrdmd.modes_activation_bitmask
    with raises(RuntimeError):
        mrdmd = MrDMD(DMD(), max_level=5, max_cycles=1)
        mrdmd.fit(X=X)
        mrdmd.modes_activation_bitmask = None

@pytest.mark.parametrize("X", data_backends)
def test_getitem_not_implemented(X):
    with raises(RuntimeError):
        mrdmd = MrDMD(DMD(), max_level=5, max_cycles=1)
        mrdmd.fit(X=X)
        mrdmd[1:3]

def test_one_dmd():
    m = MrDMD(dmd=DMD(svd_rank=4), max_level=5)

    for level in range(6):
        leaves = 2 ** level
        for leaf in range(leaves):
            dmd = m.dmd_tree[level, leaf]
            assert isinstance(dmd, DMD)
            assert dmd.operator._svd_rank == 4

def test_list_dmd():
    l = [DMD(svd_rank=5-i) for i in range(4)]
    m = MrDMD(dmd=l, max_level=3)

    for level in range(4):
        leaves = 2 ** level
        for leaf in range(leaves):
            dmd = m.dmd_tree[level, leaf]
            assert isinstance(dmd, DMD)
            assert dmd.operator._svd_rank == 5 - level

def test_tuple_dmd():
    l = tuple(DMD(svd_rank=5-i) for i in range(4))
    m = MrDMD(dmd=l, max_level=3)

    for level in range(4):
        leaves = 2 ** level
        for leaf in range(leaves):
            dmd = m.dmd_tree[level, leaf]
            assert isinstance(dmd, DMD)
            assert dmd.operator._svd_rank == 5 - level

def test_list_wrong_size_dmd():
    l = [DMD(svd_rank=5-i) for i in range(4)]
    with raises(ValueError):
        MrDMD(dmd=l, max_level=4)

def test_tuple_dmd():
    l = tuple(DMD(svd_rank=5-i) for i in range(5))
    with raises(ValueError):
        MrDMD(dmd=l, max_level=3)

def test_func_dmd():
    def f(level, leaf):
        return FbDMD(svd_rank=level*leaf)
    m = MrDMD(dmd=f, max_level=5)

    for level in range(6):
        leaves = 2 ** level
        for leaf in range(leaves):
            dmd = m.dmd_tree[level, leaf]
            assert isinstance(dmd, FbDMD)
            assert dmd.operator._svd_rank == level * leaf

@pytest.mark.parametrize("X", data_backends)
def test_quantitative_list_dmd(X):
    l = [DMD(svd_rank=4) for i in range(4)]
    m1 = MrDMD(dmd=l, max_level=3).fit(X=X).reconstructed_data
    m2 = MrDMD(DMD(svd_rank=4), max_level=3).fit(X=X).reconstructed_data
    assert_allclose(m1, m2)

@pytest.mark.parametrize("X", data_backends)
def test_quantitative_tuple_dmd(X):
    l = tuple(DMD(svd_rank=4) for i in range(4))
    m1 = MrDMD(dmd=l, max_level=3).fit(X=X).reconstructed_data
    m2 = MrDMD(DMD(svd_rank=4), max_level=3).fit(X=X).reconstructed_data
    assert_allclose(m1, m2)

@pytest.mark.parametrize("X", data_backends)
def test_quantitative_func_dmd(X):
    def f(*args):
        return FbDMD(svd_rank=4)
    m1 = MrDMD(dmd=f, max_level=4).fit(X).reconstructed_data
    m2 = MrDMD(dmd=FbDMD(svd_rank=4), max_level=4).fit(X).reconstructed_data
    assert_allclose(m1, m2)
