from __future__ import division

import numpy as np
from past.utils import old_div
from pytest import raises

from pydmd import DMD, FbDMD
from pydmd.mrdmd import MrDMD


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


sample_data = create_data()

def test_max_level_threshold():
    level = 10
    dmd = DMD()
    mrdmd = MrDMD(dmd, max_level=level, max_cycles=2)
    mrdmd.fit(X=sample_data)
    lvl_threshold = int(np.log(sample_data.shape[1]/4.)/np.log(2.)) + 1
    assert lvl_threshold == mrdmd.max_level

def test_partial_time_interval():
    level = 4
    dmd = DMD()
    mrdmd = MrDMD(dmd, max_level=level, max_cycles=2)
    mrdmd.fit(X=sample_data)
    ans = {'t0': 1200, 'tend': 1400.0, 'delta': 200.0}
    assert mrdmd.partial_time_interval(3, 6) == ans

def test_partial_time_interval2():
    level = 4
    dmd = DMD()
    mrdmd = MrDMD(dmd, max_level=level, max_cycles=2)
    mrdmd.fit(X=sample_data)
    with raises(ValueError):
        mrdmd.partial_time_interval(5, 0)

def test_partial_time_interval3():
    level = 4
    dmd = DMD()
    mrdmd = MrDMD(dmd, max_level=level, max_cycles=2)
    mrdmd.fit(X=sample_data)
    with raises(ValueError):
        mrdmd.partial_time_interval(3, 8)

def test_time_window_bins():
    level = 4
    dmd = DMD()
    mrdmd = MrDMD(dmd, max_level=level, max_cycles=2)
    mrdmd.fit(X=sample_data)
    assert len(mrdmd.time_window_bins(0, 1600)) == 2**5-1

def test_time_window_bins2():
    level = 3
    dmd = DMD()
    mrdmd = MrDMD(dmd, max_level=level, max_cycles=2)
    mrdmd.fit(X=sample_data)
    expected_bins = np.array([
        [0, 0],
        [1, 0],
        [2, 0],
        [2, 1],
        [3, 1],
        [3, 2]])
    np.testing.assert_array_equal(mrdmd.time_window_bins(200, 600), expected_bins)


def test_time_window_eigs():
    level = 2
    dmd = DMD()
    mrdmd = MrDMD(dmd, max_level=level, max_cycles=1)
    mrdmd.fit(X=sample_data)
    exp =  sum([len(dmd.eigs) for dmd in mrdmd])
    assert len(mrdmd.time_window_eigs(0, 1600)) == exp

def test_time_window_frequency():
    level = 2
    dmd = DMD()
    mrdmd = MrDMD(dmd, max_level=level, max_cycles=1)
    mrdmd.fit(X=sample_data)
    exp =  sum([len(dmd.frequency) for dmd in mrdmd])
    assert len(mrdmd.time_window_frequency(0, 1600)) == exp

def test_time_window_growth_rate():
    level = 2
    dmd = DMD()
    mrdmd = MrDMD(dmd, max_level=level, max_cycles=1)
    mrdmd.fit(X=sample_data)
    exp =  sum([len(dmd.growth_rate) for dmd in mrdmd])
    assert len(mrdmd.time_window_growth_rate(0, 1600)) == exp

def test_time_window_amplitudes():
    level = 2
    dmd = DMD()
    mrdmd = MrDMD(dmd, max_level=level, max_cycles=1)
    mrdmd.fit(X=sample_data)
    exp =  sum([len(dmd.amplitudes) for dmd in mrdmd])
    assert len(mrdmd.time_window_amplitudes(0, 1600)) == exp

def test_shape_modes():
    level = 2
    dmd = DMD(svd_rank=1)
    mrdmd = MrDMD(dmd, max_level=level, max_cycles=1)
    mrdmd.fit(X=sample_data)
    assert mrdmd.modes.shape == (sample_data.shape[0], 2**(level+1) - 1)

def test_shape_dynamics():
    level = 2
    dmd = DMD(svd_rank=1)
    mrdmd = MrDMD(dmd, max_level=level, max_cycles=1)
    mrdmd.fit(X=sample_data)
    assert mrdmd.dynamics.shape == (2**(level+1) - 1, sample_data.shape[1])

def test_reconstructed_data():
    dmd = DMD(svd_rank=1)
    mrdmd = MrDMD(dmd, max_level=6, max_cycles=2)
    mrdmd.fit(X=sample_data)
    dmd_data = mrdmd.reconstructed_data
    norm_err = (old_div(
        np.linalg.norm(sample_data - dmd_data),
        np.linalg.norm(sample_data)))
    assert norm_err < 1

def test_partial_modes1():
    max_level = 5
    level = 2
    rank = 2
    dmd = DMD(svd_rank=rank)
    mrdmd = MrDMD(dmd, max_level=6, max_cycles=2)
    mrdmd.fit(X=sample_data)
    pmodes = mrdmd.partial_modes(level)
    assert pmodes.shape == (sample_data.shape[0], 2**level * rank)

def test_partial_modes2():
    max_level = 5
    level = 2
    rank = 2
    dmd = DMD(svd_rank=rank)
    mrdmd = MrDMD(dmd, max_level=6, max_cycles=2)
    mrdmd.fit(X=sample_data)
    pmodes = mrdmd.partial_modes(level, 3)
    assert pmodes.shape == (sample_data.shape[0], rank)

def test_partial_dynamics1():
    max_level = 5
    level = 2
    rank = 2
    dmd = DMD(svd_rank=rank)
    mrdmd = MrDMD(dmd, max_level=6, max_cycles=2)
    mrdmd.fit(X=sample_data)
    pdynamics = mrdmd.partial_dynamics(level)
    assert pdynamics.shape == (2**level * rank, sample_data.shape[1])

def test_partial_dynamics2():
    max_level = 5
    level = 2
    rank = 2
    dmd = DMD(svd_rank=rank)
    mrdmd = MrDMD(dmd, max_level=6, max_cycles=2)
    mrdmd.fit(X=sample_data)
    pdynamics = mrdmd.partial_dynamics(level, 3)
    assert pdynamics.shape == (rank, sample_data.shape[1] // 2**level)

def test_eigs2():
    max_level = 5
    level = 2
    rank = -1
    dmd = DMD(svd_rank=rank)
    mrdmd = MrDMD(dmd, max_level=6, max_cycles=2)
    mrdmd.fit(X=sample_data)
    assert mrdmd.eigs.ndim == 1

def test_partial_eigs1():
    max_level = 5
    level = 2
    rank = 2
    dmd = DMD(svd_rank=rank)
    mrdmd = MrDMD(dmd, max_level=max_level, max_cycles=2)
    mrdmd.fit(X=sample_data)
    peigs = mrdmd.partial_eigs(level)
    assert peigs.shape == (rank * 2**level, )

def test_partial_eigs2():
    max_level = 5
    level = 2
    rank = 2
    dmd = DMD(svd_rank=rank)
    mrdmd = MrDMD(dmd, max_level=6, max_cycles=2)
    mrdmd.fit(X=sample_data)
    peigs = mrdmd.partial_eigs(level, 3)
    assert peigs.shape == (rank, )

def test_partial_reconstructed1():
    max_level = 5
    level = 2
    rank = 2
    dmd = DMD(svd_rank=rank)
    mrdmd = MrDMD(dmd, max_level=6, max_cycles=2)
    mrdmd.fit(X=sample_data)
    pdata = mrdmd.partial_reconstructed_data(level)
    assert pdata.shape == sample_data.shape

def test_partial_reconstructed2():
    max_level = 5
    level = 2
    rank = 2
    dmd = DMD(svd_rank=rank)
    mrdmd = MrDMD(dmd, max_level=max_level, max_cycles=2)
    mrdmd.fit(X=sample_data)
    pdata = mrdmd.partial_reconstructed_data(level, 3)
    assert pdata.shape == (sample_data.shape[0], sample_data.shape[1] // 2**level)

def test_wrong_partial_reconstructed():
    max_level = 5
    level = 2
    rank = 2
    dmd = DMD(svd_rank=rank)
    mrdmd = MrDMD(dmd, max_level=max_level, max_cycles=2)
    mrdmd.fit(X=sample_data)
    with raises(ValueError):
        pdata = mrdmd.partial_reconstructed_data(max_level+1, 2)

def test_wrong_level():
    max_level = 5
    rank = 2
    dmd = DMD(svd_rank=rank)
    mrdmd = MrDMD(dmd, max_level=max_level, max_cycles=2)
    mrdmd.fit(X=sample_data)
    with raises(ValueError):
        mrdmd.partial_modes(max_level + 1)

def test_wrong_bin():
    max_level = 5
    level = 2
    dmd = DMD()
    mrdmd = MrDMD(dmd, max_level=max_level, max_cycles=2)
    mrdmd.fit(X=sample_data)
    with raises(ValueError):
        mrdmd.partial_modes(level=level, node=2**level)

def test_consistency():
    level = 5
    dmd = DMD()
    mrdmd = MrDMD(dmd, max_level=level, max_cycles=1)
    mrdmd.fit(X=sample_data)
    np.testing.assert_array_almost_equal(mrdmd.reconstructed_data, mrdmd.modes @ mrdmd.dynamics)

def test_consistency2():
    import sys

    import numpy
    numpy.set_printoptions(threshold=sys.maxsize)

    mrdmd = MrDMD(DMD(), max_level=5, max_cycles=1)
    mrdmd.fit(X=create_data(t_size=1400))

    np.testing.assert_array_almost_equal(mrdmd.reconstructed_data, mrdmd.modes @ mrdmd.dynamics)

def test_bitmask_not_implemented():
    with raises(RuntimeError):
        mrdmd = MrDMD(DMD(), max_level=5, max_cycles=1)
        mrdmd.fit(X=sample_data)
        mrdmd.modes_activation_bitmask
    with raises(RuntimeError):
        mrdmd = MrDMD(DMD(), max_level=5, max_cycles=1)
        mrdmd.fit(X=sample_data)
        mrdmd.modes_activation_bitmask = None

def test_getitem_not_implemented():
    with raises(RuntimeError):
        mrdmd = MrDMD(DMD(), max_level=5, max_cycles=1)
        mrdmd.fit(X=sample_data)
        mrdmd[1:3]

def test_one_dmd():
    m = MrDMD(dmd=DMD(svd_rank=4), max_level=5)

    for level in range(6):
        leaves = 2 ** level
        for leaf in range(leaves):
            dmd = m.dmd_tree[level, leaf]
            assert isinstance(dmd, DMD)
            assert dmd.svd_rank == 4

def test_list_dmd():
    l = [DMD(svd_rank=5-i) for i in range(4)]
    m = MrDMD(dmd=l, max_level=3)

    for level in range(4):
        leaves = 2 ** level
        for leaf in range(leaves):
            dmd = m.dmd_tree[level, leaf]
            assert isinstance(dmd, DMD)
            assert dmd.svd_rank == 5 - level

def test_tuple_dmd():
    l = tuple(DMD(svd_rank=5-i) for i in range(4))
    m = MrDMD(dmd=l, max_level=3)

    for level in range(4):
        leaves = 2 ** level
        for leaf in range(leaves):
            dmd = m.dmd_tree[level, leaf]
            assert isinstance(dmd, DMD)
            assert dmd.svd_rank == 5 - level

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
            assert dmd.svd_rank == level * leaf

def test_quantitative_list_dmd():
    l = [DMD(svd_rank=4) for i in range(4)]
    m1 = MrDMD(dmd=l, max_level=3).fit(X=sample_data).reconstructed_data
    m2 = MrDMD(DMD(svd_rank=4), max_level=3).fit(X=sample_data).reconstructed_data
    np.testing.assert_almost_equal(m1, m2)

def test_quantitative_tuple_dmd():
    l = tuple(DMD(svd_rank=4) for i in range(4))
    m1 = MrDMD(dmd=l, max_level=3).fit(X=sample_data).reconstructed_data
    m2 = MrDMD(DMD(svd_rank=4), max_level=3).fit(X=sample_data).reconstructed_data
    np.testing.assert_almost_equal(m1, m2)

def test_quantitative_func_dmd():
    def f(level, leaf):
        return FbDMD(svd_rank=4)
    m1 = MrDMD(dmd=f, max_level=4).fit(sample_data).reconstructed_data
    m2 = MrDMD(dmd=FbDMD(svd_rank=4), max_level=4).fit(sample_data).reconstructed_data
