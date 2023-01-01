from pydmd.dmdbase import DMDBase
from pydmd import DMD
import matplotlib.pyplot as plt
import numpy as np
from pytest import raises

# 15 snapshot with 400 data. The matrix is 400x15 and it contains
# the following data: f1 + f2 where
# f1 = lambda x,t: sech(x+3)*(1.*np.exp(1j*2.3*t))
# f2 = lambda x,t: (sech(x)*np.tanh(x))*(2.*np.exp(1j*2.8*t))
sample_data = np.load('tests/test_datasets/input_sample.npy')


def test_svd_rank_default():
    dmd = DMDBase()
    assert dmd.svd_rank == 0

def test_svd_rank():
    dmd = DMDBase(svd_rank=3)
    assert dmd.svd_rank == 3

def test_tlsq_rank_default():
    dmd = DMDBase()
    assert dmd.tlsq_rank == 0

def test_tlsq_rank():
    dmd = DMDBase(tlsq_rank=2)
    assert dmd.tlsq_rank == 2

def test_exact_default():
    dmd = DMDBase()
    assert dmd.exact == False

def test_exact():
    dmd = DMDBase(exact=True)
    assert dmd.exact == True

def test_opt_default():
    dmd = DMDBase()
    assert dmd.opt == False

def test_opt():
    dmd = DMDBase(opt=True)
    assert dmd.opt == True

def test_fit_resets_mode_activation_bitmask():
    dmd = DMDBase(exact=False)
    with raises(NotImplementedError):
        dmd.fit(sample_data)

def test_plot_eigs():
    dmd = DMDBase()
    with raises(ValueError):
        dmd.plot_eigs(show_axes=True, show_unit_circle=True)

def test_plot_eigs_narrowview_empty():
    dmd = DMDBase()
    # max/min throws an error if the array is empty (max used on empty
    # array)
    dmd.operator._eigenvalues = np.array([], dtype=complex)
    with raises(ValueError):
        dmd.plot_eigs(show_axes=False, narrow_view=True, dpi=200)

def test_plot_modes_2D():
    dmd = DMDBase()
    with raises(ValueError):
        dmd.plot_modes_2D()

def test_plot_snaps_2D():
    dmd = DMDBase()
    with raises(ValueError):
        dmd.plot_snapshots_2D()

def test_advanced_snapshot_parameter2():
    dmd = DMDBase(opt=5)
    assert dmd.opt == 5

def test_translate_tpow_positive():
    dmd = DMDBase(opt=4)

    assert dmd._translate_eigs_exponent(10) == 6
    assert dmd._translate_eigs_exponent(0) == -4

def test_translate_tpow_negative():
    dmd = DMDBase(opt=-1)
    dmd._snapshots = sample_data

    assert dmd._translate_eigs_exponent(10) == 10 - (sample_data.shape[1] - 1)
    assert dmd._translate_eigs_exponent(0) == 1 - sample_data.shape[1]

def test_translate_tpow_vector():
    dmd = DMDBase(opt=-1)
    dmd._snapshots = sample_data

    tpow = np.ndarray([0,1,2,3,5,6,7,11])
    for idx,x in enumerate(dmd._translate_eigs_exponent(tpow)):
        assert x == dmd._translate_eigs_exponent(tpow[idx])

def test_sorted_eigs_default():
    dmd = DMDBase()
    assert dmd.operator._sorted_eigs == False

def test_sorted_eigs_param():
    dmd = DMDBase(sorted_eigs='real')
    assert dmd.operator._sorted_eigs == 'real'

def test_enforce_ratio_y():
    dmd = DMDBase()
    supx, infx, supy, infy = dmd._enforce_ratio(10, 20, 10, 0, 0)

    dx = supx - infx
    dy = supy - infy
    np.testing.assert_almost_equal(max(dx,dy) / min(dx,dy), 10, decimal=6)

def test_enforce_ratio_x():
    dmd = DMDBase()
    supx, infx, supy, infy = dmd._enforce_ratio(10, 0, 0, 20, 10)

    dx = supx - infx
    dy = supy - infy
    np.testing.assert_almost_equal(max(dx,dy) / min(dx,dy), 10, decimal=6)


def test_plot_limits_narrow():
    dmd = DMDBase()
    dmd.operator._eigenvalues = np.array([complex(1,2), complex(-1,-2)])
    dmd.operator._modes = np.array(np.ones((10,2)))

    tp = dmd._plot_limits(True)

    assert len(tp) == 4

    supx, infx, supy, infy = tp
    assert supx == 1.05
    assert infx == -1.05
    assert supy == 2.05
    assert infy == -2.05

def test_plot_limits():
    dmd = DMDBase()
    dmd.operator._eigenvalues = np.array([complex(-2,2), complex(3,-3)])
    dmd.operator._modes = np.array(np.ones((10,2)))

    limit = dmd._plot_limits(False)
    assert limit == 5

def test_dmd_time_wrong_key():
    dmd = DMD(svd_rank=10)
    dmd.fit(sample_data)

    with raises(KeyError):
        dmd.dmd_time['tstart'] = 10
