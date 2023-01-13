import numpy as np
from pytest import raises

from pydmd import DMD
from pydmd.dmdbase import DMDBase
from pydmd.snapshots import Snapshots

# 15 snapshot with 400 data. The matrix is 400x15 and it contains
# the following data: f1 + f2 where
# f1 = lambda x,t: sech(x+3)*(1.*np.exp(1j*2.3*t))
# f2 = lambda x,t: (sech(x)*np.tanh(x))*(2.*np.exp(1j*2.8*t))
sample_data = np.load('tests/test_datasets/input_sample.npy')


def test_svd_rank_default():
    dmd = DMDBase()
    assert dmd.operator._svd_rank == 0

def test_svd_rank():
    dmd = DMDBase(svd_rank=3)
    assert dmd.operator._svd_rank == 3

def test_tlsq_rank_default():
    dmd = DMDBase()
    assert dmd._tlsq_rank == 0

def test_tlsq_rank():
    dmd = DMDBase(tlsq_rank=2)
    assert dmd._tlsq_rank == 2

def test_exact_default():
    dmd = DMDBase()
    assert dmd.operator._exact == False

def test_exact():
    dmd = DMDBase(exact=True)
    assert dmd.operator._exact == True

def test_opt_default():
    dmd = DMDBase()
    assert dmd._opt == False

def test_opt():
    dmd = DMDBase(opt=True)
    assert dmd._opt == True

def test_fit():
    dmd = DMDBase(exact=False)
    with raises(NotImplementedError):
        dmd.fit(sample_data)

def test_advanced_snapshot_parameter2():
    dmd = DMDBase(opt=5)
    assert dmd._opt == 5

def test_translate_tpow_positive():
    dmd = DMDBase(opt=4)

    assert dmd._translate_eigs_exponent(10) == 6
    assert dmd._translate_eigs_exponent(0) == -4

def test_translate_tpow_negative():
    dmd = DMDBase(opt=-1)
    dmd._snapshots_holder = Snapshots(sample_data)

    assert dmd._translate_eigs_exponent(10) == 10 - (sample_data.shape[1] - 1)
    assert dmd._translate_eigs_exponent(0) == 1 - sample_data.shape[1]

def test_translate_tpow_vector():
    dmd = DMDBase(opt=-1)
    dmd._snapshots_holder = Snapshots(sample_data)

    tpow = np.ndarray([0,1,2,3,5,6,7,11])
    for idx,x in enumerate(dmd._translate_eigs_exponent(tpow)):
        assert x == dmd._translate_eigs_exponent(tpow[idx])

def test_sorted_eigs_default():
    dmd = DMDBase()
    assert dmd.operator._sorted_eigs == False

def test_sorted_eigs_param():
    dmd = DMDBase(sorted_eigs='real')
    assert dmd.operator._sorted_eigs == 'real'

def test_dmd_time_wrong_key():
    dmd = DMD(svd_rank=10)
    dmd.fit(sample_data)

    with raises(KeyError):
        dmd.dmd_time['tstart'] = 10
