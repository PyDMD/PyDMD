from copy import deepcopy

import numpy as np
import pytest
from ezyrb import POD, RBF
from pytest import raises, param

from pydmd import (CDMD, DMD, HODMD, DMDBase, DMDc, FbDMD, HankelDMD, MrDMD,
                   OptDMD, ParametricDMD, SpDMD)
from pydmd.dmd_modes_tuner import (ModesSelectors, ModesTuner, select_modes,
                                   selectors, stabilize_modes)

# 15 snapshot with 400 data. The matrix is 400x15 and it contains
# the following data: f1 + f2 where
# f1 = lambda x,t: sech(x+3)*(1.*np.exp(1j*2.3*t))
# f2 = lambda x,t: (sech(x)*np.tanh(x))*(2.*np.exp(1j*2.8*t))
sample_data = np.load('tests/test_datasets/input_sample.npy')

class FakeDMDOperator:
    def __init__(self):
        self.as_array = np.ones(10)

def test_select_modes():
    def stable_modes(dmd_object):
        toll = 1e-3
        return np.abs(np.abs(dmd_object.eigs) - 1) < toll
    dmd = DMD(svd_rank=10)
    dmd.fit(sample_data)
    dmdc = deepcopy(dmd)

    exp = dmd.reconstructed_data
    select_modes(dmd, stable_modes)
    np.testing.assert_array_almost_equal(exp, dmd.reconstructed_data)

    assert len(dmd.eigs) < len(dmdc.eigs)
    assert dmd.modes.shape[1] < dmdc.modes.shape[1]
    assert len(dmd.amplitudes) < len(dmdc.amplitudes)

def test_select_modes_nullified_indexes():
    def stable_modes(dmd_object):
        toll = 1e-3
        return np.abs(np.abs(dmd_object.eigs) - 1) < toll
    dmd = DMD(svd_rank=10)
    dmd.fit(sample_data)
    dmdc = deepcopy(dmd)

    _, cut_indexes = select_modes(dmd, stable_modes, nullify_amplitudes=False, return_indexes=True)
    noncut_indexes = list(set(range(len(dmdc.eigs))) - set(cut_indexes))

    assert dmd.amplitudes.shape == dmdc[noncut_indexes].amplitudes.shape

def test_select_modes_index():
    fake_dmd_operator = FakeDMDOperator()
    fake_dmd = DMD()

    eigs = np.array([1 + 1e-4, 2, 1 - 1e-2, 5, 1, 1 - 0.5*1e-3])

    setattr(fake_dmd_operator, 'eigenvalues', eigs)
    setattr(fake_dmd_operator, '_eigenvalues', eigs)
    setattr(fake_dmd_operator, '_Lambda', np.zeros(len(eigs)))
    # these are DMD eigenvectors, but we do not care in this test
    setattr(fake_dmd_operator, '_eigenvectors', np.zeros((1, len(eigs))))
    setattr(fake_dmd_operator, '_modes', np.zeros((1, len(eigs))))
    setattr(fake_dmd_operator, 'modes', np.zeros((1, len(eigs))))

    setattr(fake_dmd, '_Atilde', fake_dmd_operator)
    setattr(fake_dmd, '_b', np.zeros(len(eigs)))

    _, idx = select_modes(fake_dmd, ModesSelectors.stable_modes(max_distance_from_unity=1e-3), return_indexes=True)
    np.testing.assert_array_equal(idx, [1,2,3])

    assert fake_dmd.modes.shape[1] == 3
    assert len(fake_dmd.eigs) == 3
    assert len(fake_dmd.amplitudes) == 3

def test_select_modes_index_and_deepcopy():
    fake_dmd_operator = FakeDMDOperator()
    fake_dmd = DMD()

    eigs = np.array([1 + 1e-4, 2, 1 - 1e-2, 5, 1, 1 - 0.5*1e-3])

    setattr(fake_dmd_operator, '_eigenvalues', eigs)
    setattr(fake_dmd_operator, 'eigenvalues', eigs)
    setattr(fake_dmd_operator, '_Lambda', np.zeros(len(eigs)))
    # these are DMD eigenvectors, but we do not care in this test
    setattr(fake_dmd_operator, '_eigenvectors', np.zeros((1, len(eigs))))
    setattr(fake_dmd_operator, '_modes', np.zeros((1, len(eigs))))
    setattr(fake_dmd_operator, 'modes', np.zeros((1, len(eigs))))
    setattr(fake_dmd, '_b', np.zeros(len(eigs)))

    setattr(fake_dmd, '_Atilde', fake_dmd_operator)

    dmd2, idx = select_modes(fake_dmd, ModesSelectors.stable_modes(max_distance_from_unity=1e-3), in_place=False, return_indexes=True)
    np.testing.assert_array_equal(idx, [1,2,3])

    assert len(fake_dmd.eigs) == 6
    assert fake_dmd.modes.shape[1] == 6
    assert len(fake_dmd.amplitudes) == 6

    assert len(dmd2.eigs) == 3
    assert dmd2.modes.shape[1] == 3
    assert len(dmd2.amplitudes) == 3

def test_stable_modes_both():
    class FakeDMD:
        pass

    fake_dmd = FakeDMD()
    setattr(fake_dmd, 'eigs', np.array([1 + 1e-4, 2, 1 - 1e-2, 5, 1, 1 - 0.5*1e-3]))

    expected_result = np.array([False for _ in range(6)])
    expected_result[[0, 4, 5]] = True

    assert all(ModesSelectors.stable_modes(max_distance_from_unity=1e-3)(fake_dmd) == expected_result)

def test_stable_modes_outside_only():
    class FakeDMD:
        pass

    fake_dmd = FakeDMD()
    setattr(fake_dmd, 'eigs', np.array([1 + 1e-4, 2, 1 - 1e-2, 5, 1, 1 - 0.5*1e-3]))

    expected_result = np.array([False for _ in range(6)])
    expected_result[[0, 2, 4, 5]] = True

    assert all(ModesSelectors.stable_modes(max_distance_from_unity_outside=1e-3)(fake_dmd) == expected_result)

def test_stable_modes_inside_only():
    class FakeDMD:
        pass

    fake_dmd = FakeDMD()
    setattr(fake_dmd, 'eigs', np.array([1 + 1e-4, 2, 1 - 1e-2, 5, 1, 1 - 0.5*1e-3]))

    expected_result = np.array([False for _ in range(6)])
    expected_result[[0, 1, 3, 4, 5]] = True

    assert all(ModesSelectors.stable_modes(max_distance_from_unity_inside=1e-3)(fake_dmd) == expected_result)

def test_stable_modes_errors():
    with raises(ValueError):
        ModesSelectors.stable_modes()
    with raises(ValueError):
        ModesSelectors.stable_modes(max_distance_from_unity=1.e-2, max_distance_from_unity_inside=1.e-3)
    with raises(ValueError):
        ModesSelectors.stable_modes(max_distance_from_unity=1.e-2, max_distance_from_unity_outside=1.e-3)

def test_threshold():
    class FakeDMD:
        pass

    fake_dmd = FakeDMD()
    setattr(fake_dmd, 'eigs', np.array([complex(1, 1e-4), 2, complex(1, 1e-2), 5, 1, complex(1, 5*1e-2)]))

    expected_result = np.array([False for _ in range(6)])
    expected_result[[1, 5]] = True

    assert all(ModesSelectors.threshold(1+1.e-3, 2+1.e-10)(fake_dmd) == expected_result)

def test_compute_integral_contribution():
    np.testing.assert_almost_equal(ModesSelectors._compute_integral_contribution(
        np.array([5,0,0,1]), np.array([1,-2,3,-5,6])
    ), 442, decimal=1)

def test_integral_contribution():
    class FakeDMD:
        pass

    fake_dmd = FakeDMD()
    setattr(fake_dmd, 'dynamics', np.array([[i for _ in range(10)] for i in range(4)]))
    setattr(fake_dmd, 'modes', np.ones((20, 4)))
    setattr(fake_dmd, 'dmd_time', None)
    setattr(fake_dmd, 'original_time', None)

    expected_result = np.array([False for _ in range(4)])
    expected_result[[2, 3]] = True

    assert all(ModesSelectors.integral_contribution(2)(fake_dmd) == expected_result)

def test_integral_contribution_reconstruction():
    dmd = DMD(svd_rank=10)
    dmd.fit(sample_data)
    exp = dmd.reconstructed_data
    select_modes(dmd, ModesSelectors.integral_contribution(2))
    np.testing.assert_array_almost_equal(exp, dmd.reconstructed_data)

def test_stabilize_modes():
    dmd = DMD()
    fake_dmd_operator = FakeDMDOperator()

    eigs = np.array([complex(0.3, 0.2), complex(0.8,0.5), 1, complex(1,1.e-2), 2, complex(2,1.e-2)])
    amplitudes = np.array([1,2,3,4,5,6], dtype=complex)

    setattr(fake_dmd_operator, '_eigenvalues', eigs)
    setattr(fake_dmd_operator, 'eigenvalues', eigs)
    setattr(dmd, '_Atilde', fake_dmd_operator)
    setattr(fake_dmd_operator, 'modes', np.zeros((1, len(eigs))))

    setattr(dmd, '_b', amplitudes)

    stabilize_modes(dmd, 0.8, 1.2)
    np.testing.assert_array_almost_equal(
        dmd.eigs,
        np.array([complex(0.3, 0.2), complex(0.8,0.5) / abs(complex(0.8,0.5)),
            1, complex(1,1.e-2) / abs(complex(1,1.e-2)), 2, complex(2,1.e-2)]))

    np.testing.assert_array_almost_equal(
        dmd.amplitudes,
        np.array([1, 2*abs(complex(0.8,0.5)), 3, 4*abs(complex(1,1.e-2)), 5, 6]))

def test_stabilize_modes_index():
    dmd = DMD()
    fake_dmd_operator = FakeDMDOperator()

    eigs = np.array([complex(0.3, 0.2), complex(0.8,0.5), 1, complex(1,1.e-2), 2, complex(2,1.e-2)])
    amplitudes = np.array([1,2,3,4,5,6], dtype=complex)

    setattr(fake_dmd_operator, '_eigenvalues', eigs)
    setattr(fake_dmd_operator, 'eigenvalues', eigs)
    setattr(fake_dmd_operator, 'modes', np.zeros((1, len(eigs))))
    setattr(dmd, '_Atilde', fake_dmd_operator)

    setattr(dmd, '_b', amplitudes)

    _, indexes = stabilize_modes(dmd, 0.8, 1.2, return_indexes=True)

    np.testing.assert_array_almost_equal(
        dmd.eigs,
        np.array([complex(0.3, 0.2), complex(0.8,0.5) / abs(complex(0.8,0.5)),
            1, complex(1,1.e-2) / abs(complex(1,1.e-2)), 2, complex(2,1.e-2)]))

    np.testing.assert_array_almost_equal(
        dmd.amplitudes,
        np.array([1, 2*abs(complex(0.8,0.5)), 3, 4*abs(complex(1,1.e-2)), 5, 6]))

    np.testing.assert_almost_equal(indexes, [1,2,3])

def test_stabilize_modes_index_deepcopy():
    dmd = DMD()
    fake_dmd_operator = FakeDMDOperator()

    eigs = np.array([complex(0.3, 0.2), complex(0.8,0.5), 1, complex(1,1.e-2), 2, complex(2,1.e-2)])
    amplitudes = np.array([1,2,3,4,5,6], dtype=complex)

    setattr(fake_dmd_operator, '_eigenvalues', eigs)
    setattr(fake_dmd_operator, 'eigenvalues', eigs)
    setattr(fake_dmd_operator, 'modes', np.zeros((1, len(eigs))))
    setattr(dmd, '_Atilde', fake_dmd_operator)

    setattr(dmd, '_b', amplitudes)

    dmd2, indexes = stabilize_modes(dmd, 0.8, 1.2, in_place=False, return_indexes=True)

    np.testing.assert_array_almost_equal(
        dmd2.eigs,
        np.array([complex(0.3, 0.2), complex(0.8,0.5) / abs(complex(0.8,0.5)),
            1, complex(1,1.e-2) / abs(complex(1,1.e-2)), 2, complex(2,1.e-2)]))

    np.testing.assert_array_almost_equal(
        dmd2.amplitudes,
        np.array([1, 2*abs(complex(0.8,0.5)), 3, 4*abs(complex(1,1.e-2)), 5, 6]))

    np.testing.assert_array_almost_equal(
        dmd.eigs,
        np.array([complex(0.3, 0.2), complex(0.8,0.5),
            1, complex(1,1.e-2), 2, complex(2,1.e-2)]))

    np.testing.assert_array_almost_equal(
        dmd.amplitudes,
        np.array([1, 2, 3, 4, 5, 6]))

    np.testing.assert_almost_equal(indexes, [1,2,3])

# test that the dmd given to ModesTuner is copied with deepcopy
def test_modes_tuner_copy():
    class FakeDMD:
        pass

    fake_dmd = FakeDMD()
    setattr(fake_dmd, 'eigs', np.array([complex(1, 1e-4), 2, complex(1, 1e-2), 5, 1, complex(1, 5*1e-2)]))

    ModesTuner(fake_dmd)._dmds[0].eigs[1] = 0
    assert fake_dmd.eigs[1] == 2

# assert that passing a scalar DMD (i.e. no list) causes ModesTuner to return
# only scalar DMD instances
def test_modes_tuner_scalar_input():
    class FakeDMD:
        pass

    fake_dmd = FakeDMD()
    setattr(fake_dmd, 'eigs', np.array([complex(1, 1e-4), 2, complex(1, 1e-2), 5, 1, complex(1, 5*1e-2)]))

    mt = ModesTuner(fake_dmd, in_place=True)
    assert mt.get() == fake_dmd
    assert isinstance(mt.copy(), FakeDMD)

def test_modes_tuner_list_input():
    class FakeDMD:
        pass

    def cook_fake_dmd():
        fake_dmd = FakeDMD()
        setattr(fake_dmd, 'eigs', np.array([complex(1, 1e-4), 2, complex(1, 1e-2), 5, 1, complex(1, 5*1e-2)]))
        return fake_dmd

    dmd1 = cook_fake_dmd()
    dmd2 = cook_fake_dmd()

    mt = ModesTuner([dmd1, dmd2], in_place=True)
    assert isinstance(mt.get(), list)
    assert mt.get()[0] == dmd1
    assert mt.get()[1] == dmd2

    assert isinstance(mt.copy(), list)
    assert len(mt.copy()) == 2

def test_modes_tuner_get():
    class FakeDMD:
        pass

    fake_dmd = FakeDMD()
    setattr(fake_dmd, 'eigs', np.array([complex(1, 1e-4), 2, complex(1, 1e-2), 5, 1, complex(1, 5*1e-2)]))

    mtuner = ModesTuner(fake_dmd)

    eigs = mtuner.get().eigs
    mtuner._dmds[0].eigs[1] = 0
    assert eigs[1] == 0

def test_modes_tuner_secure_copy():
    class FakeDMD:
        pass

    fake_dmd = FakeDMD()
    setattr(fake_dmd, 'eigs', np.array([complex(1, 1e-4), 2, complex(1, 1e-2), 5, 1, complex(1, 5*1e-2)]))

    mtuner = ModesTuner(fake_dmd)

    eigs = mtuner.copy().eigs
    mtuner._dmds[0].eigs[1] = 0
    assert eigs[1] == 2

def test_modes_tuner_inplace():
    class FakeDMD:
        pass

    fake_dmd = FakeDMD()
    setattr(fake_dmd, 'eigs', np.array([complex(1, 1e-4), 2, complex(1, 1e-2), 5, 1, complex(1, 5*1e-2)]))

    mtuner = ModesTuner(fake_dmd, in_place=True)
    assert mtuner.get() == fake_dmd

    mtuner._dmds[0].eigs[1] = 0
    assert fake_dmd.eigs[1] == 0

def test_modes_tuner_inplace_list():
    class FakeDMD:
        pass

    fake_dmd = FakeDMD()
    setattr(fake_dmd, 'eigs', np.array([complex(1, 1e-4), 2, complex(1, 1e-2), 5, 1, complex(1, 5*1e-2)]))
    fake_dmd2 = FakeDMD()
    setattr(fake_dmd, 'eigs', np.array([complex(1, 1e-4), 3, complex(1, 1e-2), 5, 1, complex(1, 5*1e-2)]))

    mtuner = ModesTuner([fake_dmd, fake_dmd2], in_place=True)
    assert mtuner.get()[0] == fake_dmd
    assert mtuner.get()[1] == fake_dmd2

    mtuner._dmds[0].eigs[1] = 0
    assert fake_dmd.eigs[1] == 0

def test_modes_tuner_select_raises():
    class FakeDMD:
        pass

    fake_dmd = FakeDMD()
    setattr(fake_dmd, 'eigs', np.array([complex(1, 1e-4), 2, complex(1, 1e-2), 5, 1, complex(1, 5*1e-2)]))

    with raises(ValueError):
        ModesTuner(fake_dmd).select('ciauu')
    with raises(ValueError):
        ModesTuner(fake_dmd).select(2)

def test_modes_tuner_select():
    fake_dmd_operator = FakeDMDOperator()
    fake_dmd = DMD()

    eigs = np.array([1 + 1e-4, 2, 1 - 1e-2, 5, 1, 1 - 0.5*1e-3])

    setattr(fake_dmd_operator, 'eigenvalues', eigs)
    setattr(fake_dmd_operator, '_eigenvalues', eigs)
    setattr(fake_dmd_operator, '_Lambda', np.zeros(len(eigs)))
    # these are DMD eigenvectors, but we do not care in this test
    setattr(fake_dmd_operator, '_eigenvectors', np.zeros((1, len(eigs))))
    setattr(fake_dmd_operator, '_modes', np.zeros((1, len(eigs))))
    setattr(fake_dmd_operator, 'modes', np.zeros((1, len(eigs))))
    setattr(fake_dmd, '_b', np.zeros(len(eigs)))

    setattr(fake_dmd, '_Atilde', fake_dmd_operator)

    mtuner = ModesTuner(fake_dmd)
    mtuner.select('stable_modes', max_distance_from_unity=1e-3)
    dmd = mtuner.get()

    assert len(dmd.eigs) == 3
    assert len(dmd.amplitudes) == 3
    assert dmd.modes.shape[1] == 3

def test_modes_tuner_stabilize():
    dmd = DMD()
    fake_dmd_operator = FakeDMDOperator()

    eigs = np.array([complex(0.3, 0.2), complex(0.8,0.5), 1, complex(1,1.e-2), 2, complex(2,1.e-2)])
    amplitudes = np.array([1,2,3,4,5,6], dtype=complex)

    setattr(fake_dmd_operator, '_eigenvalues', eigs)
    setattr(fake_dmd_operator, 'eigenvalues', eigs)
    setattr(fake_dmd_operator, 'modes', np.zeros((1, len(eigs))))
    setattr(dmd, '_Atilde', fake_dmd_operator)

    setattr(dmd, '_b', amplitudes)

    mtuner = ModesTuner(dmd)
    mtuner.stabilize(inner_radius=0.8, outer_radius=1.2)
    dmd = mtuner.get()

    np.testing.assert_array_almost_equal(
        dmd.eigs,
        np.array([complex(0.3, 0.2), complex(0.8,0.5) / abs(complex(0.8,0.5)),
            1, complex(1,1.e-2) / abs(complex(1,1.e-2)), 2, complex(2,1.e-2)]))

    np.testing.assert_array_almost_equal(
        dmd.amplitudes,
        np.array([1, 2*abs(complex(0.8,0.5)), 3, 4*abs(complex(1,1.e-2)), 5, 6]))

def test_modes_tuner_stabilize_multiple():
    def cook_fake_dmd():
        dmd = DMD()
        fake_dmd_operator = FakeDMDOperator()

        eigs = np.array([complex(0.3, 0.2), complex(0.8,0.5), 1, complex(1,1.e-2), 2, complex(2,1.e-2)])
        amplitudes = np.array([1,2,3,4,5,6], dtype=complex)

        setattr(fake_dmd_operator, '_eigenvalues', eigs)
        setattr(fake_dmd_operator, 'eigenvalues', eigs)
        setattr(fake_dmd_operator, 'modes', np.zeros((1, len(eigs))))
        setattr(dmd, '_Atilde', fake_dmd_operator)

        setattr(dmd, '_b', amplitudes)

        return dmd

    dmd1 = cook_fake_dmd()
    dmd2 = cook_fake_dmd()
    dmd3 = cook_fake_dmd()

    mtuner = ModesTuner([dmd1, dmd2, dmd3])
    mtuner.stabilize(inner_radius=0.8, outer_radius=1.2)
    dmds = mtuner.get()

    assert isinstance(dmds, list)

    for dmd in dmds:
        np.testing.assert_array_almost_equal(
            dmd.eigs,
            np.array([complex(0.3, 0.2), complex(0.8,0.5) / abs(complex(0.8,0.5)),
                1, complex(1,1.e-2) / abs(complex(1,1.e-2)), 2, complex(2,1.e-2)]))

        np.testing.assert_array_almost_equal(
            dmd.amplitudes,
            np.array([1, 2*abs(complex(0.8,0.5)), 3, 4*abs(complex(1,1.e-2)), 5, 6]))

def test_modes_tuner_subset():
    def cook_fake_dmd():
        dmd = DMD()
        fake_dmd_operator = FakeDMDOperator()

        eigs = np.array([complex(0.3, 0.2), complex(0.8,0.5), 1, complex(1,1.e-2), 2, complex(2,1.e-2)])
        amplitudes = np.array([1,2,3,4,5,6], dtype=complex)

        setattr(fake_dmd_operator, '_eigenvalues', eigs)
        setattr(fake_dmd_operator, 'eigenvalues', eigs)
        setattr(fake_dmd_operator, 'modes', np.zeros((1, len(eigs))))
        setattr(dmd, '_Atilde', fake_dmd_operator)

        setattr(dmd, '_b', amplitudes)

        return dmd

    dmd1 = cook_fake_dmd()
    dmd2 = cook_fake_dmd()
    dmd3 = cook_fake_dmd()

    mtuner = ModesTuner([dmd1, dmd2, dmd3], in_place=True)
    assert len(mtuner.subset([0,2]).get()) == 2
    assert mtuner.subset([0,2]).get()[0] == dmd1
    assert mtuner.subset([0,2]).get()[1] == dmd3

    mtuner = ModesTuner([dmd1, dmd2, dmd3], in_place=False)
    assert len(mtuner.subset([0,2]).get()) == 2
    assert mtuner.subset([0,2]).get()[0] == mtuner._dmds[0]
    assert mtuner.subset([0,2]).get()[1] == mtuner._dmds[2]

def test_modes_tuner_stabilize_multiple_subset():
    def cook_fake_dmd():
        dmd = DMD()
        fake_dmd_operator = FakeDMDOperator()

        eigs = np.array([complex(0.3, 0.2), complex(0.8,0.5), 1, complex(1,1.e-2), 2, complex(2,1.e-2)])
        amplitudes = np.array([1,2,3,4,5,6], dtype=complex)

        setattr(fake_dmd_operator, '_eigenvalues', eigs)
        setattr(fake_dmd_operator, 'eigenvalues', eigs)
        setattr(fake_dmd_operator, 'modes', np.zeros((1, len(eigs))))
        setattr(dmd, '_Atilde', fake_dmd_operator)

        setattr(dmd, '_b', amplitudes)

        return dmd

    dmd1 = cook_fake_dmd()
    dmd2 = cook_fake_dmd()
    dmd3 = cook_fake_dmd()

    mtuner = ModesTuner([dmd1, dmd2, dmd3])
    mtuner.subset([0,2]).stabilize(inner_radius=0.8, outer_radius=1.2)
    dmds = mtuner.get()

    assert len(dmds) == 3

    for i in range(3):
        if i == 1:
            continue
        np.testing.assert_array_almost_equal(
            dmds[i].eigs,
            np.array([complex(0.3, 0.2), complex(0.8,0.5) / abs(complex(0.8,0.5)),
                1, complex(1,1.e-2) / abs(complex(1,1.e-2)), 2, complex(2,1.e-2)]))
        np.testing.assert_array_almost_equal(
            dmds[i].amplitudes,
            np.array([1, 2*abs(complex(0.8,0.5)), 3, 4*abs(complex(1,1.e-2)), 5, 6]))

    np.testing.assert_array_almost_equal(
            dmds[1].eigs,
            np.array([complex(0.3, 0.2), complex(0.8,0.5), 1, complex(1,1.e-2), 2, complex(2,1.e-2)]))
    np.testing.assert_array_almost_equal(
        dmds[1].amplitudes,
        np.array([1,2,3,4,5,6], dtype=complex))

def test_modes_tuner_index_scalar_dmd_raises():
    def stable_modes(dmd_object):
        toll = 1e-3
        return np.abs(np.abs(dmd_object.eigs) - 1) < toll
    dmd = DMD(svd_rank=10)
    dmd.fit(sample_data)

    with raises(ValueError):
        ModesTuner(dmd).subset([0])

def test_modes_tuner_selectors():
    assert selectors['module_threshold'] == ModesSelectors.threshold
    assert selectors['stable_modes'] == ModesSelectors.stable_modes
    assert selectors['integral_contribution'] == ModesSelectors.integral_contribution

@pytest.mark.parametrize(
    "dmd",
    [
        param(CDMD(svd_rank=-1), id="CDMD"),
        param(DMD(svd_rank=-1), id="DMD"),
        param(DMDc(svd_rank=-1), id="DMDc"),
        param(FbDMD(svd_rank=-1), id="FbDMD"),
        param(HankelDMD(svd_rank=-1, d=3), id="HankelDMD"),
        param(HODMD(svd_rank=-1, d=3), id="HODMD"),
    ],
)
def test_modes_selector_all_dmd_types(dmd):
    print('--------------------------- {} ---------------------------'.format(type(dmd)))
    if isinstance(dmd, ParametricDMD):
        repeated = np.repeat(sample_data[None], 10, axis=0)
        dmd.fit(repeated + np.random.rand(*repeated.shape), np.ones(10))
    elif isinstance(dmd, DMDc):
        snapshots = np.array([[4, 2, 1, .5, .25], [7, .7, .07, .007, .0007]])
        u = np.array([-4, -2, -1, -.5])
        B = np.array([[1, 0]]).T
        dmd.fit(snapshots, u, B)
    else:
        dmd.fit(sample_data)

    ModesTuner(dmd, in_place=True).select('integral_contribution', n=3).stabilize(1-1.e-3)
