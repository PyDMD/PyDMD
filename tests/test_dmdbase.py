from unittest import TestCase
from pydmd.dmdbase import DMDBase
from pydmd import DMD
import matplotlib.pyplot as plt
import numpy as np

# 15 snapshot with 400 data. The matrix is 400x15 and it contains
# the following data: f1 + f2 where
# f1 = lambda x,t: sech(x+3)*(1.*np.exp(1j*2.3*t))
# f2 = lambda x,t: (sech(x)*np.tanh(x))*(2.*np.exp(1j*2.8*t))
sample_data = np.load('tests/test_datasets/input_sample.npy')


class TestDmdBase(TestCase):
    def test_svd_rank_default(self):
        dmd = DMDBase()
        assert dmd.svd_rank == 0

    def test_svd_rank(self):
        dmd = DMDBase(svd_rank=3)
        assert dmd.svd_rank == 3

    def test_tlsq_rank_default(self):
        dmd = DMDBase()
        assert dmd.tlsq_rank == 0

    def test_tlsq_rank(self):
        dmd = DMDBase(tlsq_rank=2)
        assert dmd.tlsq_rank == 2

    def test_exact_default(self):
        dmd = DMDBase()
        assert dmd.exact == False

    def test_exact(self):
        dmd = DMDBase(exact=True)
        assert dmd.exact == True

    def test_opt_default(self):
        dmd = DMDBase()
        assert dmd.opt == False

    def test_opt(self):
        dmd = DMDBase(opt=True)
        assert dmd.opt == True

    def test_fit(self):
        dmd = DMDBase(exact=False)
        with self.assertRaises(NotImplementedError):
            dmd.fit(sample_data)

    def test_plot_eigs(self):
        dmd = DMDBase()
        with self.assertRaises(ValueError):
            dmd.plot_eigs(show_axes=True, show_unit_circle=True)

    def test_plot_eigs_narrowview_empty(self):
        dmd = DMDBase()
        # max/min throws an error if the array is empty (max used on empty
        # array)
        dmd.operator._eigenvalues = np.array([], dtype=complex)
        with self.assertRaises(ValueError):
            dmd.plot_eigs(show_axes=False, narrow_view=True, dpi=200)

    def test_plot_modes_2D(self):
        dmd = DMDBase()
        with self.assertRaises(ValueError):
            dmd.plot_modes_2D()

    def test_plot_snaps_2D(self):
        dmd = DMDBase()
        with self.assertRaises(ValueError):
            dmd.plot_snapshots_2D()

    def test_advanced_snapshot_parameter2(self):
        dmd = DMDBase(opt=5)
        assert dmd.opt == 5

    def test_translate_tpow_positive(self):
        dmd = DMDBase(opt=4)

        assert dmd._translate_eigs_exponent(10) == 6
        assert dmd._translate_eigs_exponent(0) == -4

    def test_translate_tpow_negative(self):
        dmd = DMDBase(opt=-1)
        dmd._snapshots = sample_data

        assert dmd._translate_eigs_exponent(10) == 10 - (sample_data.shape[1] - 1)
        assert dmd._translate_eigs_exponent(0) == 1 - sample_data.shape[1]

    def test_translate_tpow_vector(self):
        dmd = DMDBase(opt=-1)
        dmd._snapshots = sample_data

        tpow = np.ndarray([0,1,2,3,5,6,7,11])
        for idx,x in enumerate(dmd._translate_eigs_exponent(tpow)):
            assert x == dmd._translate_eigs_exponent(tpow[idx])

    def test_sorted_eigs_default(self):
        dmd = DMDBase()
        assert dmd.operator._sorted_eigs == False

    def test_sorted_eigs_param(self):
        dmd = DMDBase(sorted_eigs='real')
        assert dmd.operator._sorted_eigs == 'real'

    def test_select_modes(self):
        def stable_modes(dmd_object):
            toll = 1e-3
            return np.abs(np.abs(dmd_object.eigs) - 1) < toll
        dmd = DMD(svd_rank=10)
        dmd.fit(sample_data)
        exp = dmd.reconstructed_data
        dmd.select_modes(stable_modes)
        np.testing.assert_array_almost_equal(exp, dmd.reconstructed_data)

    def test_stable_modes_both(self):
        class FakeDMD:
            pass

        fake_dmd = FakeDMD()
        setattr(fake_dmd, 'eigs', np.array([1 + 1e-4, 2, 1 - 1e-2, 5, 1, 1 - 0.5*1e-3]))

        expected_result = np.array([False for _ in range(6)])
        expected_result[[0, 4, 5]] = True

        assert all(DMDBase.ModesSelectors.stable_modes(max_distance_from_unity=1e-3)(fake_dmd) == expected_result)

    def test_stable_modes_outside_only(self):
        class FakeDMD:
            pass

        fake_dmd = FakeDMD()
        setattr(fake_dmd, 'eigs', np.array([1 + 1e-4, 2, 1 - 1e-2, 5, 1, 1 - 0.5*1e-3]))

        expected_result = np.array([False for _ in range(6)])
        expected_result[[0, 2, 4, 5]] = True

        assert all(DMDBase.ModesSelectors.stable_modes(max_distance_from_unity_outside=1e-3)(fake_dmd) == expected_result)

    def test_stable_modes_inside_only(self):
        class FakeDMD:
            pass

        fake_dmd = FakeDMD()
        setattr(fake_dmd, 'eigs', np.array([1 + 1e-4, 2, 1 - 1e-2, 5, 1, 1 - 0.5*1e-3]))

        expected_result = np.array([False for _ in range(6)])
        expected_result[[0, 1, 3, 4, 5]] = True

        assert all(DMDBase.ModesSelectors.stable_modes(max_distance_from_unity_inside=1e-3)(fake_dmd) == expected_result)

    def test_stable_modes_errors(self):
        with self.assertRaises(ValueError):
            DMDBase.ModesSelectors.stable_modes()
        with self.assertRaises(ValueError):
            DMDBase.ModesSelectors.stable_modes(max_distance_from_unity=1.e-2, max_distance_from_unity_inside=1.e-3)
        with self.assertRaises(ValueError):
            DMDBase.ModesSelectors.stable_modes(max_distance_from_unity=1.e-2, max_distance_from_unity_outside=1.e-3)

    def test_threshold(self):
        class FakeDMD:
            pass

        fake_dmd = FakeDMD()
        setattr(fake_dmd, 'eigs', np.array([complex(1, 1e-4), 2, complex(1, 1e-2), 5, 1, complex(1, 5*1e-2)]))

        expected_result = np.array([False for _ in range(6)])
        expected_result[[1, 5]] = True

        assert all(DMDBase.ModesSelectors.threshold(1+1.e-3, 2+1.e-10)(fake_dmd) == expected_result)

    def test_compute_integral_contribution(self):
        np.testing.assert_almost_equal(DMDBase.ModesSelectors._compute_integral_contribution(
            np.array([5,0,0,1]), np.array([1,-2,3,-5,6])
        ), 442, decimal=1)

    def test_integral_contribution(self):
        class FakeDMD:
            pass

        fake_dmd = FakeDMD()
        setattr(fake_dmd, 'dynamics', np.array([[i for _ in range(10)] for i in range(4)]))
        setattr(fake_dmd, 'modes', np.ones((20, 4)))
        setattr(fake_dmd, 'dmd_time', None)
        setattr(fake_dmd, 'original_time', None)

        expected_result = np.array([False for _ in range(4)])
        expected_result[[2, 3]] = True

        assert all(DMDBase.ModesSelectors.integral_contribution(2)(fake_dmd) == expected_result)

    def test_integral_contribution_reconstruction(self):
        dmd = DMD(svd_rank=10)
        dmd.fit(sample_data)
        exp = dmd.reconstructed_data
        dmd.select_modes(DMDBase.ModesSelectors.integral_contribution(2))
        np.testing.assert_array_almost_equal(exp, dmd.reconstructed_data)

    def test_stabilize_modes(self):
        class FakeDMD:
            pass

        class FakeDMDOperator:
            pass

        dmd = DMD()
        fake_dmd_operator = FakeDMDOperator()

        eigs = np.array([complex(0.3, 0.2), complex(0.8,0.5), 1, complex(1,1.e-2), 2, complex(2,1.e-2)])
        amplitudes = np.array([1,2,3,4,5,6], dtype=complex)

        setattr(fake_dmd_operator, '_eigenvalues', eigs)
        setattr(fake_dmd_operator, 'eigenvalues', eigs)
        setattr(dmd, '_Atilde', fake_dmd_operator)

        setattr(dmd, '_b', amplitudes)

        dmd.stabilize_modes(0.8, 1.2)

        np.testing.assert_array_almost_equal(
            dmd.operator._eigenvalues,
            np.array([complex(0.3, 0.2), complex(0.8,0.5) / abs(complex(0.8,0.5)),
                1, complex(1,1.e-2) / abs(complex(1,1.e-2)), 2, complex(2,1.e-2)]))

        np.testing.assert_array_almost_equal(
            dmd._b,
            np.array([1, 2*abs(complex(0.8,0.5)), 3, 4*abs(complex(1,1.e-2)), 5, 6]))

    def test_enforce_ratio_y(self):
        dmd = DMDBase()
        supx, infx, supy, infy = dmd._enforce_ratio(10, 20, 10, 0, 0)

        dx = supx - infx
        dy = supy - infy
        np.testing.assert_almost_equal(max(dx,dy) / min(dx,dy), 10, decimal=6)

    def test_enforce_ratio_x(self):
        dmd = DMDBase()
        supx, infx, supy, infy = dmd._enforce_ratio(10, 0, 0, 20, 10)

        dx = supx - infx
        dy = supy - infy
        np.testing.assert_almost_equal(max(dx,dy) / min(dx,dy), 10, decimal=6)


    def test_plot_limits_narrow(self):
        dmd = DMDBase()
        dmd.operator._eigenvalues = np.array([complex(1,2), complex(-1,-2)])

        tp = dmd._plot_limits(True)

        assert len(tp) == 4

        supx, infx, supy, infy = tp
        assert supx == 1.05
        assert infx == -1.05
        assert supy == 2.05
        assert infy == -2.05

    def test_plot_limits(self):
        dmd = DMDBase()
        dmd.operator._eigenvalues = np.array([complex(-2,2), complex(3,-3)])

        limit = dmd._plot_limits(False)
        assert limit == 5

    def test_dmd_time_wrong_key(self):
        dmd = DMD(svd_rank=10)
        dmd.fit(sample_data)

        with self.assertRaises(KeyError):
            dmd.dmd_time['tstart'] = 10
