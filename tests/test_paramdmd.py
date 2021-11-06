from unittest import TestCase
from pydmd import DMD, ParametricDMD
from ezyrb import POD, RBF
import numpy as np

def f1(x,t):
    return 1./np.cosh(x+3)*np.exp(2.3j*t)
def f2(x,t):
    return 2./np.cosh(x)*np.tanh(x)*np.exp(2.8j*t)
def f(a):
    def fnc(x,t):
        return a*f1(x,t) + (1-a)*f2(x,t)
    return fnc

params = np.linspace(0,1,10)

x = np.linspace(0,np.pi,1000)
t = np.linspace(0,2*np.pi,100)
xgrid, tgrid = np.meshgrid(x, t)

# training dataset
training_data = np.vstack([f(a)(xgrid, tgrid)[None,:] for a in params])

# test dataset
test_parameters = [0.15, 0.75, 0.28]
testing_data = np.vstack([f(a)(xgrid, tgrid)[None,:] for a in test_parameters])

testdir = 'tests/test_datasets/param_dmd/'

class TestParamDmd(TestCase):
    def test_is_partitioned_1(self):
        assert not ParametricDMD(None, None, None).is_partitioned

    def test_is_partitioned_2(self):
        assert not ParametricDMD(DMD(), None, None).is_partitioned

    def test_is_partitioned_3(self):
        assert ParametricDMD([DMD() for _ in range(3)], None, None).is_partitioned

    def test_init(self):
        d = DMD()
        p = ParametricDMD(d, sum, np.mean)
        assert p._dmd == d
        assert p._spatial_pod == sum
        assert p._approximation == np.mean

    def test_reference_dmd_1(self):
        d = DMD()
        p = ParametricDMD(d, sum, np.mean)
        assert p._reference_dmd == d

    def test_reference_dmd_2(self):
        l = [DMD() for _ in range(3)]
        p = ParametricDMD(l, sum, np.mean)
        assert p._reference_dmd == l[0]

    def test_dmd_time_1(self):
        d = DMD()
        d.fit(np.ones((10,100)))
        d.dmd_time['tend'] = 200
        d.dmd_time['t0'] = 100

        p = ParametricDMD(d, sum, np.mean)
        assert p.dmd_time['tend'] == 200
        assert p.dmd_time['t0'] == 100

    def test_dmd_time_2(self):
        l = [DMD() for _ in range(3)]
        d = l[0]
        d.fit(np.ones((10,100)))
        d.dmd_time['tend'] = 200
        d.dmd_time['t0'] = 100

        p = ParametricDMD(l, sum, np.mean)
        assert p.dmd_time['tend'] == 200
        assert p.dmd_time['t0'] == 100

    def test_dmd_timesteps_1(self):
        d = DMD()
        d.fit(np.ones((10,100)))
        d.dmd_time['tend'] = 200

        p = ParametricDMD(d, sum, np.mean)
        assert len(p.dmd_timesteps) == 201
        assert p.dmd_timesteps[-1] == 200

    def test_dmd_timesteps_2(self):
        l = [DMD() for _ in range(3)]
        d = l[0]
        d.fit(np.ones((10,100)))
        d.dmd_time['tend'] = 200

        p = ParametricDMD(l, sum, np.mean)
        assert len(p.dmd_timesteps) == 201
        assert p.dmd_timesteps[-1] == 200

    def test_original_time_1(self):
        d = DMD()
        d.fit(np.ones((10,100)))
        d.dmd_time['tend'] = 200

        p = ParametricDMD(d, sum, np.mean)
        assert p.original_time['tend'] == 99

    def test_original_time_2(self):
        l = [DMD() for _ in range(3)]
        d = l[0]
        d.fit(np.ones((10,100)))
        d.dmd_time['tend'] = 200

        p = ParametricDMD(l, sum, np.mean)
        assert p.original_time['tend'] == 99

    def test_original_timesteps_1(self):
        d = DMD()
        d.fit(np.ones((10,100)))
        d.dmd_time['tend'] = 200

        p = ParametricDMD(d, sum, np.mean)
        assert len(p.original_timesteps) == 100
        assert p.original_timesteps[-1] == 99

    def test_original_timesteps_2(self):
        l = [DMD() for _ in range(3)]
        d = l[0]
        d.fit(np.ones((10,100)))
        d.dmd_time['tend'] = 200

        p = ParametricDMD(l, sum, np.mean)
        assert len(p.original_timesteps) == 100
        assert p.original_timesteps[-1] == 99

    def test_training_parameters_1(self):
        p = ParametricDMD(None, None, None)
        p._set_training_parameters(np.ones(10))
        assert p.training_parameters.shape == (10,1)

    def test_training_parameters_2(self):
        p = ParametricDMD(None, None, None)
        p._set_training_parameters([1 for _ in range(10)])
        assert p.training_parameters.shape == (10,1)

    def test_training_parameters_3(self):
        p = ParametricDMD(None, None, None)
        with self.assertRaises(ValueError):
            p._set_training_parameters(np.ones((10,2,2)))

    def test_parameters_1(self):
        p = ParametricDMD(None, None, None)
        p.parameters = np.ones(10)
        assert p.parameters.shape == (10,1)

    def test_parameters_2(self):
        p = ParametricDMD(None, None, None)
        p.parameters = [1 for _ in range(10)]
        assert p.parameters.shape == (10,1)

    def test_parameters_3(self):
        p = ParametricDMD(None, None, None)
        with self.assertRaises(ValueError):
            p.parameters = np.ones((10,2,2))

    def test_fit_wrong_training_shape(self):
        p = ParametricDMD(None, None, None)
        with self.assertRaises(ValueError):
            p.fit(np.ones((9,10,10)), [0 for _ in range(10)])

    # assert that fit sets properly _ntrain, _time_instants, _space_dim
    def test_fit_sets_quantities(self):
        p = ParametricDMD(DMD(), POD(), None)
        p.fit(np.ones((10,9,8)), [0 for _ in range(10)])

        assert p._ntrain == 10
        assert p._time_instants == 9
        assert p._space_dim == 8

    def test_fit_stores_training_parameters(self):
        p = ParametricDMD(DMD(), POD(), None)
        p.fit(np.ones((10,9,8)), [i for i in range(10)])
        assert np.all(p.training_parameters == np.arange(10)[:,None])

    def test_fit_stores_2d_training_parameters(self):
        params = np.hstack([
            np.arange(10)[:,None],
            np.flip(np.arange(10))[:,None]
        ])

        p = ParametricDMD(DMD(), POD(), None)
        p.fit(np.ones((10,9,8)), params)

        # test shape
        assert p.training_parameters.shape == (10,2)
        # test that the order is fine
        assert np.all(np.sum(p.training_parameters, axis=1) == np.repeat(9, 10))

    def test_training_modal_coefficients_shape(self):
        p = ParametricDMD(None, POD(rank=5), None)
        p._ntrain = 10

        res = p._training_modal_coefficients(np.ones((50,60)))
        assert len(res) == 10
        assert res[0].shape == (5,6)

    def test_training_modal_coefficients(self):
        m = np.vander(np.arange(50)/50, 60)
        p = ParametricDMD(None, POD(rank=5), None)
        p._ntrain = 10
        np.testing.assert_allclose(np.hstack(p._training_modal_coefficients(m)), POD(rank=5).fit(m).reduce(m))

    def test_training_modal_coefficients2(self):
        p = ParametricDMD(DMD(svd_rank=-1), POD(rank=5), RBF())
        p.fit(training_data, params)

        inp = np.load(testdir+'space_timemu.npy')
        actual = p._training_modal_coefficients(inp)
        expected = np.load(testdir+'traning_modal_coefficients.npy')

        np.testing.assert_allclose(actual, expected)

    def test_arrange_parametric_snapshots_shape(self):
        p = ParametricDMD(None, None, None)
        p._space_dim = 30
        p._time_instants = 20
        p._ntrain = 10
        assert p._arrange_parametric_snapshots(np.ones((10,20,30))).shape == (30, 200)

    def test_arrange_parametric_snapshots(self):
        p = ParametricDMD(None, None, None)
        p._space_dim = 3
        p._time_instants = 2
        p._ntrain = 2

        m1 = np.array([
            [1,2,3],
            [4,5,6]
        ])[None,:]
        m2 = np.array([
            [0,1,0,],
            [1,1,0]
        ])[None,:]
        m = np.vstack([m1,m2])

        expected = np.array([
            [1,4,0,1],
            [2,5,1,1],
            [3,6,0,0],
        ])

        np.testing.assert_array_equal(expected, p._arrange_parametric_snapshots(m))

    def test_arrange_parametric_snapshots2(self):
        p = ParametricDMD(DMD(svd_rank=-1), POD(rank=5), RBF())
        p.fit(training_data, params)

        expected = np.load(testdir+'space_timemu.npy')
        actual = p._arrange_parametric_snapshots(training_data)
        np.testing.assert_allclose(actual, expected)

    def test_fit_dmd_partitioned(self):
        p = ParametricDMD([DMD() for _ in range(5)], None, None)
        p._fit_dmd([np.ones((20,10)) for _ in range(5)])
        for i in range(5):
            # assert that fit was called
            assert p._dmd[i].modes is not None
            assert p._dmd[i].modes.shape[0] == 20

    def test_fit_dmd_monolithic(self):
        p = ParametricDMD(DMD(), None, None)
        p._fit_dmd([np.ones((20,10)) for _ in range(5)])
        # assert that fit was called
        assert p._dmd.modes is not None
        assert p._dmd.modes.shape[0] == 100

    def test_predict_modal_coefficients_shape(self):
        p = ParametricDMD(DMD(svd_rank=5), POD(rank=10), RBF())
        p.fit(np.ones((10,20,40)), np.arange(10))
        p.dmd_time['tend'] = 29
        assert p._predict_modal_coefficients().shape == (10*10, 30)

    def test_predict_modal_coefficients_partitioned_shape(self):
        p = ParametricDMD([DMD(svd_rank=5) for _ in range(10)], POD(rank=10), RBF())
        p.fit(np.ones((10,20,40)), np.arange(10))
        p.dmd_time['tend'] = 29
        assert p._predict_modal_coefficients().shape == (10*10, 30)

    def test_predict_modal_coefficients(self):
        p = ParametricDMD(DMD(svd_rank=-1), POD(rank=5), RBF())
        p.fit(training_data, params)

        expected = np.load(testdir+'forecasted.npy')
        np.testing.assert_allclose(p._predict_modal_coefficients(), expected)

    def test_interpolate_missing_modal_coefficients_shape(self):
        p = ParametricDMD(DMD(svd_rank=5), POD(rank=10), RBF())
        p.fit(np.ones((10,20,40)), np.arange(10))
        p.dmd_time['tend'] = 29
        p.parameters = [1.5, 5.5, 7.5]
        assert p._interpolate_missing_modal_coefficients(np.random.rand(10*10, 30)).shape == (30,3,10)

    def test_interpolate_missing_modal_coefficients_wrong_time(self):
        p = ParametricDMD(DMD(svd_rank=5), POD(rank=10), RBF())
        p.fit(np.ones((10,20,40)), np.arange(10))
        p.dmd_time['tend'] = 29
        p.parameters = [1.5, 5.5, 7.5]
        with self.assertRaises(ValueError):
            p._interpolate_missing_modal_coefficients(np.random.rand(10*10, 20))

    def test_interpolate_missing_modal_coefficients(self):
        p = ParametricDMD(DMD(svd_rank=-1), POD(rank=5), RBF())
        p.fit(training_data, params)
        p.parameters = test_parameters

        expected = np.load(testdir+'interpolated.npy')
        np.testing.assert_allclose(
            p._interpolate_missing_modal_coefficients(np.load(testdir+'forecasted.npy')),
            expected)

    def reconstructed_data_shape(self):
        p = ParametricDMD(DMD(svd_rank=5), POD(rank=10), RBF())
        p.fit(np.random.rand(10,20,50), np.arange(10))
        p.dmd_time['tend'] = 39
        p.parameters = [1.5, 5.5, 7.5,8.5]
        assert p.reconstructed_data.shape == (4, 40, 50)

    def test_reconstructed_data_noprediction(self):
        p = ParametricDMD(DMD(svd_rank=-1), POD(rank=5), RBF())
        p.fit(training_data, params)
        p.parameters = test_parameters

        rec = p.reconstructed_data

        assert rec.shape == (3,100,1000)

        np.testing.assert_allclose(rec.real, testing_data.real, atol=1.e-2)
