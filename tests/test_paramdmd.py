from pytest import raises
from pydmd import DMD, ParametricDMD
from pydmd.dmdbase import DMDTimeDict
from pydmd.paramdmd import back_roll_shape, roll_shape
from ezyrb import POD, RBF
import numpy as np
import os

testdir = 'tests/test_datasets/param_dmd/'

#def f1(x,t):
#    return 1./np.cosh(x+3)*np.exp(2.3j*t)
#def f2(x,t):
#    return 2./np.cosh(x)*np.tanh(x)*np.exp(2.8j*t)
#def f(a):
#    def fnc(x,t):
#        return a*f1(x,t) + (1-a)*f2(x,t)
#    return fnc

params = np.linspace(0,1,10)

# x = np.linspace(0,np.pi,1000)
# t = np.linspace(0,2*np.pi,100)
# xgrid, tgrid = np.meshgrid(x, t)

# training dataset
# training_data = np.vstack([f(a)(xgrid, tgrid)[None,:] for a in params])
training_data = np.swapaxes(np.load(testdir + '/training_data.npy'), 1, 2)

# test dataset
test_parameters = [0.15, 0.75, 0.28]
# testing_data = np.vstack([f(a)(xgrid, tgrid)[None,:] for a in test_parameters])
testing_data = np.swapaxes(np.load(testdir + 'testing_data.npy'), 1, 2)

def test_is_partitioned_1():
    assert not ParametricDMD(None, None, None).is_partitioned

def test_is_partitioned_2():
    assert not ParametricDMD(DMD(), None, None).is_partitioned

def test_is_partitioned_3():
    assert ParametricDMD([DMD() for _ in range(3)], None, None).is_partitioned

def test_init():
    d = DMD()
    p = ParametricDMD(d, sum, np.mean)
    assert p._dmd == d
    assert p._spatial_pod == sum
    assert p._approximation == np.mean

def test_reference_dmd_1():
    d = DMD()
    p = ParametricDMD(d, sum, np.mean)
    assert p._reference_dmd == d

def test_reference_dmd_2():
    l = [DMD() for _ in range(3)]
    p = ParametricDMD(l, sum, np.mean)
    assert p._reference_dmd == l[0]

def test_dmd_time_1():
    d = DMD()
    d.fit(np.ones((10,100)))
    d.dmd_time['tend'] = 200
    d.dmd_time['t0'] = 100

    p = ParametricDMD(d, sum, np.mean)
    assert p.dmd_time['tend'] == 200
    assert p.dmd_time['t0'] == 100

def test_dmd_time_2():
    l = [DMD() for _ in range(3)]
    d = l[0]
    d.fit(np.ones((10,100)))
    d.dmd_time['tend'] = 200
    d.dmd_time['t0'] = 100

    p = ParametricDMD(l, sum, np.mean)
    assert p.dmd_time['tend'] == 200
    assert p.dmd_time['t0'] == 100

def test_dmd_timesteps_1():
    d = DMD()
    d.fit(np.ones((10,100)))
    d.dmd_time['tend'] = 200

    p = ParametricDMD(d, sum, np.mean)
    assert len(p.dmd_timesteps) == 201
    assert p.dmd_timesteps[-1] == 200

def test_dmd_timesteps_2():
    l = [DMD() for _ in range(3)]
    d = l[0]
    d.fit(np.ones((10,100)))
    d.dmd_time['tend'] = 200

    p = ParametricDMD(l, sum, np.mean)
    assert len(p.dmd_timesteps) == 201
    assert p.dmd_timesteps[-1] == 200

def test_original_time_1():
    d = DMD()
    d.fit(np.ones((10,100)))
    d.dmd_time['tend'] = 200

    p = ParametricDMD(d, sum, np.mean)
    assert p.original_time['tend'] == 99

def test_original_time_2():
    l = [DMD() for _ in range(3)]
    d = l[0]
    d.fit(np.ones((10,100)))
    d.dmd_time['tend'] = 200

    p = ParametricDMD(l, sum, np.mean)
    assert p.original_time['tend'] == 99

def test_original_timesteps_1():
    d = DMD()
    d.fit(np.ones((10,100)))
    d.dmd_time['tend'] = 200

    p = ParametricDMD(d, sum, np.mean)
    assert len(p.original_timesteps) == 100
    assert p.original_timesteps[-1] == 99

def test_original_timesteps_2():
    l = [DMD() for _ in range(3)]
    d = l[0]
    d.fit(np.ones((10,100)))
    d.dmd_time['tend'] = 200

    p = ParametricDMD(l, sum, np.mean)
    assert len(p.original_timesteps) == 100
    assert p.original_timesteps[-1] == 99

def test_training_parameters_1():
    p = ParametricDMD(None, None, None)
    p._set_training_parameters(np.ones(10))
    assert p.training_parameters.shape == (10,1)

def test_training_parameters_2():
    p = ParametricDMD(None, None, None)
    p._set_training_parameters([1 for _ in range(10)])
    assert p.training_parameters.shape == (10,1)

def test_training_parameters_3():
    p = ParametricDMD(None, None, None)
    with raises(ValueError):
        p._set_training_parameters(np.ones((10,2,2)))

def test_parameters_1():
    p = ParametricDMD(None, None, None)
    p.parameters = np.ones(10)
    assert p.parameters.shape == (10,1)

def test_parameters_2():
    p = ParametricDMD(None, None, None)
    p.parameters = [1 for _ in range(10)]
    assert p.parameters.shape == (10,1)

def test_parameters_3():
    p = ParametricDMD(None, None, None)
    with raises(ValueError):
        p.parameters = np.ones((10,2,2))

def test_fit_wrong_training_shape():
    p = ParametricDMD(None, None, None)
    with raises(ValueError):
        p.fit(np.ones((9,10,10)), [0 for _ in range(10)])

# assert that fit sets properly _ntrain, _time_instants, _space_dim
def test_fit_sets_quantities():
    p = ParametricDMD(DMD(), POD(), None)
    p.fit(np.ones((10,9,8)), [0 for _ in range(10)])

    assert p._ntrain == 10
    assert p._time_instants == 8
    assert p._space_dim == 9

def test_fit_stores_training_parameters():
    p = ParametricDMD(DMD(), POD(), None)
    p.fit(np.ones((10,9,8)), [i for i in range(10)])
    assert np.all(p.training_parameters == np.arange(10)[:,None])

def test_fit_stores_2d_training_parameters():
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

def test_compute_training_modal_coefficients_shape():
    p = ParametricDMD(None, POD(rank=5), None)
    p._ntrain = 10

    res = p._compute_training_modal_coefficients(np.ones((50,60)))
    assert len(res) == 10
    assert res[0].shape == (5,6)

def test_compute_training_modal_coefficients():
    m = np.vander(np.arange(50)/50, 60)
    p = ParametricDMD(None, POD(rank=5), None)
    p._ntrain = 10
    np.testing.assert_allclose(np.hstack(p._compute_training_modal_coefficients(m)), POD(rank=5).fit(m).reduce(m))

def test_compute_training_modal_coefficients2():
    p = ParametricDMD(DMD(svd_rank=-1), POD(rank=5), RBF())
    p.fit(training_data, params)

    inp = np.load(testdir+'space_timemu.npy')
    actual = p._compute_training_modal_coefficients(inp)
    expected = np.load(testdir+'traning_modal_coefficients.npy')

    np.testing.assert_allclose(actual, expected, atol=1.e-12, rtol=0)

def test_arrange_parametric_snapshots_shape():
    p = ParametricDMD(None, None, None)
    assert p._arrange_parametric_snapshots(np.ones((10,20,30))).shape == (20, 300)

def test_arrange_parametric_snapshots():
    p = ParametricDMD(None, None, None)

    m1 = np.array([
        [1,2,3],
        [4,5,6]
    ])
    m2 = np.array([
        [0,1,0,],
        [1,1,0]
    ])
    m = np.stack([m1,m2])

    expected = np.array([
        [1,2,3,0,1,0],
        [4,5,6,1,1,0],
    ])

    np.testing.assert_array_equal(expected, p._arrange_parametric_snapshots(m))

def test_arrange_parametric_snapshots2():
    p = ParametricDMD(DMD(svd_rank=-1), POD(rank=5), RBF())
    p.fit(training_data, params)

    expected = np.load(testdir+'space_timemu.npy')
    actual = p._arrange_parametric_snapshots(training_data)
    np.testing.assert_allclose(actual, expected)

def test_predict_modal_coefficients_shape():
    p = ParametricDMD(DMD(svd_rank=5), POD(rank=10), RBF())
    p.fit(np.ones((10,20,40)), np.arange(10))
    p.dmd_time['tend'] = 29
    assert p._predict_modal_coefficients().shape == (10*10, 30)

def test_predict_modal_coefficients_partitioned_shape():
    p = ParametricDMD([DMD(svd_rank=5) for _ in range(10)], POD(rank=10), RBF())
    p.fit(np.ones((10,20,40)), np.arange(10))
    p.dmd_time['tend'] = 29
    assert p._predict_modal_coefficients().shape == (10*10, 30)

def test_predict_modal_coefficients():
    p = ParametricDMD(DMD(svd_rank=-1), POD(rank=5), RBF())
    p.fit(training_data, params)

    expected = np.load(testdir+'forecasted.npy')
    np.testing.assert_allclose(p._predict_modal_coefficients(), expected, rtol=0, atol=1.e-11)

def test_interpolate_missing_modal_coefficients_shape():
    p = ParametricDMD(DMD(svd_rank=5), POD(rank=10), RBF())
    p.fit(np.ones((10,20,40)), np.arange(10))
    p.dmd_time['tend'] = 49
    p.parameters = [1.5, 5.5, 7.5]
    assert p._interpolate_missing_modal_coefficients(np.random.rand(10*5,40)).shape == (3,5,40)

# def test_interpolate_missing_modal_coefficients():
#     p = ParametricDMD(DMD(svd_rank=-1), POD(rank=5), RBF())
#     p.fit(training_data, params)
#     p.parameters = test_parameters

#     expected = back_roll_shape(np.load(testdir+'interpolated.npy'))
#     input = np.load(testdir+'forecasted.npy')
#     np.testing.assert_allclose(
#         p._interpolate_missing_modal_coefficients(input),
#         expected, atol=1.e-12, rtol=0)

def reconstructed_data_shape():
    p = ParametricDMD(DMD(svd_rank=5), POD(rank=10), RBF())
    p.fit(np.random.rand(10,20,50), np.arange(10))
    p.dmd_time['tend'] = 39
    p.parameters = [1.5, 5.5, 7.5,8.5]
    assert p.reconstructed_data.shape == (4, 40, 50)

def test_reconstructed_data_noprediction():
    p = ParametricDMD(DMD(svd_rank=-1), POD(rank=5), RBF())
    p.fit(training_data, params)
    p.parameters = test_parameters

    rec = p.reconstructed_data

    np.testing.assert_allclose(rec.real, testing_data.real, atol=1.e-2)

def test_save():
    p = ParametricDMD(DMD(svd_rank=-1), POD(rank=5), RBF())
    p.fit(training_data, params)
    p.parameters = test_parameters
    p.save('pydmd.test')
    os.remove('pydmd.test')

def test_load():
    p = ParametricDMD(DMD(svd_rank=-1), POD(rank=5), RBF())
    p.fit(training_data, params)
    p.parameters = test_parameters
    p.save('pydmd.test2')
    loaded_p = ParametricDMD.load('pydmd.test2')
    np.testing.assert_array_equal(p.reconstructed_data,
                                    loaded_p.reconstructed_data)
    os.remove('pydmd.test2')

def test_load2():
    p = ParametricDMD(DMD(svd_rank=-1), POD(rank=5), RBF())
    p.fit(training_data, params)
    p.parameters = test_parameters
    p.save('pydmd.test2')
    loaded_p = ParametricDMD.load('pydmd.test2')
    assert isinstance(loaded_p, ParametricDMD)
    os.remove('pydmd.test2')

def test_set_time_monolithic():
    p = ParametricDMD(DMD(svd_rank=-1), POD(rank=5), RBF())
    p.fit(training_data, params)
    p.parameters = test_parameters

    dc = DMDTimeDict()
    dc['t0'] = 10
    dc['tend'] = 20
    dc['dt'] = 1
    p.dmd_time = dc

    assert p.dmd_time == dc
    assert p.dmd_time['t0'] == 10
    assert p.dmd_time['dt'] == 1
    assert p.dmd_time['tend'] == 20

def test_set_time_partitioned():
    dmds = [DMD(svd_rank=-1) for _ in range(len(params))]
    p = ParametricDMD(dmds, POD(rank=5), RBF())
    p.fit(training_data, params)
    p.parameters = test_parameters

    dc = DMDTimeDict()
    dc['t0'] = 10
    dc['tend'] = 20
    dc['dt'] = 1
    p.dmd_time = dc
    p._predict_modal_coefficients()

    for dmd in dmds:
        assert dmd.dmd_time == dc
        assert dmd.dmd_time['t0'] == 10
        assert dmd.dmd_time['dt'] == 1
        assert dmd.dmd_time['tend'] == 20

def test_forecast():
    dmds = [DMD(svd_rank=-1) for _ in range(len(params))]
    p = ParametricDMD(dmds, POD(rank=5), RBF())
    p.fit(training_data, params)
    p.parameters = test_parameters

    dc = DMDTimeDict()
    dc['t0'] = 101
    dc['tend'] = 200
    dc['dt'] = 1
    p.dmd_time = dc

    for r in p.reconstructed_data:
        assert r.shape == (training_data.shape[1], 100)

def test_training_modal_coefficients_property_partitioned():
    dmds = [DMD(svd_rank=-1) for _ in range(len(params))]
    p = ParametricDMD(dmds, POD(rank=5), RBF())
    p.fit(training_data, params)
    assert p.training_modal_coefficients.shape == (10, 5, 100)

def test_training_modal_coefficients_property_monolithic():
    dmd = DMD(svd_rank=-1)
    p = ParametricDMD(dmd, POD(rank=5), RBF())
    p.fit(training_data, params)
    assert p.training_modal_coefficients.shape == (10, 5, 100)

def test_no_training_modal_coefficients_light():
    dmds = [DMD(svd_rank=-1) for _ in range(len(params))]
    p = ParametricDMD(dmds, POD(rank=5), RBF(), light=True)
    p.fit(training_data, params)
    with raises(RuntimeError):
        p.training_modal_coefficients

def test_no_training_modal_coefficients_before_fit():
    dmds = [DMD(svd_rank=-1) for _ in range(len(params))]
    p = ParametricDMD(dmds, POD(rank=5), RBF(), light=True)
    with raises(RuntimeError):
        p.training_modal_coefficients

def test_forecasted_modal_coefficients_shape():
    dmds = [DMD(svd_rank=-1) for _ in range(len(params))]
    p = ParametricDMD(dmds, POD(rank=5), RBF(), light=True)
    p.fit(training_data, params)
    assert p.forecasted_modal_coefficients.shape == (10, 5, 100)
    p.dmd_time['tend'] += 10
    assert p.forecasted_modal_coefficients.shape == (10, 5, 110)

def test_interpolated_modal_coefficients_shape():
    dmds = [DMD(svd_rank=-1) for _ in range(len(params))]
    p = ParametricDMD(dmds, POD(rank=4), RBF(), light=True)
    p.fit(training_data, params)
    p.parameters = [0.25, 0.98, 0.99]
    assert p.interpolated_modal_coefficients.shape == (3, 4, 100)
    p.dmd_time['tend'] += 100
    assert p.interpolated_modal_coefficients.shape == (3, 4, 200)

def test_forecasted_modal_coefficients_reshape():
    dmds = [DMD(svd_rank=-1) for _ in range(len(params))]
    p = ParametricDMD(dmds, POD(rank=4), RBF(), light=True)
    p.fit(training_data, params)
    p.dmd_time['tend'] += 100

    forecasted_modal_coefficients = p._predict_modal_coefficients()
    np.testing.assert_allclose(p.forecasted_modal_coefficients[0,3], forecasted_modal_coefficients[3])
    np.testing.assert_allclose(p.forecasted_modal_coefficients[5,0], forecasted_modal_coefficients[20])

def test_interpolated_modal_coefficients_reshape():
    dmds = [DMD(svd_rank=-1) for _ in range(len(params))]
    p = ParametricDMD(dmds, POD(rank=5), RBF(), light=True)
    p.fit(training_data, params)
    p.dmd_time['tend'] += 100
    p.parameters = [0.25, 0.98, 0.99]

    forecasted_modal_coefficients = p._predict_modal_coefficients()
    interpolated_modal_coefficients = (
        p._interpolate_missing_modal_coefficients(
            forecasted_modal_coefficients
        )
    )
    np.testing.assert_allclose(p.interpolated_modal_coefficients[1,3], interpolated_modal_coefficients[1,3])
    np.testing.assert_allclose(p.interpolated_modal_coefficients[2,0], interpolated_modal_coefficients[2,0])

def test_forecasted_modal_coefficients_shape_monolithic():
    dmds = DMD(svd_rank=-1)
    p = ParametricDMD(dmds, POD(rank=5), RBF(), light=True)
    p.fit(training_data, params)
    assert p.forecasted_modal_coefficients.shape == (10, 5, 100)
    p.dmd_time['tend'] += 10
    assert p.forecasted_modal_coefficients.shape == (10, 5, 110)

def test_interpolated_modal_coefficients_shape_monolithic():
    dmds = DMD(svd_rank=-1)
    p = ParametricDMD(dmds, POD(rank=4), RBF(), light=True)
    p.fit(training_data, params)
    p.parameters = [0.25, 0.98, 0.99]
    assert p.interpolated_modal_coefficients.shape == (3, 4, 100)
    p.dmd_time['tend'] += 100
    assert p.interpolated_modal_coefficients.shape == (3, 4, 200)

def test_forecasted_modal_coefficients_reshape_monolithic():
    dmds = DMD(svd_rank=-1)
    p = ParametricDMD(dmds, POD(rank=4), RBF(), light=True)
    p.fit(training_data, params)
    p.dmd_time['tend'] += 100

    forecasted_modal_coefficients = p._predict_modal_coefficients()
    np.testing.assert_allclose(p.forecasted_modal_coefficients[0,3], forecasted_modal_coefficients[3])
    np.testing.assert_allclose(p.forecasted_modal_coefficients[5,0], forecasted_modal_coefficients[20])

def test_interpolated_modal_coefficients_reshape_monolithic():
    dmds = DMD(svd_rank=-1)
    p = ParametricDMD(dmds, POD(rank=5), RBF(), light=True)
    p.fit(training_data, params)
    p.dmd_time['tend'] += 100
    p.parameters = [0.25, 0.98, 0.99]

    forecasted_modal_coefficients = p._predict_modal_coefficients()
    interpolated_modal_coefficients = (
        p._interpolate_missing_modal_coefficients(
            forecasted_modal_coefficients
        )
    )
    np.testing.assert_allclose(p.interpolated_modal_coefficients[1,3], interpolated_modal_coefficients[1,3])
    np.testing.assert_allclose(p.interpolated_modal_coefficients[2,0], interpolated_modal_coefficients[2,0])

def test_no_parameters_error():
    dmds = DMD(svd_rank=-1)
    p = ParametricDMD(dmds, POD(rank=5), RBF(), light=True)
    p.fit(training_data, params)
    p.dmd_time['tend'] += 100

    with raises(ValueError, match='Unknown parameters not found. Did you set `ParametricDMD.parameters`?'):
        p.reconstructed_data
