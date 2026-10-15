import numpy as np
import pytest
from pytest import raises

from pydmd import DMD
from pydmd.dmdbase import DMDBase
from pydmd.snapshots import Snapshots

# 15 snapshot with 400 data. The matrix is 400x15 and it contains
# the following data: f1 + f2 where
# f1 = lambda x,t: sech(x+3)*(1.*np.exp(1j*2.3*t))
# f2 = lambda x,t: (sech(x)*np.tanh(x))*(2.*np.exp(1j*2.8*t))
sample_data = np.load("tests/test_datasets/input_sample.npy")


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

    tpow = np.ndarray([0, 1, 2, 3, 5, 6, 7, 11])
    for idx, x in enumerate(dmd._translate_eigs_exponent(tpow)):
        assert x == dmd._translate_eigs_exponent(tpow[idx])


def test_sorted_eigs_default():
    dmd = DMDBase()
    assert dmd.operator._sorted_eigs == False


def test_sorted_eigs_param():
    dmd = DMDBase(sorted_eigs="real")
    assert dmd.operator._sorted_eigs == "real"


def test_dmd_time_wrong_key():
    dmd = DMD(svd_rank=10)
    dmd.fit(sample_data)

    with raises(KeyError):
        dmd.dmd_time["tstart"] = 10


def test_timesteps_integer_dt():
    # fit() builds an integer time dictionary, whose timesteps must stay
    # integral and must span exactly one step per snapshot.
    dmd = DMD(svd_rank=10)
    dmd.fit(sample_data)

    expected = np.arange(sample_data.shape[1])
    np.testing.assert_array_equal(dmd.original_timesteps, expected)
    np.testing.assert_array_equal(dmd.dmd_timesteps, expected)
    assert np.issubdtype(dmd.original_timesteps.dtype, np.integer)
    assert np.issubdtype(dmd.dmd_timesteps.dtype, np.integer)


@pytest.mark.parametrize(
    "t0, dt, n_steps",
    [
        (0.0, 0.1, 2),  # smallest case exhibiting the #583 failure mode
        (0.0, 0.1, 11),
        (0.0, 0.3, 13),
        (0.0, 1 / 3, 3),  # dt with no exact binary representation
        (0.0, 0.25, 8),  # dt exactly representable, control case
        (0.5, 0.1, 7),  # t0 not on a dt boundary
        (-1.0, 0.2, 6),  # negative t0
    ],
)
def test_dmd_timesteps_float_dt(t0, dt, n_steps):
    # A floating-point dt used to produce one timestep too many: the stop
    # boundary tend + dt could round upward, and np.arange would emit a step
    # past tend. See issue #583.
    dmd = DMD(svd_rank=10)
    dmd.fit(sample_data)

    dmd.dmd_time["t0"] = t0
    dmd.dmd_time["dt"] = dt
    dmd.dmd_time["tend"] = t0 + n_steps * dt

    timesteps = dmd.dmd_timesteps

    assert len(timesteps) == n_steps + 1
    np.testing.assert_allclose(timesteps[0], t0)
    np.testing.assert_allclose(timesteps[-1], t0 + n_steps * dt)
    np.testing.assert_allclose(np.diff(timesteps), dt)


@pytest.mark.parametrize(
    "t0, tend, dt, expected_last",
    [
        (0, 10, 3, 9),  # nearest grid point is below tend
        (0.0, 0.28, 0.1, 0.3),  # nearest grid point is above tend
        (0.0, 0.25, 0.1, 0.2),  # tend exactly between two grid points
    ],
)
def test_dmd_timesteps_tend_off_grid(t0, tend, dt, expected_last):
    # When tend is not a whole number of steps from t0 there is no exactly
    # right answer. The last timestep is the grid point nearest tend, so the
    # series never runs more than dt/2 past it.
    dmd = DMD(svd_rank=10)
    dmd.fit(sample_data)

    dmd.dmd_time["t0"] = t0
    dmd.dmd_time["dt"] = dt
    dmd.dmd_time["tend"] = tend

    timesteps = dmd.dmd_timesteps

    np.testing.assert_allclose(timesteps[-1], expected_last)
    assert abs(timesteps[-1] - tend) <= abs(dt) / 2


def test_timesteps_empty_window():
    # An empty window stays empty, which is what np.arange gave when the
    # bounds were handed straight to it.
    dmd = DMD(svd_rank=10)
    dmd.fit(sample_data)

    dmd.dmd_time["t0"] = 10
    dmd.dmd_time["tend"] = 5
    dmd.dmd_time["dt"] = 1

    assert len(dmd.dmd_timesteps) == 0
