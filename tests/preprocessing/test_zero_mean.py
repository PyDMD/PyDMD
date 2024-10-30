from types import MappingProxyType

import numpy as np

from pydmd import DMDc
from pydmd.preprocessing import zero_mean_preprocessing

kwargs = MappingProxyType({"a": 1, "b": 2})


def test_zero_mean(mocker):
    dmd = mocker.Mock()
    pdmd = zero_mean_preprocessing(dmd, axis=None)

    X = np.array([1, 2, 3], dtype=float)
    pdmd.fit(X)
    fit_call_args = dmd.fit.call_args_list
    assert len(fit_call_args) == 1
    assert len(fit_call_args[0].args) == 1
    np.testing.assert_array_equal(fit_call_args[0].args[0], [-1, 0, 1])
    assert not fit_call_args[0].kwargs

    dmd.reconstructed_data = np.zeros(3, dtype=float)
    assert (pdmd.reconstructed_data == [2, 2, 2]).all()


def test_zero_mean_with_kwargs(mocker):
    dmd = mocker.Mock()
    pdmd = zero_mean_preprocessing(dmd, axis=None)

    X = np.array([1, 2, 3], dtype=float)
    pdmd.fit(X, **kwargs)
    fit_call_args = dmd.fit.call_args_list
    assert len(fit_call_args) == 1
    assert len(fit_call_args[0].args) == 1
    np.testing.assert_array_equal(fit_call_args[0].args[0], [-1, 0, 1])
    assert fit_call_args[0].kwargs == kwargs


def test_zero_mean_with_kwargs_axis0(mocker):
    dmd = mocker.Mock()
    pdmd = zero_mean_preprocessing(dmd, axis=0)

    X = np.array([[1, 2, 3], [3, 6, 9]], dtype=float)
    pdmd.fit(X, **kwargs)
    fit_call_args = dmd.fit.call_args_list
    assert len(fit_call_args) == 1
    assert len(fit_call_args[0].args) == 1
    np.testing.assert_array_equal(
        fit_call_args[0].args[0], [[-1, -2, -3], [1, 2, 3]]
    )
    assert fit_call_args[0].kwargs == kwargs


def test_zero_mean_with_kwargs_axis1(mocker):
    dmd = mocker.Mock()
    pdmd = zero_mean_preprocessing(dmd, axis=1)

    X = np.array([[1, 2, 3], [4, 5, 9]], dtype=float)
    pdmd.fit(X, **kwargs)
    fit_call_args = dmd.fit.call_args_list
    assert len(fit_call_args) == 1
    assert len(fit_call_args[0].args) == 1
    np.testing.assert_array_equal(
        fit_call_args[0].args[0], [[-1, 0, 1], [-2, -1, 3]]
    )
    assert fit_call_args[0].kwargs == kwargs


def test_zero_mean_with_kwargs_axis_default1(mocker):
    dmd = mocker.Mock()
    pdmd = zero_mean_preprocessing(dmd)

    X = np.array([[1, 2, 3], [4, 5, 9]], dtype=float)
    pdmd.fit(X, **kwargs)
    fit_call_args = dmd.fit.call_args_list
    assert len(fit_call_args) == 1
    assert len(fit_call_args[0].args) == 1
    np.testing.assert_array_equal(
        fit_call_args[0].args[0], [[-3, -2, -1], [0, 1, 5]]
    )
    assert fit_call_args[0].kwargs == kwargs


def test_dmdc():
    state_data = np.random.rand(10, 100)
    input_data = np.random.rand(1, 100)

    dmdc = DMDc(svd_rank=-1)
    dmd = zero_mean_preprocessing(dmdc)
    dmd.fit(state_data, input_data[:, :-1])

    assert isinstance(dmd.reconstructed_data(), np.ndarray)
