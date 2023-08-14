import numpy as np
from pydmd.preprocessing import zero_mean_preprocessing
from .test_pre_post_processing import _kwargs


def test_zero_mean(mocker):
    dmd = mocker.Mock()
    pdmd = zero_mean_preprocessing(dmd)

    X = np.array([1, 2, 3], dtype=float)
    pdmd.fit(X)
    fit_call_args = dmd.fit.call_args_list
    assert len(fit_call_args) == 1
    print(fit_call_args[0])
    assert (fit_call_args[0][0][0] == [-1, 0, 1]).all()

    dmd.reconstructed_data = np.zeros(3, dtype=float)
    assert (pdmd.reconstructed_data == [2, 2, 2]).all()


def test_zero_mean_with_kwargs(mocker):
    dmd = mocker.Mock()
    pdmd = zero_mean_preprocessing(dmd)

    X = np.array([1, 2, 3], dtype=float)
    pdmd.fit(X, **_kwargs)
    fit_call_args = dmd.fit.call_args_list
    assert len(fit_call_args) == 1
    assert (fit_call_args[0][0][0] == [-1, 0, 1]).all()
    assert fit_call_args[0][0][1:] == tuple(_kwargs.values())
