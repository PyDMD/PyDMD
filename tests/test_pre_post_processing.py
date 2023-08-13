import numpy as np
from unittest.mock import patch
from pydmd.pre_post_processing import (
    PrePostProcessingDMD,
    zero_mean_preprocessing,
)

_kwargs = {"param1": "value1", "param2": "value2"}


def test_pre_processing(mocker):
    partial_output = mocker.Mock()
    dmd = mocker.Mock(reconstructed_data=partial_output)
    dmd.fit = mocker.Mock()

    preproc_output = mocker.Mock()
    pre = mocker.Mock(return_value=(preproc_output, *_kwargs.values()))

    output = mocker.Mock()
    post = mocker.Mock(return_value=output)

    pdmd = PrePostProcessingDMD(dmd, pre, post)

    X = mocker.Mock()
    mock_dict = mocker.Mock()
    with patch("pydmd.pre_post_processing.dict", return_value=mock_dict):
        pdmd.fit(X, **_kwargs)

    pre.assert_called_once_with(mock_dict, X, **_kwargs)
    dmd.fit.assert_called_once_with(preproc_output, *_kwargs.values())

    assert pdmd.reconstructed_data is output
    post.assert_called_once_with(mock_dict, partial_output)
    assert pdmd._state_holder is None


def test_pre_processing_default_preprocessing(mocker):
    partial_output = mocker.Mock()
    dmd = mocker.Mock(reconstructed_data=partial_output)
    dmd.fit = mocker.Mock()

    output = mocker.Mock()
    post = mocker.Mock(return_value=output)

    pdmd = PrePostProcessingDMD(dmd, post_processing=post)

    X = mocker.Mock()
    pdmd.fit(X, **_kwargs)

    dmd.fit.assert_called_once_with(X, *_kwargs.values())
    assert not pdmd._state_holder


def test_pre_processing_default_postprocessing(mocker):
    partial_output = mocker.Mock()
    dmd = mocker.Mock(reconstructed_data=partial_output)
    dmd.fit = mocker.Mock()

    pdmd = PrePostProcessingDMD(dmd)

    X = mocker.Mock()
    pdmd.fit(X)

    assert pdmd.reconstructed_data is partial_output


def test_zero_mean(mocker):
    dmd = mocker.Mock()
    pdmd = zero_mean_preprocessing(dmd)

    X = np.array([1, 2, 3], dtype=float)
    pdmd.fit(X)
    fit_call_args = dmd.fit.call_args_list
    assert len(fit_call_args) == 1
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
