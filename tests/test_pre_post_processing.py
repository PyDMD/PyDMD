import numpy as np
from pydmd.pre_post_processing import (
    PrePostProcessingDMD,
    zero_mean_preprocessing,
)


def test_pre_processing(mocker):
    partial_output = mocker.Mock()
    dmd = mocker.Mock(reconstructed_data=partial_output)
    dmd.fit = mocker.Mock()

    preproc_state = mocker.Mock()
    pre = mocker.Mock(return_value=preproc_state)

    output = mocker.Mock()
    post = mocker.Mock(return_value=output)

    pdmd = PrePostProcessingDMD(dmd, pre, post)

    X = mocker.Mock()
    pdmd.fit(X)
    pre.assert_called_once_with(X)
    dmd.fit.assert_called_once_with(X)

    assert pdmd.reconstructed_data is output
    post.assert_called_once_with(partial_output, preproc_state)


def test_pre_processing_default_preprocessing(mocker):
    partial_output = mocker.Mock()
    dmd = mocker.Mock(reconstructed_data=partial_output)
    dmd.fit = mocker.Mock()

    output = mocker.Mock()
    post = mocker.Mock(return_value=output)

    pdmd = PrePostProcessingDMD(dmd, post_processing=post)

    X = mocker.Mock()
    pdmd.fit(X)
    dmd.fit.assert_called_once_with(X)

    assert pdmd.reconstructed_data is output
    post.assert_called_once_with(partial_output)


def test_pre_processing_default_postprocessing(mocker):
    partial_output = mocker.Mock()
    dmd = mocker.Mock(reconstructed_data=partial_output)
    dmd.fit = mocker.Mock()

    preproc_state = mocker.Mock()
    pre = mocker.Mock(return_value=preproc_state)

    pdmd = PrePostProcessingDMD(dmd, pre)

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
