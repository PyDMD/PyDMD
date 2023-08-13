from unittest.mock import patch
from copy import copy, deepcopy
from pydmd.preprocessing import PrePostProcessingDMD

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
    with patch(
        "pydmd.preprocessing.pre_post_processing.dict", return_value=mock_dict
    ):
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


def test_copy(mocker):
    dmd = mocker.Mock()
    dmd.fit = mocker.Mock()
    pdmd = PrePostProcessingDMD(dmd)
    assert copy(pdmd)._pre_post_processed_dmd == dmd


def test_deepcopy(mocker):
    dmd = mocker.Mock()
    dmd.fit = mocker.Mock()
    pdmd = PrePostProcessingDMD(dmd)
    assert deepcopy(pdmd)._pre_post_processed_dmd is not None
