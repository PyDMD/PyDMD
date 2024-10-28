from copy import copy, deepcopy
from unittest.mock import patch

from pydmd.preprocessing.pre_post_processing import PrePostProcessingDMD


def test_pre_processing(mocker):
    partial_output = mocker.Mock()
    dmd = mocker.Mock(reconstructed_data=partial_output)
    dmd.fit = mocker.Mock()

    pre_post_processing = mocker.Mock()

    preproc_output = mocker.Mock()
    preprocessed_training_data = mocker.Mock()
    pre_post_processing.pre_processing = mocker.Mock(
        return_value=(preproc_output, preprocessed_training_data)
    )

    output = mocker.Mock()
    pre_post_processing.post_processing = mocker.Mock(return_value=output)

    pdmd = PrePostProcessingDMD(dmd, pre_post_processing)

    X = mocker.Mock()
    pdmd.fit(X, a=2, b=3)

    pre_post_processing.pre_processing.assert_called_once_with(X)
    dmd.fit.assert_called_once_with(preprocessed_training_data, a=2, b=3)

    assert pdmd.reconstructed_data is output
    pre_post_processing.post_processing.assert_called_once_with(
        preproc_output, partial_output
    )


def test_copy(mocker):
    dmd = mocker.Mock()
    dmd.fit = mocker.Mock()
    pdmd = PrePostProcessingDMD(dmd)
    assert copy(pdmd)._dmd == dmd


def test_deepcopy(mocker):
    dmd = mocker.Mock()
    dmd.fit = mocker.Mock()
    pdmd = PrePostProcessingDMD(dmd)
    assert deepcopy(pdmd)._dmd is not None
