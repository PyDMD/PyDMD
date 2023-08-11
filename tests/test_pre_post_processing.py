from pydmd import PrePostProcessingDMD


def test_pre_processing(mocker):
    partial_output = mocker.Mock()
    dmd = mocker.Mock(reconstructed_data=partial_output)
    dmd.fit = mocker.Mock()

    preproc_state = mocker.Mock()
    pre = mocker.Mock(return_value=preproc_state)

    output = mocker.Mock()
    post = mocker.Mock(return_value=output)

    pdmd = PrePostProcessingDMD(dmd, pre, post)

    input = mocker.Mock()
    pdmd.fit(input)
    pre.assert_called_once_with(input)
    dmd.fit.assert_called_once_with(input)

    assert pdmd.reconstructed_data is output
    post.assert_called_once_with(partial_output, preproc_state)
