import numpy as np
from pydmd.preprocessing.hankel import (
    hankel_preprocessing,
    _preprocessing,
    _reconstructions,
)
from pydmd.utils import pseudo_hankel_matrix
from pydmd import DMD

sample_data = np.load("tests/test_datasets/input_sample.npy").real
_d = 3


def test_rec_method_first():
    dmd = hankel_preprocessing(
        DMD(svd_rank=-1), d=_d, reconstruction_method="first"
    )
    dmd.fit(X=sample_data)

    rec = dmd.reconstructed_data
    allrec = _reconstructions(_preprocessing({}, sample_data, d=_d)[0], d=_d)
    for i in range(rec.shape[1] - _d):
        np.testing.assert_allclose(rec[:, i].real, allrec[i, 0])
    np.testing.assert_allclose(rec[:, -2].real, allrec[-2, 1])
    np.testing.assert_allclose(rec[:, -1].real, allrec[-1, 2])


def test_rec_method_mean():
    dmd = hankel_preprocessing(
        DMD(svd_rank=-1), d=_d, reconstruction_method="mean"
    )
    dmd.fit(X=sample_data)

    rec = dmd.reconstructed_data
    preprocessed = _preprocessing({}, sample_data, d=_d)[0]
    allrec = _reconstructions(preprocessed, d=_d)
    np.testing.assert_allclose(rec, np.nanmean(allrec, axis=1).T)


def test_rec_method_weighted():
    dmd = hankel_preprocessing(
        DMD(svd_rank=-1), d=_d, reconstruction_method=[10, 20, 30]
    )
    dmd.fit(X=sample_data)

    rec = dmd.reconstructed_data
    preprocessed = _preprocessing({}, sample_data, d=_d)[0]
    allrec = _reconstructions(preprocessed, d=_d)
    np.testing.assert_allclose(
        rec, np.ma.average(allrec, axis=1, weights=[10, 20, 30]).T
    )
