import numpy as np
import pytest

from pydmd import DMD
from pydmd.preprocessing.hankel import hankel_preprocessing

sample_data = np.load("tests/test_datasets/input_sample.npy").real


@pytest.mark.parametrize(
    "reconstruction_method", ["first", "mean", [10, 20, 30]]
)
def test_hankel(reconstruction_method):
    print(reconstruction_method)
    dmd = hankel_preprocessing(
        DMD(svd_rank=-1), d=3, reconstruction_method=reconstruction_method
    )
    dmd.fit(X=sample_data)
    np.testing.assert_allclose(dmd.reconstructed_data, sample_data)
