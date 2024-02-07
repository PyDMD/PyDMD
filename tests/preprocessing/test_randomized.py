import numpy as np
from numpy.testing import assert_allclose

from pydmd import DMD, RDMD
from pydmd.preprocessing import randomized_preprocessing

sample_data = np.load("tests/test_datasets/input_sample.npy")


def test_reconstruction():
    """
    Test that a pydmd module preprocessed with
    randomization produces accurate reconstructions.
    """
    pdmd = randomized_preprocessing(DMD(svd_rank=2), svd_rank=2)
    pdmd.fit(sample_data)
    assert_allclose(pdmd.reconstructed_data, sample_data)


def test_shape():
    """
    Test that a pydmd module preprocessed with
    randomization stores data with the expected shape.
    """
    pdmd = randomized_preprocessing(
        DMD(svd_rank=2),
        svd_rank=2,
        oversampling=10,
    )
    pdmd.fit(sample_data)
    assert pdmd.snapshots.shape == (12, sample_data.shape[-1])


def test_rdmd():
    """
    Test that RDMD and DMD with randomization are essentially the same.
    Compares snapshots, eigenvalues, and reconstructions.
    """
    rdmd = RDMD(svd_rank=2, oversampling=10, power_iters=2, seed=1234)
    rdmd.fit(sample_data)

    pdmd = randomized_preprocessing(
        DMD(svd_rank=2),
        svd_rank=2,
        oversampling=10,
        power_iters=2,
        seed=1234,
    )
    pdmd.fit(sample_data)

    assert_allclose(
        pdmd.snapshots,
        rdmd.compression_matrix.dot(rdmd.snapshots),
    )
    assert_allclose(np.sort(pdmd.eigs), np.sort(rdmd.eigs))
    assert_allclose(pdmd.reconstructed_data, rdmd.reconstructed_data)
