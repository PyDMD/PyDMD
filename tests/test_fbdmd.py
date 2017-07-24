from unittest import TestCase
from dmd.fbdmd import FbDMD
import numpy as np


def create_noisy_data():
	mu = 0.
	sigma = 0.	# noise standard deviation
	m = 100	 # number of snapshot
	noise = np.random.normal(mu, sigma, m)	# gaussian noise
	A = np.array([[1., 1.], [-1., 2.]])
	A /= np.sqrt(3)
	n = 2
	X = np.zeros((n, m))
	X[:, 0] = np.array([0.5, 1.])
	# evolve the system and perturb the data with noise
	for k in range(1, m):
		X[:, k] = A.dot(X[:, k - 1])
		X[:, k - 1] += noise[k - 1]
	return X


sample_data = create_noisy_data()


class TestFbDmd(TestCase):
	def test_modes_shape(self):
		dmd = FbDMD()
		dmd.fit(X=sample_data)
		assert dmd.modes.shape[1] == 2

	def test_truncation_shape(self):
		dmd = FbDMD(svd_rank=1)
		dmd.fit(X=sample_data)
		assert dmd.modes.shape[1] == 1

	def test_amplitudes_1(self):
		dmd = FbDMD()
		dmd.fit(X=sample_data)
		assert dmd.amplitudes.shape == (2, 2)

	def test_amplitudes_2(self):
		dmd = FbDMD(svd_rank=2)
		dmd.fit(X=sample_data)
		assert dmd.amplitudes.shape == (dmd.svd_rank, dmd.svd_rank)

	def test_vander(self):
		dmd = FbDMD()
		dmd.fit(X=sample_data)
		assert dmd.vander.shape == (2, 100)

	def test_eigs_1(self):
		dmd = FbDMD()
		dmd.fit(X=sample_data)
		assert len(dmd.eigs) == 2

	def test_eigs_2(self):
		dmd = FbDMD(svd_rank=1)
		dmd.fit(X=sample_data)
		assert len(dmd.eigs) == 1

	def test_eigs_modulus_1(self):
		dmd = FbDMD(svd_rank=0)
		dmd.fit(X=sample_data)
		np.testing.assert_almost_equal(
			np.linalg.norm(dmd.eigs[0]), 1., decimal=6
		)

	def test_eigs_modulus_2(self):
		dmd = FbDMD(svd_rank=0, exact=True)
		dmd.fit(X=sample_data)
		np.testing.assert_almost_equal(
			np.linalg.norm(dmd.eigs[1]), 1., decimal=6
		)

	def test_reconstructed_data(self):
		dmd = FbDMD(exact=True)
		dmd.fit(X=sample_data)
		dmd_data = dmd.reconstructed_data
		dmd_data_correct = np.load('tests/test_datasets/fbdmd_data.npy')
		assert np.allclose(dmd_data, dmd_data_correct)

	def test_plot_eigs_1(self):
		dmd = FbDMD()
		dmd.fit(X=sample_data)
		dmd.plot_eigs(show_axes=True, show_unit_circle=True)

	def test_plot_eigs_2(self):
		dmd = FbDMD()
		dmd.fit(X=sample_data)
		dmd.plot_eigs(show_axes=False, show_unit_circle=False)
