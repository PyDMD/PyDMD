from unittest import TestCase
from pydmd.dmd import DMD

import numpy as np

# 15 snapshot with 400 data. The matrix is 400x15 and it contains
# the following data: f1 + f2 where
# f1 = lambda x,t: sech(x+3)*(1.*np.exp(1j*2.3*t))
# f2 = lambda x,t: (sech(x)*np.tanh(x))*(2.*np.exp(1j*2.8*t))
sample_data = np.load('tests/test_datasets/input_sample.npy')


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


noisy_data = create_noisy_data()


class TestDmd(TestCase):
	def test_shape(self):
		dmd = DMD(svd_rank=-1)
		dmd.fit(X=sample_data)
		assert dmd.modes.shape[1] == sample_data.shape[1] - 1

	def test_truncation_shape(self):
		dmd = DMD(svd_rank=3)
		dmd.fit(X=sample_data)
		assert dmd.modes.shape[1] == 3

	def test_eigs_1(self):
		dmd = DMD(svd_rank=-1)
		dmd.fit(X=sample_data)
		assert len(dmd.eigs) == 14

	def test_eigs_2(self):
		dmd = DMD(svd_rank=5)
		dmd.fit(X=sample_data)
		assert len(dmd.eigs) == 5

	def test_dynamics(self):
		dmd = DMD(svd_rank=5)
		dmd.fit(X=sample_data)
		assert dmd.dynamics.shape == (5, sample_data.shape[1])

	def test_reconstructed_data(self):
		dmd = DMD()
		dmd.fit(X=sample_data)
		dmd_data = dmd.reconstructed_data
		assert np.allclose(dmd_data, sample_data)

	def test_original_timesteps(self):
		dmd = DMD()
		dmd.fit(X=sample_data)
		assert np.array_equal(
			dmd.original_timesteps, np.arange(sample_data.shape[1])
		)

	def test_dmd_time(self):
		dmd = DMD()
		dmd.fit(X=sample_data)
		dmd.dmd_time['t0'] = 10
		dmd.dmd_time['tend'] = 14
		expected_data = sample_data[:, -5:]
		assert np.allclose(dmd.reconstructed_data, expected_data)

	def test_plot_eigs_1(self):
		dmd = DMD()
		dmd.fit(X=sample_data)
		dmd.plot_eigs(show_axes=True, show_unit_circle=True)

	def test_plot_eigs_2(self):
		dmd = DMD()
		dmd.fit(X=sample_data)
		dmd.plot_eigs(show_axes=False, show_unit_circle=False)

	def test_plot_modes_1(self):
		dmd = DMD()
		dmd.fit(X=sample_data)
		with self.assertRaises(ValueError):
			dmd.plot_modes_2D()

	def test_plot_modes_2(self):
		dmd = DMD(svd_rank=-1)
		dmd.fit(X=sample_data)
		dmd.plot_modes_2D((1, 2, 5), x=np.arange(20), y=np.arange(20))

	def test_plot_modes_3(self):
		dmd = DMD()
		snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
		dmd.fit(X=snapshots)
		dmd.plot_modes_2D()

	def test_plot_snapshots_1(self):
		dmd = DMD()
		dmd.fit(X=sample_data)
		with self.assertRaises(ValueError):
			dmd.plot_snapshots_2D()

	def test_plot_snapshots_2(self):
		dmd = DMD(svd_rank=-1)
		dmd.fit(X=sample_data)
		dmd.plot_snapshots_2D((1, 2, 5), x=np.arange(20), y=np.arange(20))

	def test_plot_snapshots_3(self):
		dmd = DMD()
		snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
		dmd.fit(X=snapshots)
		dmd.plot_snapshots_2D()

	def test_tdmd_plot(self):
		dmd = DMD(tlsq_rank=3)
		dmd.fit(X=sample_data)
		dmd.plot_eigs(show_axes=False, show_unit_circle=False)
