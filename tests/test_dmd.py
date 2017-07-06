from unittest import TestCase
from dmd import DMD

import numpy as np

# 15 snapshot with 400 data. The matrix is 400x15 and it contains
# the following data: f1 + f2 where
# f1 = lambda x,t: sech(x+3)*(1.*np.exp(1j*2.3*t))
# f2 = lambda x,t: (sech(x)*np.tanh(x))*(2.*np.exp(1j*2.8*t))
sample_data = np.load('tests/test_datasets/input_sample.npy')


class TestDmd(TestCase):
	def test_shape(self):
		dmd = DMD()
		dmd.fit(X=sample_data)
		assert dmd.modes.shape[1] == sample_data.shape[1] - 1

	def test_truncation_shape(self):
		dmd = DMD(svd_rank=3)
		dmd.fit(X=sample_data)
		assert dmd.modes.shape[1] == 3

	def test_amplitudes_1(self):
		dmd = DMD()
		dmd.fit(X=sample_data)
		assert dmd.amplitudes.shape == (14, 14)

	def test_amplitudes_2(self):
		dmd = DMD(svd_rank=3)
		dmd.fit(X=sample_data)
		assert dmd.amplitudes.shape == (dmd.svd_rank, dmd.svd_rank)

	def test_vander(self):
		dmd = DMD(exact=True)
		dmd.fit(X=sample_data)
		assert dmd.vander.shape == (14, 15)

	def test_eigs_1(self):
		dmd = DMD()
		dmd.fit(X=sample_data)
		assert len(dmd.eigs) == 14

	def test_eigs_2(self):
		dmd = DMD(svd_rank=5)
		dmd.fit(X=sample_data)
		assert len(dmd.eigs) == 5

	def test_reconstructed_data(self):
		dmd = DMD()
		dmd.fit(X=sample_data)
		dmd_data = dmd.reconstructed_data
		assert np.allclose(dmd_data, sample_data)

	def test_plot_eigs_1(self):
		dmd = DMD()
		dmd.fit(X=sample_data)
		dmd.plot_eigs(show_axes=True, show_unit_circle=True)

	def test_plot_eigs_2(self):
		dmd = DMD()
		dmd.fit(X=sample_data)
		dmd.plot_eigs(show_axes=False, show_unit_circle=False)
