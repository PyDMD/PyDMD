from unittest import TestCase
from dmd.dmd import DMD

import numpy as np

sample_data = np.load('tests/test_datasets/input_sample.npy')
# print sample_data.shape


class TestDmd(TestCase):
	def test_dmd_shape(self):
		dmd = DMD()
		dmd.fit(X=sample_data)
		assert dmd.modes.shape[1] == sample_data.shape[1] - 1

	def test_dmd_truncation_shape(self):
		dmd = DMD(k=3)
		dmd.fit(X=sample_data)
		assert dmd.modes.shape[1] == 3

	def test_dmd_decomposition(self):
		dmd = DMD()
		dmd.fit(X=sample_data)
		dmd_data = dmd.reconstructed_data
		assert np.allclose(dmd_data, sample_data)
