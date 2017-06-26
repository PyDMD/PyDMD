from unittest import TestCase
from dmd.dmd import dmd

import numpy as np

sample_data = np.load('tests/test_datasets/input_sample.npy')


class TestDmd(TestCase):
	def test_dmd_shape(self):
		Phi, B, V = dmd(sample_data)
		assert Phi.shape[1] == sample_data.shape[1] - 1

	def test_dmd_truncation(self):
		Phi, B, V = dmd(sample_data, k=2)
		assert Phi.shape[1] == 2

	def test_dmd_decomposition(self):
		Phi, B, V = dmd(sample_data)
		dmd_data = Phi.dot(B).dot(V)
		assert np.allclose(dmd_data, sample_data)
