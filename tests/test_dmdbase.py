from unittest import TestCase
from dmd.dmdbase import DMDBase

import numpy as np

# 15 snapshot with 400 data. The matrix is 400x15 and it contains
# the following data: f1 + f2 where
# f1 = lambda x,t: sech(x+3)*(1.*np.exp(1j*2.3*t))
# f2 = lambda x,t: (sech(x)*np.tanh(x))*(2.*np.exp(1j*2.8*t))
sample_data = np.load('tests/test_datasets/input_sample.npy')


class TestDmdBase(TestCase):
	def test_svd_rank_default(self):
		dmd = DMDBase()
		assert dmd.svd_rank == 0

	def test_svd_rank(self):
		dmd = DMDBase(svd_rank=3)
		assert dmd.svd_rank == 3

	def test_tlsq_rank_default(self):
		dmd = DMDBase()
		assert dmd.tlsq_rank == 0

	def test_tlsq_rank(self):
		dmd = DMDBase(tlsq_rank=2)
		assert dmd.tlsq_rank == 2

	def test_exact_default(self):
		dmd = DMDBase()
		assert dmd.exact == False

	def test_exact(self):
		dmd = DMDBase(exact=True)
		assert dmd.exact == True

	def test_fit(self):
		dmd = DMDBase(exact=False)
		with self.assertRaises(NotImplementedError):
			dmd.fit()

	def test_plot_eigs(self):
		dmd = DMDBase()
		with self.assertRaises(ValueError):
			dmd.plot_eigs(show_axes=True, show_unit_circle=True)

	def test_plot_modes(self):
		dmd = DMDBase()
		with self.assertRaises(ValueError):
			dmd.plot_modes(range(4), range(3))
