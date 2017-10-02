from unittest import TestCase
from dmd.mrdmd import MrDMD

import numpy as np


def create_data():
	x = np.linspace(-10, 10, 80)
	t = np.linspace(0, 20, 1600)
	Xm, Tm = np.meshgrid(x, t)

	D = np.exp(-np.power(Xm / 2, 2)) * np.exp(0.8j * Tm)
	D += np.sin(0.9 * Xm) * np.exp(1j * Tm)
	D += np.cos(1.1 * Xm) * np.exp(2j * Tm)
	D += 0.6 * np.sin(1.2 * Xm) * np.exp(3j * Tm)
	D += 0.6 * np.cos(1.3 * Xm) * np.exp(4j * Tm)
	D += 0.2 * np.sin(2.0 * Xm) * np.exp(6j * Tm)
	D += 0.2 * np.cos(2.1 * Xm) * np.exp(8j * Tm)
	D += 0.1 * np.sin(5.7 * Xm) * np.exp(10j * Tm)
	D += 0.1 * np.cos(5.9 * Xm) * np.exp(12j * Tm)
	D += 0.1 * np.random.randn(*Xm.shape)
	D += 0.03 * np.random.randn(*Xm.shape)
	D += 5 * np.exp(-np.power((Xm + 5) / 5, 2)
					) * np.exp(-np.power((Tm - 5) / 5, 2))
	D[:800, 40:] += 2
	D[200:600, 50:70] -= 3
	D[800:, :40] -= 2
	D[1000:1400, 10:30] += 3
	D[1000:1080, 50:70] += 2
	D[1160:1240, 50:70] += 2
	D[1320:1400, 50:70] += 2
	return D.T


sample_data = create_data()


class TestMrDmd(TestCase):
	def test_shape_modes(self):
		level = 5
		dmd = MrDMD(svd_rank=1, max_level=level, max_cycles=2)
		dmd.fit(X=sample_data)
		assert dmd.modes.shape == (sample_data.shape[0], 2**level - 1)

	def test_shape_dynamics(self):
		level = 5
		dmd = MrDMD(svd_rank=1, max_level=level, max_cycles=2)
		dmd.fit(X=sample_data)
		assert dmd.dynamics.shape == (2**level - 1, sample_data.shape[1])

	def test_reconstructed_data(self):
		dmd = MrDMD(svd_rank=0, max_level=6, max_cycles=2, exact=True)
		dmd.fit(X=sample_data)
		dmd_data = dmd.reconstructed_data
		print(dmd_data.shape)
		print(sample_data.shape)
		norm_err = (
			np.linalg.norm(sample_data - dmd_data) /
			np.linalg.norm(sample_data)
		)
		assert norm_err < 1

	def test_partial_modes1(self):
		max_level = 5
		level = 2
		rank = 2
		dmd = MrDMD(svd_rank=rank, max_level=max_level, max_cycles=2)
		dmd.fit(X=sample_data)
		pmodes = dmd.partial_modes(level)
		assert pmodes.shape == (sample_data.shape[0], 2**level * rank)

	def test_partial_modes2(self):
		max_level = 5
		level = 2
		rank = 2
		dmd = MrDMD(svd_rank=rank, max_level=max_level, max_cycles=2)
		dmd.fit(X=sample_data)
		pmodes = dmd.partial_modes(level, 3)
		assert pmodes.shape == (sample_data.shape[0], rank)

	def test_partial_dynamics1(self):
		max_level = 5
		level = 2
		rank = 2
		dmd = MrDMD(svd_rank=rank, max_level=max_level, max_cycles=2)
		dmd.fit(X=sample_data)
		pdynamics = dmd.partial_dynamics(level)
		assert pdynamics.shape == (2**level * rank, sample_data.shape[1])

	def test_partial_dynamics2(self):
		max_level = 5
		level = 2
		rank = 2
		dmd = MrDMD(svd_rank=rank, max_level=max_level, max_cycles=2)
		dmd.fit(X=sample_data)
		pdynamics = dmd.partial_dynamics(level, 3)
		assert pdynamics.shape == (rank, sample_data.shape[1] / 2**level)

	def test_eigs2(self):
		max_level = 5
		level = 2
		rank = -1
		dmd = MrDMD(svd_rank=rank, max_level=max_level, max_cycles=2)
		dmd.fit(X=sample_data)
		assert dmd.eigs.ndim == 1

	def test_partial_eigs1(self):
		max_level = 5
		level = 2
		rank = 2
		dmd = MrDMD(svd_rank=rank, max_level=max_level, max_cycles=2)
		dmd.fit(X=sample_data)
		peigs = dmd.partial_eigs(level)
		assert peigs.shape == (rank * 2**level, )

	def test_partial_eigs2(self):
		max_level = 5
		level = 2
		rank = 2
		dmd = MrDMD(svd_rank=rank, max_level=max_level, max_cycles=2)
		dmd.fit(X=sample_data)
		peigs = dmd.partial_eigs(level, 3)
		assert peigs.shape == (rank, )

	def test_partial_reconstructed1(self):
		max_level = 5
		level = 2
		rank = 2
		dmd = MrDMD(svd_rank=rank, max_level=max_level, max_cycles=2)
		dmd.fit(X=sample_data)
		pdata = dmd.partial_reconstructed_data(level)
		assert pdata.shape == sample_data.shape

	def test_partial_reconstructed2(self):
		max_level = 5
		level = 2
		rank = 2
		dmd = MrDMD(svd_rank=rank, max_level=max_level, max_cycles=2)
		dmd.fit(X=sample_data)
		pdata = dmd.partial_reconstructed_data(level, 3)
		assert pdata.shape == (
			sample_data.shape[0], sample_data.shape[1] / 2**level
		)

	def test_wrong_partial_reconstructed(self):
		max_level = 5
		level = 2
		rank = 2
		dmd = MrDMD(svd_rank=rank, max_level=max_level, max_cycles=2)
		dmd.fit(X=sample_data)
		with self.assertRaises(ValueError):
			pdata = dmd.partial_reconstructed_data(max_level, 2)

	def test_wrong_level(self):
		max_level = 5
		dmd = MrDMD(max_level=max_level)
		dmd.fit(sample_data)
		with self.assertRaises(ValueError):
			dmd.partial_modes(max_level + 1)

	def test_wrong_bin(self):
		max_level = 5
		level = 2
		dmd = MrDMD(max_level=max_level)
		dmd.fit(sample_data)
		with self.assertRaises(ValueError):
			dmd.partial_modes(level=level, node=2**level)

	def test_wrong_plot_eig1(self):
		dmd = MrDMD(svd_rank=-1, max_level=7, max_cycles=1)
		dmd.fit(X=sample_data)
		with self.assertRaises(ValueError):
			dmd.plot_eigs(
				show_axes=True, show_unit_circle=True, figsize=(8, 8), level=7
			)

	def test_wrong_plot_eig2(self):
		dmd = MrDMD(svd_rank=1, max_level=7, max_cycles=1)
		with self.assertRaises(ValueError):
			dmd.plot_eigs()

	def test_plot_eig1(self):
		dmd = MrDMD(svd_rank=-1, max_level=7, max_cycles=1)
		dmd.fit(X=sample_data)
		dmd.plot_eigs(show_axes=True, show_unit_circle=True, figsize=(8, 8))

	def test_plot_eig2(self):
		dmd = MrDMD(svd_rank=-1, max_level=7, max_cycles=1)
		dmd.fit(X=sample_data)
		dmd.plot_eigs(show_axes=True, show_unit_circle=False, title='Title')

	def test_plot_eig3(self):
		dmd = MrDMD(svd_rank=-1, max_level=7, max_cycles=1)
		dmd.fit(X=sample_data)
		dmd.plot_eigs(show_axes=False, show_unit_circle=False, level=1, node=0)
