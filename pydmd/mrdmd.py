"""
Derived module from dmdbase.py for multi-resolution dmd.

Reference:
- Kutz, J. Nathan, Xing Fu, and Steven L. Brunton. Multiresolution Dynamic Mode
Decomposition. SIAM Journal on Applied Dynamical Systems 15.2 (2016): 713-735.
"""
from __future__ import division
from builtins import range
from past.utils import old_div
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

from .dmdbase import DMDBase


class MrDMD(DMDBase):
	"""
	Multi-resolution Dynamic Mode Decomposition

	:param int svd_rank: rank truncation in SVD. Default is 0, that means no
		truncation.
	:param int tlsq_rank: rank truncation computing Total Least Square. Default
		is 0, that means no truncation.
	:param bool exact: flag to compute either exact DMD or projected DMD.
		Default is False.
	:param int max_cycles: the maximum number of mode oscillations in any given
		time scale. Default is 1.
	:param int max_level: the maximum number of levels. Defualt is 6.
	"""

	def __init__(
		self, svd_rank=0, tlsq_rank=0, exact=False, max_cycles=1, max_level=6
	):
		super(MrDMD, self).__init__(svd_rank, tlsq_rank, exact)
		self.max_cycles = max_cycles
		self.max_level = max_level

	def _index_list(self, level, node):
		"""
		Private method that return the right index element from a given level
			and node.
		:param int level: the level in the binary tree.
		:param int node: the node id.
		:rtype: int
		:return: the index of the list that contains the binary tree.
		"""
		if level > self.max_level:
			raise ValueError("Invalid level: greater than `max_level`")

		if node >= 2**level:
			raise ValueError("Invalid node")

		return 2**level + node - 1

	@property
	def reconstructed_data(self):
		"""
		numpy.ndarray: DMD reconstructed_data
		"""
		data = np.sum(
			np.array(
				[
					self.partial_reconstructed_data(i)
					for i in range(self.max_level)
				]
			),
			axis=0
		)
		return data

	@property
	def modes(self):
		"""
		numpy.ndarray: the matrix that contains all the modes, stored by
		column, starting from the slowest level to the fastest one.
		"""
		return np.hstack(tuple(self._modes))

	@property
	def dynamics(self):
		"""
		numpy.ndarray: the matrix that contains the time evolution, starting
		from the slowest level to the fastest one.
		"""
		return np.vstack(
			tuple([self.partial_dynamics(i) for i in range(self.max_level)])
		)

	@property
	def eigs(self):
		"""
		numpy.ndarray: the array from the eigendecomposition of Atilde,
		starting from the slowest level to the fastest one.
		"""
		return np.concatenate(self._eigs)

	def partial_modes(self, level, node=None):
		"""
		Return the modes at the specific `level` and at the specific `node`; if
		`node` is not specified, the method returns all the modes of the given
		`level` (all the nodes).
		:param int level: the index of the level from where the modes are
			extracted.
		:param int node: the index of the node from where the modes are
			extracted; if None, the modes are extracted from all the nodes of
			the given level. Default is None.
		"""
		if node:
			return self._modes[self._index_list(level, node)]

		indeces = [self._index_list(level, i) for i in range(2**level)]
		return np.hstack(tuple([self._modes[idx] for idx in indeces]))

	def partial_dynamics(self, level, node=None):
		"""
		Return the time evolution of the specific `level` and of the specific
		`node`; if `node` is not specified, the method returns the time evolution
		of the given `level` (all the nodes).
		:param int level: the index of the level from where the time evolution
			is extracted.
		:param int node: the index of the node from where the time evolution is
			extracted; if None, the time evolution is extracted from all the
			nodes of the given level. Default is None.
		"""

		def dynamic(eigs, amplitudes, step, nsamples):
			omega = old_div(np.log(np.power(eigs, old_div(1., step))), self.original_time['dt'])
			partial_timestep = np.arange(nsamples) * self.dmd_time['dt']
			vander = np.exp(np.multiply(*np.meshgrid(omega, partial_timestep)))
			return (vander * amplitudes).T

		if node:
			indeces = [self._index_list(level, node)]
		else:
			indeces = [self._index_list(level, i) for i in range(2**level)]

		level_dynamics = [
			dynamic(
				self._eigs[idx], self._b[idx], self._steps[idx],
				self._nsamples[idx]
			) for idx in indeces
		]
		return scipy.linalg.block_diag(*level_dynamics)

	def partial_eigs(self, level, node=None):
		"""
		Return the eigenvalues of the specific `level` and of the specific
		`node`; if `node` is not specified, the method returns the eigenvalues
		of the given `level` (all the nodes).
		:param int level: the index of the level from where the eigenvalues is
			extracted.
		:param int node: the index of the node from where the eigenvalues is
			extracted; if None, the time evolution is extracted from all the
			nodes of the given level. Default is None.
		"""
		if level >= self.max_level:
			raise ValueError(
				'The level input parameter ({}) has to be less then the max_level ({}). '
				'Remember that the starting index is 0'.format(
					level, self.max_level
				)
			)
		if node:
			return self._eigs[self._index_list(level, node)]

		indeces = [self._index_list(level, i) for i in range(2**level)]
		return np.concatenate([self._eigs[idx] for idx in indeces])

	def partial_reconstructed_data(self, level, node=None):
		"""
		Return the reconstructed data computed using the modes and the time
		evolution at the specific `level` and at the specific `node`; if `node`
		is not specified, the method returns the reconstructed data
		of the given `level` (all the nodes).
		:param int level: the index of the level.
		:param int node: the index of the node from where the time evolution is
			extracted; if None, the time evolution is extracted from all the
			nodes of the given level. Default is None.

		"""
		if level >= self.max_level:
			raise ValueError(
				'The level input parameter ({}) has to be less then the max_level ({}). '
				'Remember that the starting index is 0'.format(
					level, self.max_level
				)
			)
		modes = self.partial_modes(level, node)
		dynamics = self.partial_dynamics(level, node)

		return modes.dot(dynamics)

	def fit(self, X, Y=None):
		"""
		Compute the Dynamic Modes Decomposition to the input data.

		:param iterable or numpy.ndarray X: the input snapshots.
		:param itarable or numpy.ndarray Y: if specified, it provides the
			snapshots at the next time step. Its dimension must be equal to X.
			Default is None.
		"""
		self._fit_read_input(X, Y)

		# To avoid recursion function, use FIFO list to simulate the tree
		# structure
		data_queue = []
		Xraw = np.append(self._X, self._Y[:, -1].reshape(-1, 1), axis=1)
		data_queue.append(Xraw)

		current_level = 1

		# Reset the lists
		self._eigs = []
		self._Atilde = []
		self._modes = []
		self._b = []
		self._nsamples = []
		self._steps = []

		while data_queue:
			Xraw = data_queue.pop(0)

			n_samples = Xraw.shape[1]
			# subsamples frequency to detect slow modes
			nyq = 8 * self.max_cycles

			try:
				step = int(np.floor(old_div(n_samples, nyq)))
				Xsub = Xraw[:, ::step]
				Xc = Xsub[:, :-1]
				Yc = Xsub[:, 1:]

				Xc, Yc = self._compute_tlsq(Xc, Yc, self.tlsq_rank)

				U, s, V = self._compute_svd(Xc, self.svd_rank)

				#---------------------------------------------------------------
				# DMD Modes
				#---------------------------------------------------------------
				Sinverse = np.diag(old_div(1., s))
				Atilde = U.T.conj().dot(Yc).dot(V).dot(Sinverse)

				# Exact or projected DMD
				basis = Yc.dot(V).dot(Sinverse) if self.exact else U

				eigs, mode_coeffs = np.linalg.eig(Atilde)

				rho = old_div(float(self.max_cycles), n_samples)
				slow_modes = (np.abs(old_div(np.log(eigs), (2. * np.pi * step)))) <= rho
				modes = basis.dot(mode_coeffs)[:, slow_modes]
				eigs = eigs[slow_modes]

				#---------------------------------------------------------------
				# DMD Amplitudes and Dynamics
				#---------------------------------------------------------------
				Vand = np.vander(np.power(eigs, old_div(1., step)), n_samples, True)
				b = np.linalg.lstsq(modes, Xc[:, 0])[0]

				Psi = (Vand.T * b).T
			except:
				modes = np.zeros((Xraw.shape[0], 1))
				Psi = np.zeros((1, Xraw.shape[1]))
				Atilde = np.zeros(0)
				eigs = np.zeros(0)
				b = np.zeros(0)
				step = 0

			self._modes.append(modes)
			self._b.append(b)
			self._Atilde.append(Atilde)
			self._eigs.append(eigs)
			self._nsamples.append(n_samples)
			self._steps.append(step)

			Xraw -= modes.dot(Psi)

			if current_level < 2**(self.max_level - 1):
				current_level += 1
				half = int(np.ceil(old_div(Xraw.shape[1], 2)))
				data_queue.append(Xraw[:, :half])
				data_queue.append(Xraw[:, half:])

		self.dmd_time = {'t0': 0, 'tend': self._X.shape[1] + 1, 'dt': 1}
		self.original_time = {'t0': 0, 'tend': self._X.shape[1] + 1, 'dt': 1}

		return self

	def plot_eigs(
		self,
		show_axes=True,
		show_unit_circle=True,
		figsize=(8, 8),
		title='',
		level=None,
		node=None
	):
		"""
		Plot the eigenvalues.

		:param show_axes bool: if True, the axes will be showed in the plot.
			Default is True.
		:param show_axes bool: if True, the circle with unitary radius and
			center in the origin will be showed. Default is True.
		"""
		if self._eigs is None:
			raise ValueError(
				'The eigenvalues have not been computed.'
				'You have to perform the fit method.'
			)

		if level:
			peigs = self.partial_eigs(level=level, node=node)
		else:
			peigs = self.eigs

		plt.figure(figsize=figsize)
		plt.title(title)
		fig = plt.gcf()
		ax = plt.gca()

		if not level:
			cmap = plt.get_cmap('viridis')
			colors = [cmap(i) for i in np.linspace(0, 1, self.max_level)]

			points = []
			for lvl in range(self.max_level):
				indeces = [self._index_list(lvl, i) for i in range(2**lvl)]
				eigs = np.concatenate([self._eigs[idx] for idx in indeces])

				points.append(
					ax.plot(eigs.real, eigs.imag, '.', color=colors[lvl])[0]
				)
		else:
			points = []
			points.append(
				ax.plot(peigs.real, peigs.imag, 'bo', label='Eigenvalues')[0]
			)

		# set limits for axis
		limit = np.max(np.ceil(np.absolute(peigs)))
		ax.set_xlim((-limit, limit))
		ax.set_ylim((-limit, limit))

		plt.ylabel('Imaginary part')
		plt.xlabel('Real part')

		if show_unit_circle:
			unit_circle = plt.Circle(
				(0., 0.), 1., color='green', fill=False, linestyle='--'
			)
			ax.add_artist(unit_circle)

		# Dashed grid
		gridlines = ax.get_xgridlines() + ax.get_ygridlines()
		for line in gridlines:
			line.set_linestyle('-.')
		ax.grid(True)

		ax.set_aspect('equal')

		# x and y axes
		if show_axes:
			ax.annotate(
				'',
				xy=(np.max([limit * 0.8, 1.]), 0.),
				xytext=(np.min([-limit * 0.8, -1.]), 0.),
				arrowprops=dict(arrowstyle="->")
			)
			ax.annotate(
				'',
				xy=(0., np.max([limit * 0.8, 1.])),
				xytext=(0., np.min([-limit * 0.8, -1.])),
				arrowprops=dict(arrowstyle="->")
			)

		# legend
		if level:
			labels = ['Eigenvalues - level {}'.format(level)]
		else:
			labels = [
				'Eigenvalues - level {}'.format(i)
				for i in range(self.max_level)
			]

		if show_unit_circle:
			points += [unit_circle]
			labels += ['Unit circle']

		ax.add_artist(plt.legend(points, labels, loc='best'))
		plt.show()
