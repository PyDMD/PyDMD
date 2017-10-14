"""
Base module for the DMD: `fit` method must be implemented in inherited classes
"""
import os
import numpy as np
import matplotlib.pyplot as plt


class DMDBase(object):
	"""
	Dynamic Mode Decomposition base class.

	:param int svd_rank: rank truncation in SVD. If 0, the method computes the
		optimal rank and uses it for truncation; if positive number, the method
		uses the argument for the truncation; if -1, the method does not
		compute truncation.
	:param int tlsq_rank: rank truncation computing Total Least Square. Default
		is 0, that means no truncation.
	:param bool exact: flag to compute either exact DMD or projected DMD.
		Default is False.
	:param original_time: dictionary that contains information about the time
		window where the system is sampled: `t0` is the time of the first input
		snapshot, `tend` is the time of the last input snapshot and `dt` is the
		delta time between the snapshots.
	:param dmd_time: dictionary that contains information about the time
		window where the system is reconstructed: `t0` is the time of the first
		approximated solition, `tend` is the time of the last approximated
		solution and `dt` is the delta time between the approximated solutions.
	"""

	def __init__(self, svd_rank=0, tlsq_rank=0, exact=False):
		self.svd_rank = svd_rank
		self.tlsq_rank = tlsq_rank
		self.exact = exact
		self.original_time = None
		self.dmd_time = None

		self._eigs = None
		self._Atilde = None
		self._modes = None	# Phi
		self._b = None	# amplitudes
		self._X = None
		self._Y = None
		self._snapshots_shape = None

	@property
	def dmd_timesteps(self):
		"""
		numpy.ndarray: the time intervals of the reconstructed system.
		"""
		return np.arange(
			self.dmd_time['t0'], self.dmd_time['tend'] + self.dmd_time['dt'],
			self.dmd_time['dt']
		)

	@property
	def original_timesteps(self):
		"""
		numpy.ndarray: the time intervals of the original snapshots.
		"""
		return np.arange(
			self.original_time['t0'],
			self.original_time['tend'] + self.original_time['dt'],
			self.original_time['dt']
		)

	@property
	def modes(self):
		"""
		numpy.ndarray: the matrix that contains the DMD modes, stored by column.
		"""
		return self._modes

	@property
	def atilde(self):
		"""
		numpy.ndarray: the reduced Koopman operator A
		"""
		return self._Atilde

	@property
	def eigs(self):
		"""
		numpy.ndarray: the eigenvalues from the eigendecomposition of Atilde
		"""
		return self._eigs

	@property
	def dynamics(self):
		"""
		numpy.ndarray: the matrix that contains all the time evolution, stored
		by row.
		"""
		omega = np.log(self.eigs) / self.original_time['dt']
		vander = np.exp(np.multiply(*np.meshgrid(omega, self.dmd_timesteps)))
		return (vander * self._b).T

	@property
	def reconstructed_data(self):
		"""
		numpy.ndarray: DMD reconstructed data.
		"""
		return self.modes.dot(self.dynamics)

	def fit(self, X, Y=None):
		"""
		Abstract method to fit the snapshots matrices.

		Not implemented, it has to be implemented in subclasses.
		"""
		raise NotImplementedError(
			'Subclass must implement abstract method {}.fit'.
			format(self.__class__.__name__)
		)

	def _fit_read_input(self, X, Y=None):
		"""
		Private method that takes as input the snapshots and stores them into a
		2D matrix, by column. If the input data is already formatted as 2D
		array, the method saves it, otherwise it also saves the original
		snapshots shape and reshapes the snapshots.

		:param iterable or numpy.ndarray X: the input snapshots.
		:param itarable or numpy.ndarray Y: if specified, it provides the
			snapshots at the next time step. Its dimension must be equal to X.
			Default is None.
		"""

		# If the data is already 2D ndarray
		if isinstance(X, np.ndarray) and X.ndim == 2:
			if Y is None:
				self._X = X[:, :-1]
				self._Y = X[:, 1:]
			else:
				self._X = X
				self._Y = Y
			return

		self._snapshots_shape = X[0].shape
		reshapedX = np.transpose([snapshot.reshape(
			-1,
		) for snapshot in X])

		if Y is None:
			self._Y = reshapedX[:, 1:]
			self._X = reshapedX[:, :-1]
		else:
			self._X = reshapedX
			self._Y = np.transpose([snapshot.reshape(
				-1,
			) for snapshot in Y])

	@staticmethod
	def _compute_tlsq(X, Y, tlsq_rank):
		"""
		Compute Total Least Square

		:param X numpy.ndarray: the first matrix;
		:param X numpy.ndarray: the second matrix;
		:param tlsq_rank int: the rank for the truncation;

		References:
		https://arxiv.org/pdf/1703.11004.pdf
		https://arxiv.org/pdf/1502.03854.pdf
		"""
		# Do not perform tlsq
		if tlsq_rank is 0:
			return X, Y

		V = np.linalg.svd(np.append(X, Y, axis=0), full_matrices=False)[-1]
		rank = min(tlsq_rank, V.shape[0])
		VV = V[:rank, :].conj().T.dot(V[:rank, :])

		return X.dot(VV), Y.dot(VV)

	@staticmethod
	def _compute_svd(X, svd_rank):
		"""
		Truncated Singular Value Decomposition

		:param X numpy.ndarray: the matrix to decompose;
		:param svd_rank int: the rank for the truncation; If 0, the method
			computes the optimal rank and uses it for truncation; if positive
			number, the method uses the argument for the truncation; if -1, the
			method does not compute truncation.

		References:
		- Gavish, Matan, and David L. Donoho, The optimal hard threshold for
		singular values is, IEEE Transactions on Information Theory 60.8
		(2014): 5040-5053.

		"""
		U, s, V = np.linalg.svd(X, full_matrices=False)
		V = V.conj().T

		if svd_rank is 0:
			omega = lambda x: 0.56 * x**3 - 0.95 * x**2 + 1.82 * x + 1.43
			beta = np.divide(*sorted(X.shape))
			tau = np.median(s) * omega(beta)
			rank = np.sum(s > tau)
		elif svd_rank > 0:
			rank = min(svd_rank, U.shape[1])
		else:
			rank = X.shape[1]

		U = U[:, :rank]
		V = V[:, :rank]
		s = s[:rank]

		return U, s, V

	def plot_eigs(
		self, show_axes=True, show_unit_circle=True, figsize=(8, 8), title=''
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

		plt.figure(figsize=figsize)
		plt.title(title)
		fig = plt.gcf()
		ax = plt.gca()

		points, = ax.plot(
			self._eigs.real, self._eigs.imag, 'bo', label='Eigenvalues'
		)

		# set limits for axis
		limit = np.max(np.ceil(np.absolute(self._eigs)))
		ax.set_xlim((-limit, limit))
		ax.set_ylim((-limit, limit))

		plt.ylabel('Imaginary part')
		plt.xlabel('Real part')

		if show_unit_circle:
			unit_circle = plt.Circle(
				(0., 0.),
				1.,
				color='green',
				fill=False,
				label='Unit circle',
				linestyle='--'
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
		if show_unit_circle:
			ax.add_artist(
				plt.legend(
					[points, unit_circle], ['Eigenvalues', 'Unit circle'],
					loc=1
				)
			)
		else:
			ax.add_artist(plt.legend([points], ['Eigenvalues'], loc=1))

		plt.show()

	def plot_modes_2D(
		self, index_mode=None, filename=None, x=None, y=None, order='C'
	):
		"""
		Plot the DMD Modes.

		:param index_mode int or sequence of int: the index of the modes to
			plot. By default, all the modes are plotted.
		:param filename str: filename
		:param x numpy.ndarray: domain abscissa
		:param y numpy.ndarray: domain ordinate
		:param order str: {'C', 'F', 'A'}, default 'C'.
			Read the elements of snapshots using this index order, and place
			the elements into the reshaped array using this index order. It
			has to be the same used to store the snapshot. 'C' means to read/
			write the elements using C-like index order, with the last axis
			index changing fastest, back to the first axis index changing slowest.
			'F' means to read / write the elements using Fortran-like index order,
			with the first index changing fastest, and the last index changing
			slowest. Note that the 'C' and 'F' options take no account of the
			memory layout of the underlying array, and only refer to the order
			of indexing. 'A' means to read / write the elements in Fortran-like
			index order if a is Fortran contiguous in memory, C-like order otherwise.
		"""
		if self._modes is None:
			raise ValueError(
				'The modes have not been computed.'
				'You have to perform the fit method.'
			)

		if x is None and y is None:
			if self._snapshots_shape is None:
				raise ValueError(
					'No information about the original shape of the snapshots.'
				)

			if len(self._snapshots_shape) != 2:
				raise ValueError(
					'The dimension of the input snapshots is not 2D.'
				)

		# If domain dimensions have not been passed as argument,
		# use the snapshots dimensions
		if x is None and y is None:
			x = np.arange(self._snapshots_shape[0])
			y = np.arange(self._snapshots_shape[1])

		xgrid, ygrid = np.meshgrid(x, y)

		if index_mode is None:
			index_mode = range(self._modes.shape[1])
		elif isinstance(index_mode, int):
			index_mode = [index_mode]

		if filename:
			basename, ext = os.path.splitext(filename)

		for idx in index_mode:
			fig = plt.figure()
			fig.suptitle('DMD Mode {}'.format(idx))

			real_ax = fig.add_subplot(1, 2, 1)
			imag_ax = fig.add_subplot(1, 2, 2)

			mode = self._modes.T[idx].reshape(xgrid.shape, order=order)

			real = real_ax.pcolor(
				xgrid,
				ygrid,
				mode.real,
				cmap='jet',
				vmin=mode.real.min(),
				vmax=mode.real.max()
			)
			imag = imag_ax.pcolor(
				xgrid,
				ygrid,
				mode.imag,
				vmin=mode.imag.min(),
				vmax=mode.imag.max()
			)

			fig.colorbar(real, ax=real_ax)
			fig.colorbar(imag, ax=imag_ax)

			real_ax.set_aspect('auto')
			imag_ax.set_aspect('auto')

			real_ax.set_title('Real')
			imag_ax.set_title('Imag')

			# padding between elements
			plt.tight_layout(pad=2.)

			if filename:
				plt.savefig('{0}.{1}{2}'.format(basename, idx, ext))
				plt.close(fig)

		if not filename:
			plt.show()

	def plot_snapshots_2D(
		self, index_snap=None, filename=None, x=None, y=None, order='C'
	):
		"""
		Plot the snapshots.

		:param snapshots int or sequence of int: the index of the modes to
			plot. By default, all the modes are plotted.
		:param filename str: filename
		:param x numpy.ndarray: domain abscissa
		:param y numpy.ndarray: domain ordinate
		:param order str: {'C', 'F', 'A'}, default 'C'.
			Read the elements of snapshots using this index order, and place
			the elements into the reshaped array using this index order. It
			has to be the same used to store the snapshot. 'C' means to read/
			write the elements using C-like index order, with the last axis
			index changing fastest, back to the first axis index changing slowest.
			'F' means to read / write the elements using Fortran-like index order,
			with the first index changing fastest, and the last index changing
			slowest. Note that the 'C' and 'F' options take no account of the
			memory layout of the underlying array, and only refer to the order
			of indexing. 'A' means to read / write the elements in Fortran-like
			index order if a is Fortran contiguous in memory, C-like order otherwise.
		"""
		if self._X is None and self._Y is None:
			raise ValueError('Input snapshots not found.')

		if x is None and y is None:
			if self._snapshots_shape is None:
				raise ValueError(
					'No information about the original shape of the snapshots.'
				)

			if len(self._snapshots_shape) != 2:
				raise ValueError(
					'The dimension of the input snapshots is not 2D.'
				)

		# If domain dimensions have not been passed as argument,
		# use the snapshots dimensions
		if x is None and y is None:
			x = np.arange(self._snapshots_shape[0])
			y = np.arange(self._snapshots_shape[1])

		xgrid, ygrid = np.meshgrid(x, y)

		snapshots = np.append(self._X, self._Y[:, -1].reshape(-1, 1), axis=1)

		if index_snap is None:
			index_snap = range(self._modes.shape[1])
		elif isinstance(index_snap, int):
			index_snap = [index_snap]

		if filename:
			basename, ext = os.path.splitext(filename)

		for idx in index_snap:
			fig = plt.figure()
			fig.suptitle('Snapshot {}'.format(idx))

			snapshot = snapshots.T[idx].real.reshape(xgrid.shape, order=order)

			contour = plt.pcolor(
				xgrid,
				ygrid,
				snapshot,
				vmin=snapshot.min(),
				vmax=snapshot.max()
			)

			fig.colorbar(contour)

			if filename:
				plt.savefig('{0}.{1}{2}'.format(basename, idx, ext))
				plt.close(fig)

		if not filename:
			plt.show()
