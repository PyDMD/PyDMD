import numpy as np
import matplotlib.pyplot as plt


class DMDBase(object):
	"""
	Dynamic Mode Decomposition

	This method decomposes

	:param numpy.ndarray X: the input matrix with dimension `m`x`n`
	:param int k: rank truncation in SVD
	"""

	def __init__(self, svd_rank=0, tlsq_rank=0, exact=False):
		self.svd_rank = svd_rank
		self.tlsq_rank = tlsq_rank
		self.exact = exact

		self._basis = None	# spatial basis vectors
		self._mode_coeffs = None
		self._eigs = None
		self._Atilde = None
		self._modes = None	# Phi
		self._amplitudes = None	 # B
		self._vander = None	 # Vander

	@property
	def modes(self):
		return self._modes

	@property
	def eigs(self):
		return self._eigs

	@property
	def amplitudes(self):
		return self._amplitudes

	@property
	def vander(self):
		return self._vander

	@property
	def reconstructed_data(self):
		return self._modes.dot(self._amplitudes).dot(self._vander)

	def fit(self, *args):
		"""
		Abstract method to fit the snapshots matrices.

		Not implemented, it has to be implemented in subclasses.
		"""
		raise NotImplementedError(
			'Subclass must implement abstract method {}.fit'.format(
		 		self.__class__.__name__
			)
		)
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
		Singular Value Decomposition truncated

		:param X numpy.ndarray: the matrix to decompose;
		:param svd_rank int: the rank for the truncation;
		"""
		U, s, V = np.linalg.svd(X, full_matrices=False)
		V = V.conj().T

		if svd_rank is 0:
			return U, s, V

		rank = min(svd_rank, U.shape[1])

		U = U[:, :rank]
		V = V[:, :rank]
		s = s[:rank]

		return U, s, V


	def plot_eigs(self, show_axes=True, show_unit_circle=True):
		"""
		"""
		if self._eigs is None:
			raise ValueError(
				'The eigenvalues have not been computed.' 
				'You have to perform the fit method.'
			)

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
			unit_circle = plt.Circle((0., 0.),
									 1.,
									 color='green',
									 fill=False,
									 label='Unit circle',
									 linestyle='--')
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
				plt.legend([points, unit_circle],
						   ['Eigenvalues', 'Unit circle'],
						   loc=1)
			)
		else:
			ax.add_artist(plt.legend([points], ['Eigenvalues'], loc=1))

		plt.show()
