import numpy as np
import matplotlib.pyplot as plt


class DMD(object):
	"""
	Dynamic Mode Decomposition

	This method decomposes

	:param numpy.ndarray X: the input matrix with dimension `m`x`n`
	:param int k: rank truncation in SVD
	"""
	def __init__(self, k=None):
		self.k = k
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

	def fit(self, X, Y=None):
		"""
		"""
		n_samples = X.shape[1]
		# split the data
		if Y is None:
			Y = X[:, 1:]
			X = X[:, :-1]

		#---------------------------------------------------------------------------
		# Singular Value Decomposition
		#---------------------------------------------------------------------------
		U, s, V = np.linalg.svd(X, full_matrices=False)
		V = np.conjugate(V.T)

		if self.k is not None:
			U = U[:, 0:self.k]
			V = V[:, 0:self.k]
			s = s[0:self.k]

		Sinverse = np.diag(1. / s)

		#---------------------------------------------------------------------------
		# DMD Modes
		#---------------------------------------------------------------------------
		self._Atilde = np.transpose(U).dot(Y).dot(V).dot(Sinverse)
		# exact DMD
		self._basis = Y.dot(V).dot(Sinverse)
		self._eigs, self._mode_coeffs = np.linalg.eig(self._Atilde)

		self._modes = self._basis.dot(self._mode_coeffs)

		#---------------------------------------------------------------------------
		# DMD Amplitudes and Dynamics
		#---------------------------------------------------------------------------
		b = np.linalg.lstsq(self._modes, X[:, 0])[0]
		self._amplitudes = np.diag(b)

		self._vander = np.fliplr(np.vander(self._eigs, N=n_samples))

		return self

	def plot_eigs(self, show_axes=False, show_unit_circle=False):
		"""
		"""
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
