import numpy as np


class DMD(object):
	"""
	Dynamic Mode Decomposition

	This method decomposes

	:param numpy.ndarray X: the input matrix with dimension `m`x`n`
	:param int k: rank truncation in SVD
	"""
	def __init__(self, k=None):
		self.k = k
		self._basis = None  # spatial basis vectors
		self._mode_coeffs = None  # DMD mode coefficients
		self._eigs = None # DMD eigenvalues
		self._Atilde = None  # The full DMD matrix
		self._modes = None # Phi
		self._amplitudes = None # B
		self._vander = None # Vander

	@property
	def modes(self):
		return self._modes

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

		self._vander = np.fliplr(np.vander(self._eigs, N=X.shape[1]))

		return self
