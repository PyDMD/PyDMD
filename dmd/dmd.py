import numpy as np
import matplotlib.pyplot as plt

from dmdbase import DMDBase


class DMD(DMDBase):
	"""
	Dynamic Mode Decomposition

	This method decomposes

	:param numpy.ndarray X: the input matrix with dimension `m`x`n`
	:param int k: rank truncation in SVD
	"""

	def fit(self, X, Y=None):
		"""
		"""
		n_samples = X.shape[1]
		# split the data
		if Y is None:
			Y = X[:, 1:]
			X = X[:, :-1]

		X, Y = self._compute_tlsq(X, Y, self.tlsq_rank)

		U, s, V = self._compute_svd(X, self.svd_rank)

		#-----------------------------------------------------------------------
		# DMD Modes
		#-----------------------------------------------------------------------
		Sinverse = np.diag(1. / s)
		self._Atilde = np.transpose(U).dot(Y).dot(V).dot(Sinverse)

		if self.exact:
			# exact DMD
			self._basis = Y.dot(V).dot(Sinverse)
		else:
			# projected DMD
			self._basis = U

		self._eigs, self._mode_coeffs = np.linalg.eig(self._Atilde)

		self._modes = self._basis.dot(self._mode_coeffs)

		#-----------------------------------------------------------------------
		# DMD Amplitudes and Dynamics
		#-----------------------------------------------------------------------
		b = np.linalg.lstsq(self._modes, X[:, 0])[0]
		self._amplitudes = np.diag(b)

		self._vander = np.fliplr(np.vander(self._eigs, N=n_samples))

		return self
