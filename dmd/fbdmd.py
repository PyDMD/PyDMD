"""
Derived module from dmd.py for forward/backward dmd.
"""
import numpy as np
from scipy.linalg import sqrtm
from dmdbase import DMDBase


class FbDMD(DMDBase):
	"""
	Reference: Dawson et al. https://arxiv.org/abs/1507.02264
	"""

	def fit(self, X, Y=None):
		"""
		"""
		n_samples = X.shape[1]
		# split the data
		if Y is None:
			Y = X[:, 1:]
			X = X[:, :-1]

		# Singular Value Decomposition - Backward
		U, s, V = self._compute_svd(Y, self.svd_rank)

		# DMD Modes - Backward
		Sinverse = np.diag(1. / s)
		bAtilde = np.transpose(U).dot(X).dot(V).dot(Sinverse)

		# Singular Value Decomposition - Forward
		U, s, V = self._compute_svd(X, self.svd_rank)

		# DMD Modes - Forward
		Sinverse = np.diag(1. / s)
		fAtilde = np.transpose(U).dot(Y).dot(V).dot(Sinverse)

		# A tilde
		self._Atilde = sqrtm(fAtilde.dot(np.linalg.inv(bAtilde)))

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
