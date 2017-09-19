"""
Derived module from dmdbase.py for forward/backward dmd.
"""
import numpy as np
from scipy.linalg import sqrtm
from dmdbase import DMDBase


class FbDMD(DMDBase):
	"""
	Forward/backward DMD class.
	
	:param numpy.ndarray X: the input matrix with dimension `m`x`n`
	:param int svd_rank: rank truncation in SVD. Default is 0, that means no
		truncation.
	:param int tlsq_rank: rank truncation computing Total Least Square. Default
		is 0, that means no truncation.
	:param bool exact: flag to compute either exact DMD or projected DMD.
		Default is False.

	Reference: Dawson et al. https://arxiv.org/abs/1507.02264
	"""

	def fit(self, X, Y=None):
		"""
		Compute the Dynamics Modes Decomposition to the input data.

		:param iterable or numpy.ndarray X: the input snapshots.
		:param itarable or numpy.ndarray Y: if specified, it provides the
			snapshots at the next time step. Its dimension must be equal to X.
			Default is None.
		"""
		self._fit_read_input(X, Y)
		n_samples = X.shape[1]
		
		X, Y = self._compute_tlsq(self._X, self._Y, self.tlsq_rank)

		# Singular Value Decomposition - Backward
		U, s, V = self._compute_svd(Y, self.svd_rank)

		# DMD Modes - Backward
		Sinverse = np.diag(1. / s)
		bAtilde = U.T.conj().dot(X).dot(V).dot(Sinverse)

		# Singular Value Decomposition - Forward
		U, s, V = self._compute_svd(X, self.svd_rank)

		# DMD Modes - Forward
		Sinverse = np.diag(1. / s)
		fAtilde = U.T.conj().dot(Y).dot(V).dot(Sinverse)

		# A tilde
		self._Atilde = sqrtm(fAtilde.dot(np.linalg.inv(bAtilde)))

		basis = Y.dot(V).dot(Sinverse) if self.exact else U

		self._eigs, mode_coeffs = np.linalg.eig(self._Atilde)
		self._modes = basis.dot(mode_coeffs)

		#-----------------------------------------------------------------------
		# DMD Amplitudes and Dynamics
		#-----------------------------------------------------------------------
		b = np.linalg.lstsq(self._modes, X[:, 0])[0]

		vander = np.fliplr(np.vander(self._eigs, N=n_samples))
		self._dynamics = (vander.T * b).T

		return self
