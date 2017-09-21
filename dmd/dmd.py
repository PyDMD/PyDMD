"""
Derived module from dmdbase.py for classic dmd.
"""
import numpy as np

from .dmdbase import DMDBase


class DMD(DMDBase):
	"""
	Dynamic Mode Decomposition

	:param numpy.ndarray X: the input matrix with dimension `m`x`n`
	:param int svd_rank: rank truncation in SVD. Default is 0, that means no
		truncation.
	:param int tlsq_rank: rank truncation computing Total Least Square. Default
		is 0, that means no truncation.
	:param bool exact: flag to compute either exact DMD or projected DMD.
		Default is False.
	"""

	def fit(self, X, Y=None):
		"""
		Compute the Dynamic Modes Decomposition to the input data.

		:param iterable or numpy.ndarray X: the input snapshots.
		:param itarable or numpy.ndarray Y: if specified, it provides the
			snapshots at the next time step. Its dimension must be equal to X.
			Default is None.
		"""
		self._fit_read_input(X, Y)
		n_samples = self._X.shape[1] + 1

		X, Y = self._compute_tlsq(self._X, self._Y, self.tlsq_rank)

		U, s, V = self._compute_svd(X, self.svd_rank)

		#-----------------------------------------------------------------------
		# DMD Modes
		#-----------------------------------------------------------------------
		Sinverse = np.diag(1. / s)
		self._Atilde = U.T.conj().dot(Y).dot(V).dot(Sinverse)

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
