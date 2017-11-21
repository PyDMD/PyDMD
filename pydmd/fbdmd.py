"""
Derived module from dmdbase.py for forward/backward dmd.
"""
import numpy as np

from scipy.linalg import sqrtm
from .dmdbase import DMDBase


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

	def fit(self, X):
		"""
		Compute the Dynamics Modes Decomposition to the input data.

		:param iterable or numpy.ndarray X: the input snapshots.
		:param itarable or numpy.ndarray Y: if specified, it provides the
			snapshots at the next time step. Its dimension must be equal to X.
			Default is None.
		"""
		self._snapshots, self._snapshots_shape = self._col_major_2darray(X)

		n_samples = self._snapshots.shape[1]
		X = self._snapshots[:, :-1]
		Y = self._snapshots[:, 1:]

		X, Y = self._compute_tlsq(X, Y, self.tlsq_rank)

		Uy, sy, Vy = self._compute_svd(Y, self.svd_rank)
		Ux, sx, Vx = self._compute_svd(X, self.svd_rank)

		if len(sy) != len(sx):
			raise ValueError(
				'Different number of singular value;'
				'please consider to specify the svd_rank'
			)

		bAtilde = self._build_lowrank_op(Uy, sy, Vy, X)
		fAtilde = self._build_lowrank_op(Ux, sx, Vx, Y)

		self._Atilde = sqrtm(fAtilde.dot(np.linalg.inv(bAtilde)))

		self._eigs, self._modes = self._eig_from_lowrank_op(
			self._Atilde, Y, Ux, sx, Vx, self.exact
		)

		self._b = self._compute_amplitudes(self._modes, self._snapshots)

		self.original_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}
		self.dmd_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}

		return self
