"""
Derived module from dmdbase.py for classic dmd.
"""
from .dmdbase import DMDBase


class DMD(DMDBase):
	"""
	Dynamic Mode Decomposition

	:param int svd_rank: rank truncation in SVD. If 0, the method computes the
		optimal rank and uses it for truncation; if positive number, the method
		uses the argument for the truncation; if -1, the method does not
		compute truncation.
	:param int tlsq_rank: rank truncation computing Total Least Square. Default
		is 0, that means no truncation.
	:param bool exact: flag to compute either exact DMD or projected DMD.
		Default is False.
	"""

	def fit(self, X):
		"""
		Compute the Dynamic Modes Decomposition to the input data.

		:param X: the input snapshots.
		:type X: numpy.ndarray or iterable
		"""
		self._snapshots, self._snapshots_shape = self._col_major_2darray(X)

		n_samples = self._snapshots.shape[1]
		X = self._snapshots[:, :-1]
		Y = self._snapshots[:, 1:]

		X, Y = self._compute_tlsq(X, Y, self.tlsq_rank)

		U, s, V = self._compute_svd(X, self.svd_rank)

		self._Atilde = self._build_lowrank_op(U, s, V, Y)

		self._eigs, self._modes = self._eig_from_lowrank_op(
			self._Atilde, Y, U, s, V, self.exact
		)

		self._b = self._compute_amplitudes(self._modes, self._snapshots)

		# Default timesteps
		self.original_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}
		self.dmd_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}

		return self
