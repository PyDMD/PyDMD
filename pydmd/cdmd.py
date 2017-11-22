"""
Derived module from dmdbase.py for compressive dmd.
"""
from __future__ import division
from past.utils import old_div
import numpy as np
import scipy.sparse

from .dmdbase import DMDBase


class CDMD(DMDBase):
	"""
	Compressive Dynamic Mode Decomposition

	:param numpy.ndarray X: the input matrix with dimension `m`x`n`
	:param int svd_rank: rank truncation in SVD. Default is 0, that means no
		truncation.
	:param int tlsq_rank: rank truncation computing Total Least Square. Default
		is 0, that means no truncation.
	:param bool exact: flag to compute either exact DMD or projected DMD.
		Default is False.
	:param str or callable: the method for compress the input data.
	"""

	def __init__(self, svd_rank=0, tlsq_rank=0, compress_method='uniform'):
		super(CDMD, self).__init__(svd_rank, tlsq_rank)
		self.compress_method = compress_method

	def _compress_snapshots(self, snapshots, C):
		"""
		Private method that compresses the input snapshots by pre-multiply the
		input matrix by `C`.

		:param snapshots numpy.array: the matrix that contains the snapshots,
			stored by column.
		:param C str or numpy.ndarray: the matrix that pre-multiplies the
			snapshots matrix in order to compress it; valid values are:
			- 'normal': the matrix C with dimension (`nsnaps`, `ndim`) is
			  randomly generated with normal distribution with mean equal to
			  0.0 and standard deviation equal to 1.0;
			- 'uniform': the matrix C with dimension (`nsnaps`, `ndim`) is
			  randomly generated with uniform distribution between 0 and 1;
			- 'sparse': the matrix C with dimension (`nsnaps`, `ndim`) is
			  random sparse matrix;
			- 'sample': the matrix C with dimension (`nsnaps`, `ndim`) where
			  each row contains an element equal to 1 and all the others
			  element are null.

			If `C` is a numpy.array, its dimension must be (`nsnaps`, `ndim`).
	
		"""

		def swap(tup):
			a, b = tup
			return b, a

		sample_matrix = np.zeros((swap(snapshots.shape)))
		sample_matrix[np.arange(snapshots.shape[1]),
					  np.random.choice(*snapshots.shape, replace=False)] = 1.

		available_methods = {
			'uniform': np.random.uniform(0, 1, size=(swap(snapshots.shape))),
			'sparse': scipy.sparse.random(*swap(snapshots.shape), density=1.),
			'normal': np.random.normal(0, 1, size=(swap(snapshots.shape))),
			'sample': sample_matrix,
		}

		if isinstance(C, str):
			C = available_methods[C]

		# compress the matrix
		Y = C.dot(snapshots)

		return Y, C

	def fit(self, X):
		"""
		Compute the Dynamic Modes Decomposition to the input data.

		:param iterable or numpy.ndarray X: the input snapshots.
		"""
		self._snapshots, self._snapshots_shape = self._col_major_2darray(X)

		compress_snpshots, C = self._compress_snapshots(
			self._snapshots, self.compress_method
		)

		n_samples = self._snapshots.shape[1]
		X = self._snapshots[:, :-1]
		Y = self._snapshots[:, 1:]

		X, Y = self._compute_tlsq(X, Y, self.tlsq_rank)

		U, s, V = self._compute_svd(X, self.svd_rank)

		self._Atilde = self._build_lowrank_op(U, s, V, Y)

		# No projected modes for cdmd
		self._eigs, self._modes = self._eig_from_lowrank_op(
			self._Atilde, Y, U, s, V, True
		)

		self._b = self._compute_amplitudes(self._modes, self._snapshots)

		# Default timesteps
		self.original_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}
		self.dmd_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}

		return self
