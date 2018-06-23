"""
Derived module from dmdbase.py for compressed dmd.
As a reference consult this work by Erichson, Brunton and Kutz:
https://doi.org/10.1007/s11554-016-0655-2
"""
from __future__ import division
import numpy as np
import scipy.sparse

from .dmdbase import DMDBase


class CDMD(DMDBase):
    """
    Compressed Dynamic Mode Decomposition.

    Compute the dynamic mode decomposition after the multiplication of the
    snapshots matrix by the `compression_matrix`, in order to compress the
    input data. It is possible use a custom matrix for the compression or chose
    between the preconstructed matrices. Available values for
    `compression_matrix` are:

    - 'normal': the matrix C with dimension (`nsnaps`, `ndim`) is randomly
      generated with normal distribution with mean equal to 0.0 and standard
      deviation equal to 1.0;
    - 'uniform': the matrix C with dimension (`nsnaps`, `ndim`) is
      randomly generated with uniform distribution between 0 and 1;
    - 'sparse': the matrix C with dimension (`nsnaps`, `ndim`) is a
      random sparse matrix;
    - 'sample': the matrix C with dimension (`nsnaps`, `ndim`) where
      each row contains an element equal to 1 and all the other
      elements are null.

    :param svd_rank: the rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive interger, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation.
    :type svd_rank: int or float
    :param int tlsq_rank: rank truncation computing Total Least Square. Default
        is 0, that means TLSQ is not applied.
    :param compression_matrix: the matrix that pre-multiplies the snapshots
        matrix in order to compress it; if `compression_matrix` is a
        numpy.ndarray, its dimension must be (`nsnaps`, `ndim`). Default value
        is '`uniform`'.
    :type compression_matrix: {'linear', 'sparse', 'uniform', 'sample'} or
        numpy.ndarray
    :param bool opt: flag to compute optimized DMD. Default is False.
    """

    def __init__(self,
                 svd_rank=0,
                 tlsq_rank=0,
                 compression_matrix='uniform',
                 opt=False):
        super(CDMD, self).__init__(svd_rank, tlsq_rank, True, opt)
        self.compression_matrix = compression_matrix

    def _compress_snapshots(self):
        """
        Private method that compresses the snapshots matrix by pre-multiplying
        it by the chosen `compression_matrix`.

        :return: the compressed snapshots.
        :rtype: numpy.ndarray
        """

        C_shape = (self._snapshots.shape[1], self._snapshots.shape[0])
        if isinstance(self.compression_matrix, np.ndarray):
            C = self.compression_matrix
        elif self.compression_matrix == 'uniform':
            C = np.random.uniform(0, 1, size=(C_shape))
        elif self.compression_matrix == 'sparse':
            C = scipy.sparse.random(*C_shape, density=1.)
        elif self.compression_matrix == 'normal':
            C = np.random.normal(0, 1, size=(C_shape))
        elif self.compression_matrix == 'sample':
            C = np.zeros(C_shape)
            C[np.arange(self._snapshots.shape[1]),
              np.random.choice(*self._snapshots.shape, replace=False)] = 1.

        # compress the matrix
        Y = C.dot(self._snapshots)

        return Y

    def fit(self, X):
        """
        Compute the Dynamic Modes Decomposition to the input data.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        """
        self._snapshots, self._snapshots_shape = self._col_major_2darray(X)

        compressed_snapshots = self._compress_snapshots()

        n_samples = compressed_snapshots.shape[1]
        X = compressed_snapshots[:, :-1]
        Y = compressed_snapshots[:, 1:]

        X, Y = self._compute_tlsq(X, Y, self.tlsq_rank)

        U, s, V = self._compute_svd(X, self.svd_rank)

        self._Atilde = self._build_lowrank_op(U, s, V, Y)

        # No projected modes for cdmd
        self._eigs, self._modes = self._eig_from_lowrank_op(
            self._Atilde, self._snapshots[:, 1:], U, s, V, True)

        self._b = self._compute_amplitudes(self._modes, self._snapshots,
                                           self._eigs, self.opt)

        # Default timesteps
        self.original_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}
        self.dmd_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}

        return self
