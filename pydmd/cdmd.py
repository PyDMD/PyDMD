"""
Derived module from dmdbase.py for compressed dmd.
As a reference consult this work by Erichson, Brunton and Kutz:
https://doi.org/10.1007/s11554-016-0655-2
"""
from __future__ import division
import numpy as np
import scipy.sparse
from scipy.linalg import sqrtm

from .dmdbase import DMDBase, DMDTimeDict
from .dmdoperator import DMDOperator

from .utils import compute_tlsq, compute_svd


class CDMDOperator(DMDOperator):
    """
    DMD operator for Compressed-DMD.

    :param svd_rank: the rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive interger, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation.
    :type svd_rank: int or float
    :param rescale_mode: Scale Atilde as shown in
        10.1016/j.jneumeth.2015.10.010 (section 2.4) before computing its
        eigendecomposition. None means no rescaling, 'auto' means automatic
        rescaling using singular values, otherwise the scaling factors.
    :type rescale_mode: {'auto'} or None or numpy.ndarray
    :param bool forward_backward: If True, the low-rank operator is computed
        like in fbDMD (reference: https://arxiv.org/abs/1507.02264). Default is
        False.
    :param sorted_eigs: Sort eigenvalues (and modes/dynamics accordingly) by
        magnitude if `sorted_eigs='abs'`, by real part (and then by imaginary
        part to break ties) if `sorted_eigs='real'`. Default: False.
    :type sorted_eigs: {'real', 'abs'} or False
    :param tikhonov: tikhonov parameter for regularization
        If 0, no regularization is applied, if float, it is used as the lambda tikhonov 
        parameter
    :type tikhonov: int or float
    """

    def __init__(self, svd_rank, rescale_mode, forward_backward, sorted_eigs, tikhonov):
        super().__init__(svd_rank=svd_rank, exact=True,
                         rescale_mode=rescale_mode,
                         forward_backward=forward_backward,
                         sorted_eigs=sorted_eigs,
                         tikhonov=tikhonov)
        self._Atilde = None

    def compute_operator(self, compressedX, compressedY, nonCompressedY):
        """
        Compute the low-rank operator.

        :param numpy.ndarray compressedX: the compressed version of the matrix
            containing the snapshots x0,..x{n-1} by column.
        :param numpy.ndarray compressedY: the compressed version of the matrix
            containing the snapshots x1,..x{n} by column.
        :param numpy.ndarray nonCompressedY: the matrix containing the
            snapshots x1,..x{n} by column.
        :return: the (truncated) left-singular vectors matrix, the (truncated)
            singular values array, the (truncated) right-singular vectors
            matrix of compressedX.
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
        """

        U, s, V = compute_svd(compressedX, svd_rank=self._svd_rank)

        atilde = self._least_square_operator(U, s, V, compressedY)

        if self._forward_backward:
            # b stands for "backward"
            bU, bs, bV = compute_svd(compressedY, svd_rank=self._svd_rank)
            atilde_back = self._least_square_operator(bU, bs, bV, compressedX)
            atilde = sqrtm(atilde.dot(np.linalg.inv(atilde_back)))

        self._Atilde = atilde
        self._compute_eigenquantities()
        self._compute_modes(nonCompressedY, U, s, V)

        return U, s, V


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
    :param opt: argument to control the computation of DMD modes amplitudes.
        See :class:`DMDBase`. Default is False.
    :type opt: bool or int
    :param rescale_mode: Scale Atilde as shown in
            10.1016/j.jneumeth.2015.10.010 (section 2.4) before computing its
            eigendecomposition. None means no rescaling, 'auto' means automatic
            rescaling using singular values, otherwise the scaling factors.
    :type rescale_mode: {'auto'} or None or numpy.ndarray
    :param bool forward_backward: If True, the low-rank operator is computed
        like in fbDMD (reference: https://arxiv.org/abs/1507.02264). Default is
        False.
    :param sorted_eigs: Sort eigenvalues (and modes/dynamics accordingly) by
        magnitude if `sorted_eigs='abs'`, by real part (and then by imaginary
        part to break ties) if `sorted_eigs='real'`.
    :type sorted_eigs: {'real', 'abs'} or False
    :param sorted_eigs: Sort eigenvalues (and modes/dynamics accordingly) by
        magnitude if `sorted_eigs='abs'`, by real part (and then by imaginary
        part to break ties) if `sorted_eigs='real'`. Default: False.
    :type sorted_eigs: {'real', 'abs'} or False
    """

    def __init__(self, svd_rank=0, tlsq_rank=0, compression_matrix='uniform',
                 opt=False, rescale_mode=None, forward_backward=False,
                 sorted_eigs=False, tikhonov=0):

        self._tlsq_rank = tlsq_rank
        self._opt = opt
        self._compression_matrix = compression_matrix

        self._Atilde = CDMDOperator(svd_rank=svd_rank,
                                    rescale_mode=rescale_mode,
                                    forward_backward=forward_backward,
                                    sorted_eigs=sorted_eigs,
                                    tikhonov=tikhonov)

    @property
    def compression_matrix(self):
        """The compression matrix"""
        return self._compression_matrix

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

        X, Y = compute_tlsq(X, Y, self.tlsq_rank)
        self.operator.compute_operator(X, Y, self._snapshots[:, 1:])

        # Default timesteps
        self.original_time = DMDTimeDict(
            {'t0': 0, 'tend': n_samples - 1, 'dt': 1})
        self.dmd_time = DMDTimeDict({'t0': 0, 'tend': n_samples - 1, 'dt': 1})

        self._b = self._compute_amplitudes()

        return self
