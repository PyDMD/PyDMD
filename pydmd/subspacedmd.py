"""
Derived module from dmdbase.py for Subspace DMD.

Reference:
Takeishi, Naoya, Yoshinobu Kawahara, and Takehisa Yairi. "Subspace dynamic mode
decomposition for stochastic Koopman analysis." Physical Review E 96.3 (2017):
033310.
"""

import numpy as np

from .dmdbase import DMDBase
from .dmdoperator import DMDOperator


def reducedsvd(X, r=None):
    """
    Computes the reduced SVD of `X` taking only the first `r` modes.

    :param np.ndarray X: Matrix to be used for SVD.
    :param int r: Number of modes to be retained, if `None` the rank of `X` is
        used instead.
    :rtype: tuple
    :return: Left singular vectors, singular values and right singular vectors.
    """

    U, s, V = np.linalg.svd(X, full_matrices=True)
    if r is None:
        r = np.linalg.matrix_rank(X)
    return U[:, :r], s[:r], V.conj().T[:, :r]


class SubspaceDMDOperator(DMDOperator):
    """
    Subspace Dynamic Mode Decomposition operator class.

    :param rescale_mode: Scale Atilde as shown in
        10.1016/j.jneumeth.2015.10.010 (section 2.4) before computing its
        eigendecomposition. None means no rescaling, 'auto' means automatic
        rescaling using singular values, otherwise the scaling factors.
    :type rescale_mode: {'auto'} or None or numpy.ndarray
    :param sorted_eigs: Sort eigenvalues (and modes/dynamics accordingly) by
        magnitude if `sorted_eigs='abs'`, by real part (and then by imaginary
        part to break ties) if `sorted_eigs='real'`. Default: False.
    :type sorted_eigs: {'real', 'abs'} or False
    """

    def __init__(self, rescale_mode, sorted_eigs):
        super().__init__(
            svd_rank=-1,
            exact=True,
            forward_backward=False,
            rescale_mode=rescale_mode,
            sorted_eigs=sorted_eigs,
            tikhonov_regularization=None,
        )

        self._Atilde = None
        self._modes = None
        self._Lambda = None

    def compute_operator(self, Yp, Yf):
        """
        Compute the low-rank operator.

        Computes also modes, eigenvalues and DMD amplitudes.

        :param numpy.ndarray Yp: Matrix `Yp` as defined in the original paper.
        :param numpy.ndarray Yp: Matrix `Yf` as defined in the original paper.
        """

        n = Yp.shape[0] // 2

        _, _, Vp = reducedsvd(Yp)
        O = Yf.dot(Vp).dot(Vp.T.conj())

        Uq, _, _ = reducedsvd(O)
        r = min(np.linalg.matrix_rank(O), n)
        Uq1, Uq2 = Uq[:n, :r], Uq[n:, :r]

        U, S, V = reducedsvd(Uq1)

        self._Atilde = self._least_square_operator(U, S, V, Uq2)
        self._compute_eigenquantities()

        M = Uq2.dot(V) * np.reciprocal(S)
        self._compute_modes(M)

    def _compute_modes(self, M):
        """
        Private method that computes eigenvalues and eigenvectors of the
        high-dimensional operator (stored in self.modes and self.Lambda).

        :param numpy.ndarray M: Matrix `M` as defined in the original paper.
        """

        if self._rescale_mode is None:
            W = self.eigenvectors
        else:
            # compute W as shown in arXiv:1409.5496 (section 2.4)
            factors_sqrt = np.diag(np.power(self._rescale_mode, 0.5))
            W = factors_sqrt.dot(self.eigenvectors)

        # compute the eigenvectors of the high-dimensional operator
        high_dimensional_eigenvectors = M.dot(W) * np.reciprocal(
            self.eigenvalues
        )

        # eigenvalues are the same of lowrank
        high_dimensional_eigenvalues = self.eigenvalues

        self._modes = high_dimensional_eigenvectors
        self._Lambda = high_dimensional_eigenvalues


class SubspaceDMD(DMDBase):
    """
    Subspace Dynamic Mode Decomposition

    :param opt: argument to control the computation of DMD modes amplitudes.
        See :class:`DMDBase`. Default is False.
    :type opt: bool or int
    :param rescale_mode: Scale Atilde as shown in
            10.1016/j.jneumeth.2015.10.010 (section 2.4) before computing its
            eigendecomposition. None means no rescaling, 'auto' means automatic
            rescaling using singular values, otherwise the scaling factors.
    :type rescale_mode: {'auto'} or None or numpy.ndarray
    :param sorted_eigs: Sort eigenvalues (and modes/dynamics accordingly) by
        magnitude if `sorted_eigs='abs'`, by real part (and then by imaginary
        part to break ties) if `sorted_eigs='real'`. Default: False.
    :type sorted_eigs: {'real', 'abs'} or False
    """

    def __init__(
        self,
        opt=False,
        rescale_mode=None,
        sorted_eigs=False,
    ):
        self._tlsq_rank = 0
        self._original_time = None
        self._dmd_time = None
        self._opt = opt

        self._b = None
        self._snapshots = None
        self._snapshots_shape = None

        self._modes_activation_bitmask_proxy = None

        self._Atilde = SubspaceDMDOperator(
            rescale_mode=rescale_mode,
            sorted_eigs=sorted_eigs,
        )

    def fit(self, X):
        """
        Compute the SubspaceDynamic Modes Decomposition to the input data.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        """
        self._snapshots, self._snapshots_shape = self._col_major_2darray(X)

        n_samples = self._snapshots.shape[1]
        Y0 = self._snapshots[:, :-3]
        Y1 = self._snapshots[:, 1:-2]
        Y2 = self._snapshots[:, 2:-1]
        Y3 = self._snapshots[:, 3:]

        Yp = np.vstack((Y0, Y1))
        Yf = np.vstack((Y2, Y3))

        self.operator.compute_operator(Yp, Yf)

        # Default timesteps
        self._set_initial_time_dictionary(
            {"t0": 0, "tend": n_samples - 1, "dt": 1}
        )

        self._b = self._compute_amplitudes()

        return self
