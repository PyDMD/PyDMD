"""
Derived module from dmdbase.py for Subspace DMD.

Reference:
Takeishi, Naoya, Yoshinobu Kawahara, and Takehisa Yairi. "Subspace dynamic mode
decomposition for stochastic Koopman analysis." Physical Review E 96.3 (2017):
033310.
"""

from .dmdbase import DMDBase
from .dmdoperator import DMDOperator
from .utils import compute_svd
from pydmd.linalg import build_linalg_module, is_array
from .snapshots import Snapshots


class SubspaceDMDOperator(DMDOperator):
    """
    Subspace Dynamic Mode Decomposition operator class.

    :param svd_rank: the rank for the truncation; if -1 all the columns of
        :math:`U_q` are used, if `svd_rank` is an integer grater than zero
        it is used as the number of columns retained from  :math:`U_q`.
        `svd_rank=0` or float values are not supported.
    :type svd_rank: int
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

    def __init__(self, svd_rank, rescale_mode, sorted_eigs):
        super().__init__(
            svd_rank=svd_rank,
            exact=True,
            forward_backward=False,
            rescale_mode=rescale_mode,
            sorted_eigs=sorted_eigs,
            tikhonov_regularization=None,
        )

        if svd_rank is None:
            self._svd_rank = 0
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
        n = Yp.shape[-2] // 2

        _, _, Vp = compute_svd(Yp, svd_rank=self._svd_rank)
        linalg_module = build_linalg_module(Yp)

        O = linalg_module.multi_dot((Yf, Vp, Vp.swapaxes(-1, -2).conj()))

        Uq, _, _ = compute_svd(O, svd_rank=self._svd_rank)

        r = min(Uq.shape[-1], n, abs(self._svd_rank))
        Uq1 = Uq[..., :n, :r]
        Uq2 = Uq[..., n:, :r]

        U, S, V = compute_svd(Uq1, svd_rank=self._svd_rank)

        self._Atilde = self._least_square_operator(U, S, V, Uq2)
        self._compute_eigenquantities()

        M = linalg_module.dot(Uq2, V) / (S[:, None] if Yp.ndim == 3 else S)
        self._compute_modes(M)

    def _compute_modes(self, M):
        """
        Private method that computes eigenvalues and eigenvectors of the
        high-dimensional operator (stored in self.modes and self.Lambda).

        :param numpy.ndarray M: Matrix `M` as defined in the original paper.
        """
        linalg_module = build_linalg_module(M)

        if self._rescale_mode is None:
            W = self.eigenvectors
        elif is_array(self._rescale_mode):
            # compute W as shown in arXiv:1409.5496 (section 2.4)
            if len(self._rescale_mode) != self.as_array.shape[0]:
                raise ValueError("Scaling by an invalid number of coefficients")
            scaling_factors = linalg_module.to(
                self.as_array, self._rescale_mode
            )
            factors_sqrt = linalg_module.diag_matrix(
                linalg_module.pow(scaling_factors, 0.5)
            )
            W = linalg_module.dot(factors_sqrt, self.eigenvectors)
        else:
            raise ValueError(
                f"Invalid value for rescale_mode: {self._rescale_mode}."
            )

        # compute the eigenvectors of the high-dimensional operator
        high_dimensional_eigenvectors = linalg_module.dot(M, W) / (
            self.eigenvalues[:, None] if M.ndim == 3 else self.eigenvalues
        )

        # eigenvalues are the same of lowrank
        high_dimensional_eigenvalues = self.eigenvalues

        self._modes = high_dimensional_eigenvectors
        self._Lambda = high_dimensional_eigenvalues


class SubspaceDMD(DMDBase):
    """
    Subspace Dynamic Mode Decomposition

     :param svd_rank: the rank for the truncation; if -1 all the columns of
        :math:`U_q` are used, if `svd_rank` is an integer grater than zero
        it is used as the number of columns retained from  :math:`U_q`.
        `svd_rank=0` or float values are not supported.
    :type svd_rank: int
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
        svd_rank=-1,
        opt=False,
        rescale_mode=None,
        sorted_eigs=False,
    ):
        self._tlsq_rank = 0
        self._original_time = None
        self._dmd_time = None
        self._opt = opt

        self._b = None
        self._snapshots_holder = None

        self._modes_activation_bitmask_proxy = None

        self._Atilde = SubspaceDMDOperator(
            svd_rank=svd_rank,
            rescale_mode=rescale_mode,
            sorted_eigs=sorted_eigs,
        )

    def fit(self, X, batch=False):
        """
        Compute the Subspace Dynamic Modes Decomposition to the input data.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        :param batch: If `True`, the first dimension is dedicated to batching.
        :type batch: bool
        """
        self._reset()

        self._snapshots_holder = Snapshots(X, batch=batch)

        n_samples = self.snapshots.shape[-1]
        Y0 = self.snapshots[..., :-3]
        Y1 = self.snapshots[..., 1:-2]
        Y2 = self.snapshots[..., 2:-1]
        Y3 = self.snapshots[..., 3:]

        linalg_module = build_linalg_module(X)
        Yp = linalg_module.cat((Y0, Y1), axis=-2)
        Yf = linalg_module.cat((Y2, Y3), axis=-2)

        self.operator.compute_operator(Yp, Yf)

        # Default timesteps
        self._set_initial_time_dictionary(
            {"t0": 0, "tend": n_samples - 1, "dt": 1}
        )

        self._b = self._compute_amplitudes()

        return self
