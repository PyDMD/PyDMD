"""
Derived module from dmdbase.py for classic dmd.
"""

import numpy as np

from .dmdbase import DMDBase
from .linalg import assert_same_linalg_type, build_linalg_module
from .snapshots import Snapshots
from .utils import compute_tlsq


class DMD(DMDBase):
    """
    Dynamic Mode Decomposition

    :param svd_rank: the rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive interger, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation.
    :type svd_rank: int or float
    :param int tlsq_rank: rank truncation computing Total Least Square. Default
        is 0, that means TLSQ is not applied.
    :param bool exact: flag to compute either exact DMD or projected DMD.
        Default is False.
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
        part to break ties) if `sorted_eigs='real'`. Default: False.
    :type sorted_eigs: {'real', 'abs'} or False
    :param tikhonov_regularization: Tikhonov parameter for the regularization.
        If `None`, no regularization is applied, if `float`, it is used as the
        :math:`\lambda` tikhonov parameter.
    :type tikhonov_regularization: int or float
    """

    def fit(self, X):
        """
        Compute the Dynamic Modes Decomposition to the input data.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        """
        self._reset()

        self._snapshots_holder = Snapshots(X)

        n_samples = self.snapshots.shape[-1]
        X = self.snapshots[..., :-1]
        Y = self.snapshots[..., 1:]

        X, Y = compute_tlsq(X, Y, self._tlsq_rank)
        self._svd_modes, _, _ = self.operator.compute_operator(X, Y)

        # Default timesteps
        self._set_initial_time_dictionary(
            {"t0": 0, "tend": n_samples - 1, "dt": 1}
        )

        self._b = self._compute_amplitudes()

        return self

    def predict(self, X):
        """
        Predict the output Y given the input X using the fitted DMD model.

        :param numpy.ndarray X: the input vector.
        :return: one time-step ahead predicted output.
        :rtype: numpy.ndarray
        """
        assert_same_linalg_type(X, self.modes)
        
        linalg_module = build_linalg_module(X)
        return linalg_module.multi_dot(
            (self.modes, linalg_module.diag_matrix(self.eigs),
                linalg_module.pinv(self.modes), X)
        )
