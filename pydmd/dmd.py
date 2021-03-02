"""
Derived module from dmdbase.py for classic dmd.
"""

# --> Import standard python packages
import numpy as np
from scipy.linalg import pinv2

# --> Import PyDMD base class for DMD.
from .dmdbase import DMDBase


def pinv(x): return pinv2(x, rcond=10 * np.finfo(float).eps)


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
    :param bool opt: flag to compute optimal amplitudes. See :class:`DMDBase`.
        Default is False.
    :param rescale_mode: Scale Atilde as shown in
            10.1016/j.jneumeth.2015.10.010 (section 2.4) before computing its
            eigendecomposition. None means no rescaling, 'auto' means automatic
            rescaling using singular values, otherwise the scaling factors.
    :type rescale_mode: {'auto'} or None or numpy.ndarray
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

        self._svd_modes = U

        self._eigs, self._modes = self._eig_from_lowrank_op(
            self._Atilde, Y, U, s, V, self.exact)

        # Default timesteps
        self.original_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}
        self.dmd_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}

        self._b = self._compute_amplitudes(self._modes, self._snapshots,
                                           self._eigs, self.opt)

        return self

    def predict(self, X):
        """Predict the output Y given the input X using the fitted DMD model.

        Parameters
        ----------
        X : numpy array
            Input data.

        Returns
        -------
        Y : numpy array
            Predicted output.

        """

        # --> Predict using the SVD modes as the basis.
        if self.exact is False:
            Y = np.linalg.multi_dot(
                [self._svd_modes, self._Atilde, self._svd_modes.T.conj(), X]
            )
        # --> Predict using the DMD modes as the basis.
        elif self.exact is True:
            adjoint_modes = pinv(self._modes)
            Y = np.linalg.multi_dot(
                [self._modes, np.diag(self._eigs), adjoint_modes, X]
            )

        return Y
