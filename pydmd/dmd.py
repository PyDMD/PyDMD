"""
Derived module from dmdbase.py for classic dmd.
"""

# --> Import standard python packages
import numpy as np

# --> Import PyDMD base class for DMD.
from .dmdbase import DMDBase

from .dmdoperator import DMDOperator
from .utils import compute_tlsq

from scipy.linalg import pinv2

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
    :param opt: argument to control the computation of DMD modes amplitudes. See
        :class:`DMDBase`. Default is False.
    :type opt: bool or int
    :param rescale_mode: Scale Atilde as shown in
            10.1016/j.jneumeth.2015.10.010 (section 2.4) before computing its
            eigendecomposition. None means no rescaling, 'auto' means automatic
            rescaling using singular values, otherwise the scaling factors.
    :type rescale_mode: {'auto'} or None or numpy.ndarray
    :param bool forward_backward: If True, the low-rank operator is computed
        like in fbDMD (reference: https://arxiv.org/abs/1507.02264). Default is
        False.
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

        X, Y = compute_tlsq(X, Y, self.tlsq_rank)
        self._svd_modes, _, _ = self.operator.compute_operator(X,Y)

        # Default timesteps
        self.original_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}
        self.dmd_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}

        self._b = self._compute_amplitudes()

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
        return np.linalg.multi_dot([self.modes, np.diag(self.eigs),
            pinv(self.modes), X])
