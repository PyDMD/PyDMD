"""
Derived module from dmdbase.py for dmd with control.

Reference:
- Proctor, J.L., Brunton, S.L. and Kutz, J.N., 2016. Dynamic mode decomposition
with control. SIAM Journal on Applied Dynamical Systems, 15(1), pp.142-161.
"""
from .dmdbase import DMDBase
from past.utils import old_div
import numpy as np


class DMDc(DMDBase):
    """
    Dynamic Mode Decomposition with control.
    This version does not allow to manipulate the temporal window within the
    system is reconstructed.

    :param svd_rank: the rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive interger, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation.
    :type svd_rank: int or float
    :param int tlsq_rank: rank truncation computing Total Least Square. Default
        is 0, that means no truncation.
    :param bool opt: flag to compute optimal amplitudes. See :class:`DMDBase`.
        Default is False.
    :param svd_rank_omega: the rank for the truncation of the aumented matrix
        omega composed by the left snapshots matrix and the control. Used only
        for the `_fit_B_unknown` method of this class. It should be greater or
        equal than `svd_rank`. For the possible values please refer to the
        `svd_rank` parameter description above.
    :param numpy.array rescale_mode: None means no rescaling, empty array means
        automatic rescaling using SV, otherwise the user chooses the preferred
        scaling.
    :type svd_rank_omega: int or float
    """
    def __init__(self, svd_rank=0, tlsq_rank=0, opt=False, svd_rank_omega=-1,
        rescale_mode=None):
        self.svd_rank = svd_rank
        self.tlsq_rank = tlsq_rank
        self.opt = opt
        self.svd_rank_omega = svd_rank_omega
        self.original_time = None
        self.rescale_mode = rescale_mode

        self._eigs = None
        self._Atilde = None
        self._B = None
        self._modes = None  # Phi
        self._b = None  # amplitudes
        self._snapshots = None
        self._snapshots_shape = None
        self._controlin = None
        self._controlin_shape = None
        self._basis = None

    @property
    def B(self):
        """
        Get the operator B.

        :return: the operator B.
        :rtype: numpy.ndarray
        """
        return self._B

    @property
    def basis(self):
        """
        Get the basis used to reduce the linear operator to the low dimensional
        space.

        :return: the matrix which columns are the basis vectors.
        :rtype: numpy.ndarray
        """
        return self._basis

    def reconstructed_data(self, control_input=None):
        """
        Return the reconstructed data, computed using the `control_input`
        argument. If the `control_input` is not passed, the original input (in
        the `fit` method) is used. The input dimension has to be consistent
        with the dynamics.

        :param numpy.ndarray control_input: the input control matrix.
        :return: the matrix that contains the reconstructed snapshots.
        :rtype: numpy.ndarray
        """
        if control_input is None:
            controlin, controlin_shape = self._controlin, self._controlin_shape
        else:
            controlin, controlin_shape = self._col_major_2darray(control_input)

        if controlin.shape[1] != self.dynamics.shape[1] - 1:
            raise RuntimeError(
                'The number of control inputs and the number of snapshots to reconstruct has to be the same'
            )

        eigs = np.power(self.eigs,
                        old_div(self.dmd_time['dt'], self.original_time['dt']))
        A = self.modes.dot(np.diag(eigs)).dot(np.linalg.pinv(self.modes))

        data = [self._snapshots[:, 0]]

        for i, u in enumerate(controlin.T):
            data.append(A.dot(data[i]) + self._B.dot(u))

        data = np.array(data).T
        return data

    def _fit_B_known(self, X, Y, B):
        """
        Private method that performs the dynamic mode decomposition algorithm
        with control when the matrix `B` is provided.

        :param numpy.ndarray X: the first matrix of original snapshots.
        :param numpy.ndarray Y: the second matrix of original snapshots.
        :param numpy.ndarray I: the input control matrix.
        :param numpy.ndarray B: the matrix B.
        """
        X, Y = self._compute_tlsq(X, Y, self.tlsq_rank)

        U, s, V = self._compute_svd(X, self.svd_rank)

        self._Atilde = U.T.conj().dot(Y - B.dot(self._controlin)).dot(V).dot(
            np.diag(np.reciprocal(s)))
        self._eigs, self._modes = self._eig_from_lowrank_op(
            self._Atilde, (Y - B.dot(self._controlin)), U, s, V, True)
        self._b = self._compute_amplitudes(self._modes, self._snapshots,
                                           self._eigs, self.opt)

        self._B = B
        self._basis = U
        return self

    def _fit_B_unknown(self, X, Y):
        """
        Private method that performs the dynamic mode decomposition algorithm
        with control when the matrix `B` is not provided.

        :param numpy.ndarray X: the first matrix of original snapshots.
        :param numpy.ndarray Y: the second matrix of original snapshots.
        """
        omega = np.vstack([X, self._controlin])

        Up, sp, Vp = self._compute_svd(omega, self.svd_rank_omega)

        Up1 = Up[:self._snapshots.shape[0], :]
        Up2 = Up[self._snapshots.shape[0]:, :]

        Ur, sr, Vr = self._compute_svd(Y, self.svd_rank)
        self._basis = Ur

        self._Atilde = Ur.T.conj().dot(Y).dot(Vp).dot(np.diag(
            np.reciprocal(sp))).dot(Up1.T.conj()).dot(Ur)
        self._Btilde = Ur.T.conj().dot(Y).dot(Vp).dot(np.diag(
            np.reciprocal(sp))).dot(Up2.T.conj())
        self._B = Ur.dot(self._Btilde)

        self._eigs, modes = np.linalg.eig(self._Atilde)
        self._modes = Y.dot(Vp).dot(np.diag(np.reciprocal(sp))).dot(
            Up1.T.conj()).dot(Ur).dot(modes)

        self._b = self._compute_amplitudes(self._modes, X, self._eigs, self.opt)

    def fit(self, X, I, B=None):
        """
        Compute the Dynamic Modes Decomposition with control given the original
        snapshots and the control input data. The matrix `B` that controls how
        the control input influences the system evolution can be provided by
        the user; otherwise, it is computed by the algorithm.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        :param I: the control input.
        :type I: numpy.ndarray or iterable
        :param numpy.ndarray B: matrix that controls the control input
            influences the system evolution.
        :type B: numpy.ndarray or iterable
        """
        self._snapshots, self._snapshots_shape = self._col_major_2darray(X)
        self._controlin, self._controlin_shape = self._col_major_2darray(I)

        n_samples = self._snapshots.shape[1]
        X = self._snapshots[:, :-1]
        Y = self._snapshots[:, 1:]

        self.original_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}
        self.dmd_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}

        if B is None:
            self._fit_B_unknown(X, Y)
        else:
            self._fit_B_known(X, Y, B)

        return self
