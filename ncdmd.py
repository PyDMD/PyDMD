"""
Derived module from dmdbase.py for dmd with sensor noise.
Adopted from:
    Dawson et al.,"Charaterizing and correcting for the effect of sensor noise in 
    the dynamic mode decomposition", 2016, Section 2.3
"""
from .dmdbase import DMDBase


class ncDMD(DMDBase):
    """
    Dynamic Mode Decomposition, when you have the noise signal properties

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
    :param bool opt: flag to compute optimized DMD. Default is False.
    """

    def fit(self, Xm, Noise):
        """
        Compute the Dynamic Modes Decomposition to the input data.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        """
        self._snapshots, self._snapshots_shape = self._col_major_2darray(Xm)
        snapshots_noise, snapshots_noise_shape = self._col_major_2darray(Noise)

        n_samples = self._snapshots.shape[1]
        X = self._snapshots[:, :-1]
        Y = self._snapshots[:, 1:]
        n, m = X.shape # n = no_time, m = no_voxels
        NX = snapshots_noise[:, :-1]
        NY = snapshots_noise[:, 1:]
        
        
        X, Y = self._compute_tlsq(X, Y, self.tlsq_rank)

        ###
        # $A = A_m ( I - E(N_x.N_x^T) (X.X^T)^{-1})^{-1}$
        implementation_method = 1
        if implementation_method ==1:
            import numpy as np
            from numpy.linalg import inv
            svd_rank0 = self.svd_rank
            U, s, V = self._compute_svd(X, svd_rank0)
            I_r = np.identity(svd_rank0)
            NX_r = NX.dot(V)
            X_r = X.dot(V)
            cov_noise = np.cov(NX_r.T)
            X_XT_inv = inv(X_r.T.dot(X_r))
            Atilde_m = self._build_lowrank_op(U, s, V, Y)
            self._Atilde = np.dot(Atilde_m , inv(I_r- np.dot(cov_noise,X_XT_inv)))
        elif implementation_method == 2:
            U, s, V = self._compute_svd(X, self.svd_rank)
            Atilde_m = self._build_lowrank_op(U, s, V, Y)
            Xm = X
            U, s, V = self._compute_svd(X, self.svd_rank)
            Atilde_m = self._build_lowrank_op(U, s, V, Y)
            
        ###
        #U, s, V = self._compute_svd(X, self.svd_rank)
        #self._Atilde = self._build_lowrank_op(U, s, V, Y)

        self._eigs, self._modes = self._eig_from_lowrank_op(
            self._Atilde, Y, U, s, V, self.exact)

        self._b = self._compute_amplitudes(self._modes, self._snapshots,
                                           self._eigs, self.opt)

        # Default timesteps
        self.original_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}
        self.dmd_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}

        return self
