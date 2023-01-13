"""Derived module from dmdbase.py for sparsity-promoting DMD."""

import numpy as np
from numpy.linalg import solve
from scipy.sparse import (
    csc_matrix as sparse,
    vstack as spvstack,
    hstack as sphstack,
)
from scipy.sparse.linalg import spsolve

from .dmd import DMD


def soft_thresholding_operator(v, k):
    """
    Soft-thresholding operator as defined in 10.1063/1.4863670.

    :param np.ndarray v: The vector on which we apply the operator.
    :param float k: The threshold.
    :return np.ndarray: The result of the application of the soft-tresholding
        operator on ´v´.
    """
    return np.multiply(
        np.multiply(np.divide(1 - k, np.abs(v)), v), np.abs(v) > k
    )


class SpDMD(DMD):
    """
    Sparsity-Promoting Dynamic Mode Decomposition. Promotes solutions having an
    high number of amplitudes set to zero (i.e. *sparse solutions*).
    Reference: 10.1063/1.4863670

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
        Default is True.
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
    :param float abs_tolerance: Controls the convergence of ADMM. See
        :func:`_loop_condition` for more details.
    :param float rel_tolerance: Controls the convergence of ADMM. See
        :func:`_loop_condition` for more details.
    :param int max_iterations: The maximum number of iterations performed by
        ADMM, after that the algorithm is stopped.
    :param float rho: Controls the convergence of ADMM. For a reference on the
        optimal value for `rho` see 10.1109/TAC.2014.2354892 or
        10.3182/20120914-2-US-4030.00038.
    :param float gamma: Controls the level of "promotion" assigned to sparse
        solution. Increasing `gamma` will result in an higher number of
        zero-amplitudes.
    :param bool verbose: If `False`, the information provided by SpDMD (like
        the number of iterations performed by ADMM) are not shown.
    :param bool enforce_zero: If `True` the DMD amplitudes which should be set
        to zero according to the solution of ADMM are manually set to 0 (since
        we solve a sparse linear system to find the optimal vector of DMD
        amplitudes very small terms may survive in some cases).
    :param release_memory: If `True` the intermediate matrices computed by the
        algorithm are deleted after the termination of a call to :func:`fit`.
    """

    def __init__(self, svd_rank=0, tlsq_rank=0, exact=True, opt=False,
                 rescale_mode=None, forward_backward=False, sorted_eigs=False,
                 abs_tolerance=1.0e-6, rel_tolerance=1.0e-4,
                 max_iterations=10000, rho=1, gamma=10, verbose=True,
                 enforce_zero=True, release_memory=True,
                 zero_absolute_tolerance=1.e-12):
        super().__init__(
            svd_rank=svd_rank,
            tlsq_rank=tlsq_rank,
            exact=exact,
            opt=opt,
            rescale_mode=rescale_mode,
            forward_backward=forward_backward,
            sorted_eigs=sorted_eigs,
        )

        self.rho = rho
        self.gamma = gamma
        self._max_iterations = max_iterations
        self._abs_tol = abs_tolerance
        self._rel_tol = rel_tolerance
        self._verbose = verbose
        self._enforce_zero = enforce_zero
        self._release_memory = release_memory
        self._zero_absolute_tolerance = zero_absolute_tolerance

        self._P = None
        self._q = None
        self._Plow = None

        self._modes_activation_bitmask_proxy = None

    def fit(self, X):
        """
        Compute the Dynamic Modes Decomposition of the input data.
        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        """
        super().fit(X)

        P, q = self._optimal_dmd_matrices()

        self._P = sparse(P)
        self._q = q
        # Cholesky factorization of matrix P + (rho/2)*I
        Prho = P + np.identity(len(self.amplitudes)) * self.rho / 2
        self._Plow = np.linalg.cholesky(Prho)

        # find which amplitudes are to be set to 0
        zero_amplitudes = self._find_zero_amplitudes()

        # compute the (sparse) vector of optimal DMD amplitudes
        self._b = self._optimal_amplitudes(zero_amplitudes)
        # re-allocate the Proxy to avoid problems due to the fact that we
        # re-computed the amplitudes
        self._allocate_modes_bitmask_proxy()

        # release memory
        if self._release_memory:
            self._P = None
            self._q = None
            self._Plow = None

        return self

    def _update_alpha(self, beta, lmbd):
        """
        Update the vector :math:`\\alpha_k` of DMD amplitudes.
        :param np.ndarray beta: Current value of :math:`\\beta_k` (vector of
            non-zero amplitudes).
        :param np.ndarray lmbd: Current value of :math:`\\lambda_k` (vector of
            Lagrande multipliers).
        :return: The updated value :math:`\\alpha_{k+1}`.
        :rtype: np.ndarray
        """
        uk = beta - lmbd / self.rho
        return solve(
            self._Plow.conj().T, solve(self._Plow, self._q + uk * self.rho / 2)
        )

    def _update_beta(self, alpha, lmbd):
        """
        Update the vector :math:`\\beta` of non-zero amplitudes.
        :param np.ndarray alpha: Updated value of :math:`\\alpha_{k+1}` (vector
            of DMD amplitudes).
        :param np.ndarray lmbd: Current value of :math:`\\lambda_k` (vector
            of Lagrange multipliers).
        :return: The updated value :math:`\\beta_{k+1}`.
        :rtype: np.ndarray
        """
        return soft_thresholding_operator(
            alpha + lmbd / self.rho, self.gamma / self.rho
        )

    def _update_lagrangian(self, alpha, beta, lmbd):
        """
        Update the vector :math:`\\lambda` of Lagrange multipliers.
        :param np.ndarray alpha: Updated value of :math:`\\alpha_{k+1}` (vector
            of DMD amplitudes).
        :param np.ndarray beta: Updated value of :math:`\\beta_{k+1}` (vector
            of non-zero amplitudes).
        :param np.ndarray lmbd: Current value of :math:`\\lambda_k` (vector
            of Lagrange multipliers).
        :return: The updated value :math:`\\lambda_{k+1}`.
        :rtype: np.ndarray
        """
        return lmbd + (alpha - beta) * self.rho

    def _update(self, beta, lmbd):
        """
        Operate an entire step of ADMM.
        :param np.ndarray beta: Current value of :math:`\\beta_k` (vector of
            non-zero amplitudes).
        :param np.ndarray lmbd: Current value of :math:`\\lambda_k` (vector of
            Lagrande multipliers).
        :return: A tuple containing the updated values
            :math:`\\alpha_{k+1},\\beta_{k+1},\\lambda_{k+1}` (in this order).
        :rtype: tuple
        """
        a_new = self._update_alpha(beta, lmbd)
        b_new = self._update_beta(a_new, lmbd)
        l_new = self._update_lagrangian(a_new, b_new, lmbd)

        return a_new, b_new, l_new

    def _loop_condition(self, alpha, beta, lmbd, old_beta):
        """
        Check whether ADMM can stop now, or should perform another iteration.
        :param np.ndarray alpha: Current value of :math:`\\alpha_k` (vector
            of DMD amplitudes).
        :param np.ndarray beta: Current value of :math:`\\beta_k` (vector of
            non-zero amplitudes).
        :param np.ndarray lmbd: Current value of :math:`\\lambda_k` (vector
            of Lagrange multipliers).
        :param np.ndarray old_beta: Old value of :math:`\\beta_{k-1}` (vector
            of non-zero amplitudes).
        :return bool: `True` if ADMM can stop now, `False` otherwise.
        """
        primal_residual = np.linalg.norm(alpha - beta)
        dual_residual = self.rho * np.linalg.norm(beta - old_beta)

        eps_primal = np.sqrt(len(alpha)) * self._abs_tol + self._rel_tol * max(
            np.linalg.norm(alpha), np.linalg.norm(beta)
        )
        eps_dual = np.sqrt(
            len(alpha)
        ) * self._abs_tol + self._rel_tol * np.linalg.norm(lmbd)

        return primal_residual < eps_primal and dual_residual < eps_dual

    def _find_zero_amplitudes(self):
        """
        Use ADMM to find which amplitudes (i.e. their position in the
        DMD amplitudes array) which can be set to zero according to the given
        parameters. Note that this method does not compute amplitudes, but
        only which amplitudes are to be set to 0. Optimal amplitudes should be
        computed separately afterwards
        (see :func:`_find_sparsity_promoting_amplitudes`).
        :return np.ndarray: A boolean vector whose `True` items correspond to
            amplitudes which should be set to 0.
        """
        n_amplitudes = len(self.amplitudes)

        # initial values of lmbd and beta are all 0
        beta0 = np.zeros(n_amplitudes, dtype="complex")
        lmbd0 = np.zeros(n_amplitudes, dtype="complex")

        # perform a first step of ADMM
        alpha, beta, lmbd = self._update(beta0, lmbd0)
        old_beta = beta0

        # count the number of iterations of ADMM
        i = 0

        # at the beginning of each iteration check if ADMM can stop (because of
        # loop_condition or number of iterations)
        while (not self._loop_condition(alpha, beta, lmbd, old_beta) and
               i < self._max_iterations):
            i += 1

            old_beta = beta
            alpha, beta, lmbd = self._update(beta, lmbd)

        if self._verbose:
            print("ADMM: {} iterations".format(i))

        # zero values in beta are associated with DMD amplitudes which can be
        # set to 0
        return np.abs(old_beta) < self._zero_absolute_tolerance

    def _optimal_amplitudes(self, zero_amplitudes):
        """
        Find the optimal DMD amplitudes with the constraint that the given
        indexes should be set to 0.
        :param np.ndarray zero_amplitudes: Boolean vector.
        :return np.ndarray: Vector of optimal DMD amplitudes. Amplitudes at
            indexes corresponding to `True` indexes in `zero_amplitudes` are
            set to 0.
        """
        n_amplitudes = len(self.amplitudes)
        n_of_zero = np.count_nonzero(zero_amplitudes)

        # vectors of the canonical base of R^n_amplitudes, from which we
        # extract only those corresponding to items set to 0 in zero_amplitudes
        E = np.identity(n_amplitudes)[:, zero_amplitudes]

        # left hand side of the linear system whose solution is the vector of
        # optimal DMD amplitudes.
        KKT = spvstack(
            [
                sphstack([self._P, E], format="csc"),
                sphstack(
                    [
                        E.conj().T,
                        sparse((n_of_zero, n_of_zero), dtype="complex"),
                    ],
                    format="csc",
                ),
            ],
            format="csc",
        )

        # right hand side of the linear system
        rhs = np.concatenate(
            (
                self._q,
                np.zeros((n_of_zero,)),
            )
        )

        opt_amps = spsolve(KKT, rhs)[:n_amplitudes]
        if self._enforce_zero:
            opt_amps[zero_amplitudes] = 0
        return opt_amps
