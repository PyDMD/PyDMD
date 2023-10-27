"""
Variable Projection for DMD. Reformulation of original paper
(https://epubs.siam.org/doi/abs/10.1137/M1124176) s.t. sparse matrix computation
is substiuted by outer products. Further the optimization is
reformulated s.t. SciPy's nonlinear least squares optimizer
can handle "complex" parameters. 
"""
import warnings
from typing import Any, Dict, Tuple, Union
import numpy as np
from scipy.optimize import OptimizeResult, least_squares
from scipy.linalg import qr
from .dmd import DMDBase
from .dmdoperator import DMDOperator
from .snapshots import Snapshots


OPT_DEF_ARGS: Dict[str, Any] = {  # pylint: disable=unused-variable
    "method": 'trf',
    "tr_solver": 'exact',
    'loss': 'linear',
    "x_scale": 'jac',
    "gtol": 1e-8,
    "xtol": 1e-8,
    "ftol": 1e-8
}


def __svht(sigma_svd: np.ndarray,  # pylint: disable=unused-variable
           rows: int,
           cols: int,
           sigma: float = None) -> int:
    """
    Determine optimal rank for svd matrix approximation,
    based on https://arxiv.org/pdf/1305.5870.pdf.

    Args:
        sigma_svd (np.ndarray):
            Diagonal matrix of "enonomy" svd

        rows (int):
            number of rows of original data matrix

        cols (int):
            number of columns of original data matrix

        sigma (float, optional):
            Signal noise if known. Defaults to None.

    Raises:
        ValueError: If sigma_svd is not a 1d-array

    Returns:
        int: optimal rank
    """

    if len(sigma_svd.shape) != 1:
        raise ValueError("Expected 1d array for diagonal matrix!")

    beta = float(cols) / \
        float(rows) if rows > cols else float(rows) / float(cols)
    tau_star = 0

    if sigma is not None:
        sigma = abs(sigma)
        lambda_star = np.sqrt(
            2 * (beta + 1) + \
            (8 * beta / (beta + 1 + np.sqrt(beta * beta + 14 * beta + 1))))
        tau_star = lambda_star * sigma * \
            np.sqrt(float(cols) if cols > rows else float(rows))
    else:

        median = np.median(sigma_svd)
        beta_square = beta * beta
        beta_cubic = beta * beta_square
        omega = 0.56 * beta_cubic - 0.95 * beta_square + 1.82 * beta + 1.43
        tau_star = median * omega

    rank = np.where(sigma_svd >= tau_star)[0]
    r_out = 0
    if rank.size == 0:
        r_out = sigma_svd.shape[-1]

    else:
        r_out = rank[-1] + 1
    return r_out


def __compute_rank(sigma_x: np.ndarray,
                   rows: int,
                   cols: int,
                   rank: Union[float, int],
                   sigma: float = None) -> int:
    """
    Compute rank without duplicate SVD computation.
    """
    if 0 < rank < 1:
        cumulative_energy = np.cumsum(
            np.square(sigma_x) / np.square(sigma_x).sum())
        __rank = np.searchsorted(cumulative_energy, rank) + 1
    elif rank == 0:
        __rank = __svht(sigma_x, rows, cols, sigma)
    elif rank >= 1:
        __rank = int(rank)
    else:
        raise ValueError(f"Invalid rank specified, provided {rank}!")
    return min(__rank, sigma_x.size)


def __compute_dmd_ev(x_current: np.ndarray,  # pylint: disable=unused-variable
                     x_next: np.ndarray,
                     rank: Union[float, int] = 0) -> np.ndarray:
    """Compute DMD eigenvalues.

    Args:
        x_current (np.ndarray):
            Observable $X$.

        x_next (np.ndarray):
            Observable $X'$.

        rank (int, optional):
            Rank for computation. If 0. svht calculates optimal rank.
            Defaults to 0.

    Returns: DMD eigenvalues $\\lambda$
    """

    u_x, sigma_x, v_x_t = np.linalg.svd(
        x_current, full_matrices=False, hermitian=False)
    __rank = __compute_rank(
        sigma_x, x_current.shape[0], x_current.shape[1], rank)

    # columns of v need to be multiplicated with inverse sigma
    sigma_inv_approx = np.reciprocal(sigma_x[:__rank])

    a_approx = np.linalg.multi_dot([u_x[:, :__rank].conj().T,
                                    x_next,
                                    sigma_inv_approx.reshape(1, -1) * \
                                    v_x_t[:__rank, :].conj().T])

    return np.linalg.eigvals(a_approx)


class OptimizeHelper:
    """ Helper Class to store intermediate results
    """
    __slots__ = ["phi", "phi_inv", "u_svd",
                 "s_inv", "v_svd", "b_matrix", "rho"]

    def __init__(self, l_in: int, m_in: int, n_in: int) -> None:
        self.phi: np.ndarray = np.empty((m_in, l_in), dtype=np.complex128)
        self.u_svd: np.ndarray = np.empty((m_in, l_in), dtype=np.complex128)
        self.s_inv: np.ndarray = np.empty((l_in,), dtype=np.complex128)
        self.v_svd: np.ndarray = np.empty((l_in, l_in), dtype=np.complex128)
        self.b_matrix: np.ndarray = np.empty((l_in, n_in), dtype=np.complex128)
        self.rho: np.ndarray = np.empty((m_in, n_in), dtype=np.complex128)


def __compute_dmd_rho(alphas: np.ndarray,
                      time: np.ndarray,
                      data: np.ndarray,
                      opthelper: OptimizeHelper) -> np.ndarray:
    r"""Compute the residual for DMD

    Args:
        alphas (np.ndarray):
            DMD eigenvalues to optimize,
            where :math: `\alpha \in \mathbb{C}^l`,
            but here :math: `\alpha \in \mathbb{R}^{2l}`
            since optimizer cannot deal with complex numbers.

        time (np.ndarray):
            1D time array.

        data (np.ndarray):
            data :math: `X \n C^{m \times n}`.

        opthelper (OptimizeHelper):
            Optimization helper to speed up 
            computations mainly for jacobian.

    Returns:
        np.ndarray: 1D resudial :math: `\rho \in \mathbb{R}^{2mn}`.
    """

    __alphas = np.zeros((alphas.shape[-1] // 2,), dtype=np.complex128)
    __alphas.real = alphas[:alphas.shape[-1] // 2]
    __alphas.imag = alphas[alphas.shape[-1] // 2:]

    phi = np.exp(np.outer(time, __alphas))
    __u, __s, __v_t = np.linalg.svd(phi, hermitian=False, full_matrices=False)
    __idx = np.where(__s.real != 0.)[0]
    __s_inv = np.zeros_like(__s)
    __s_inv[__idx] = np.reciprocal(__s[__idx])

    rho = data - np.linalg.multi_dot([__u, __u.conj().T, data])
    rho_flat = np.ravel(rho)
    rho_out = np.zeros((2 * rho_flat.shape[-1], ), dtype=np.float64)
    rho_out[:rho_flat.shape[-1]] = rho_flat.real
    rho_out[rho_flat.shape[-1]:] = rho_flat.imag

    opthelper.phi = phi
    opthelper.u_svd = __u
    opthelper.s_inv = __s_inv
    opthelper.v_svd = __v_t.conj().T
    opthelper.rho = rho
    opthelper.b_matrix = np.linalg.multi_dot([opthelper.v_svd * \
                                              __s_inv.reshape((1, -1)),
                                              opthelper.u_svd.conj().T,
                                              data])
    return rho_out


def __compute_dmd_jac(alphas: np.ndarray,
                      time: np.ndarray,
                      data: np.ndarray,
                      opthelper: OptimizeHelper) -> np.ndarray:
    r"""Compute the Jacobian.
       Note that the Jacobian needs to be real,
       so complex and real parts are split.

    Args:
        alphas (np.ndarray):
            DMD eigenvalues to optimize,
            where :math: `\alpha \in \mathbb{C}^l`,
            but here :math: `\alpha \in \mathbb{R}^{2l}` since optimizer cannot
            deal with complex numbers.

        time (np.ndarray):
            1D time array.

        data (np.ndarray):
            data :math: `X \n C^{m \times n}`

        opthelper (OptimizeHelper):
            Optimization helper to speed up computations mainly for
            jacobian. The entities are computed in '__compute_dmd_rho'.

    Returns:
        np.ndarray: Jacobian :math: `J \in \mathbb{R}^{mn \times 2l}`.
    """
    __alphas = np.zeros((alphas.shape[-1] // 2,), dtype=np.complex128)
    __alphas.real = alphas[:alphas.shape[-1] // 2]
    __alphas.imag = alphas[alphas.shape[-1] // 2:]
    jac_out = np.zeros((2 * np.prod(data.shape), alphas.shape[-1]))

    for j in range(__alphas.shape[-1]):
        d_phi_j = time * opthelper.phi[:, j]
        __outer = np.outer(d_phi_j, opthelper.b_matrix[j, :])
        __a_j = __outer - \
            np.linalg.multi_dot(
                [opthelper.u_svd, opthelper.u_svd.conj().T, __outer])
        __g_j = np.linalg.multi_dot([opthelper.u_svd * \
                                     opthelper.s_inv.reshape((1, -1)),
                                     np.outer(opthelper.v_svd[j, :].conj(),
                                              d_phi_j.conj() @ opthelper.rho)])
        # Compute the jacobian J_mat_j = - (A_j + G_j).
        __jac = -__a_j - __g_j
        __jac_flat = np.ravel(__jac)

        # construct the overall jacobian for optimized
        # J_real = |Re{J} -Im{J}|
        #          |Im{J}  Re{J}|

        # construct real part for optimization
        jac_out[:jac_out.shape[0] // 2, j] = __jac_flat.real
        jac_out[jac_out.shape[0] // 2:, j] = __jac_flat.imag

        # construct imaginary part for optimization
        jac_out[:jac_out.shape[0] // 2,
                __alphas.shape[-1] + j] = -__jac_flat.imag
        jac_out[jac_out.shape[0] // 2:,
                __alphas.shape[-1] + j] = __jac_flat.real

    return jac_out


def __compute_dmd_varpro(alphas_init: np.ndarray,
                         time: np.ndarray,
                         data: np.ndarray,
                         opthelper: OptimizeHelper,
                         **optargs) -> OptimizeResult:
    r"""Compute Variable Projection (VarPro) for DMD

    Args:
        alphas_init (np.ndarray):
            Initial DMD eigenvalues s.t. :math: `\alpha \in \mathbb{R}^{2l}`.
            Normally :math: `\alpha \in \mathbb{R}^{l}`, but optimizer
            requires real numbers.

        time (np.ndarray):
            1D time array.

        data (np.ndarray):
            data (np.ndarray): data :math: `X \n C^{m \times n}`.

    Returns:
        OptimizeResult: Optimization result.
    """
    return least_squares(__compute_dmd_rho,
                         alphas_init,
                         __compute_dmd_jac,
                         **optargs, args=[time, data, opthelper])


def select_best_samples_fast(data: np.ndarray,
                             comp: float = 0.9) -> np.ndarray:
    """Select library samples fast QR decomposition with pivoting.
       comp defines the compression.


    Args:
        data (np.ndarray):
            Input data

        eps (float, optional):
            Threshold for sum of absolute normalized
            dot products.

    Raises:
        ValueError: If data is not a 2D array.

    Raises:
        ValueError: If not 0 <= comp < 1

    Returns:
        np.ndarray: indices.
    """
    if len(data.shape) != 2:
        raise ValueError("Expected 2D array!")

    if not 0 < comp < 1:
        raise ValueError("Compression must be in (0, 1)]")

    n_samples = int(data.shape[-1]*(1. - comp))
    pcolumn = qr(data, mode='economic', pivoting=True)[-1]
    __idx = pcolumn[:n_samples]
    return __idx


def compute_varprodmd_any(data: np.ndarray,  # pylint: disable=unused-variable
                          time: np.ndarray,
                          optargs: Dict[str, Any],
                          rank: Union[float, int] = 0.,
                          use_proj: bool = True,
                          compression: float = 0) -> Tuple[np.ndarray,
                                                           np.ndarray,
                                                           np.ndarray,
                                                           np.ndarray,
                                                           OptimizeResult]:
    """Compute DMD given arbitary timesteps.

    Args:
        data (np.ndarray): data (np.ndarray):
            data :math: `X \n C^{n \times m}`.

        rank (Union[float, int], optional):
            Rank for initial DMD computation. Defaults to 0.
            If rank :math: `r = 0`, the rank is chosen automatically,
            else desired rank is used.

        optargs (Dict[str, Any], optional):
            Default arguments for 'least_squares' optimizer.
            Defaults to None. Use 'OPT_DEF_ARGS'.

    Raises:
        ValueError: If data is not a 2D array.
        ValueError: If time is not a 1D array.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, OptimizeResult]:
            Modes, cont. eigenvalues, eigenfunctions and OptimizationResult.
    """
    if len(data.shape) != 2:
        raise ValueError("data needs to be 2D array")

    if len(time.shape) != 1:
        raise ValueError("time needs to be a 1D array")

    u_r, s_r, v_r_t = np.linalg.svd(data, full_matrices=False)
    __rank = __compute_rank(s_r, data.shape[0], data.shape[1], rank)
    u_r = u_r[:, :__rank]
    s_r = s_r[:__rank]
    v_r = v_r_t[:__rank, :].conj().T
    data_in = v_r.conj().T * s_r.reshape((-1, 1)) if use_proj else data

    # trapezoidal derivative approximation
    y_in = (data_in[:, :-1] + data_in[:, 1:]) / 2.
    dt_in = time[1:] - time[:-1]
    z_in = (data_in[:, 1:] - data_in[:, :-1]) / dt_in.reshape((1, -1))
    omegas = __compute_dmd_ev(y_in, z_in, __rank)
    omegas_in = np.zeros((2*omegas.shape[-1],), dtype=np.float64)
    omegas_in[:omegas.shape[-1]] = omegas.real
    omegas_in[omegas.shape[-1]:] = omegas.imag

    if compression > 0:
        __idx = select_best_samples_fast(data_in, compression)
        if __idx.size > 1:
            indices = __idx
        else:
            indices = np.arange(data_in.shape[-1])
        data_in = data_in[:, indices]
        time_in = time[indices]

    else:
        time_in = time
        indices = np.arange(data_in.shape[-1])

    if data_in.shape[-1] < omegas.shape[-1]:
        msg = "Attempting to solve underdeterimined system. "
        msg += "Decrease desired rank or compression!"
        warnings.warn(msg)

    opthelper = OptimizeHelper(u_r.shape[-1], *data_in.shape)
    opt = __compute_dmd_varpro(
        omegas_in, time_in, data_in.T, opthelper, **optargs)
    omegas.real = opt.x[:opt.x.shape[-1] // 2]
    omegas.imag = opt.x[opt.x.shape[-1] // 2:]
    xi = u_r @ opthelper.b_matrix.T if use_proj else opthelper.b_matrix.T
    eigenf = np.linalg.norm(xi, axis=0)
    return xi / eigenf.reshape((1, -1)), omegas, eigenf, indices, opt


def optdmd_predict(phi: np.ndarray,  # pylint: disable=unused-variable
                   omegas: np.ndarray,
                   eigenf: np.ndarray,
                   time: np. ndarray) -> np.ndarray:
    """ Perform DMD prediction

    Args:
        phi (np.ndarray): DMD modes
        omegas (np.ndarray): DMD cont. eigenvalues.
        eigenf (np.ndarray): DMD eigenfunctions / eigenvalues.
        time (np.ndarray): 1D time array.

    Returns:
        np.ndarray: Prediction of DMD given parameters.
    """
    return phi @ (np.exp(np.outer(omegas, time)) * eigenf.reshape(-1, 1))


class VarProOperator(DMDOperator):
    """Variable Projection Operator

    Args:
        DMDOperator (DMDOperator):
            The classic DMD operator
    """

    def __init__(self,
                 svd_rank: Union[float, int],
                 exact: bool,
                 sorted_eigs: Union[bool, str],
                 compression: float,
                 optargs: Dict[str, Any]):
        """"""
        super().__init__(svd_rank,
                         exact,
                         False,
                         None,
                         sorted_eigs,
                         False)
        self._sorted_eigs = sorted_eigs
        self._svd_rank = svd_rank
        self._exact = exact
        self._optargs: Dict[str, Any] = optargs
        self._compression: float = compression
        self._modes: np.ndarray = None
        self._eigenvalues: np.ndarray = None

    def compute_operator(self, data: np.ndarray, time: np.ndarray) \
          -> Tuple[np.ndarray,
                   OptimizeResult]:
        """Compute the VarProDMD operator

        Args:
            data (np.ndarray):
                Measurements/Observables

            time (Union[np.ndarray, float]):
                Timesteps or sampling time. If time is float
                the parameter is interpreted as sampling time,
                else it is interpreted as 1D array

        Raises:
            ValueError: If sorted_eigs parameter is not supported.

        Returns:
            Tuple[np.ndarray, OptimizeResult]:
                DMD amplitudes and the optimization result.
        """

        (self._modes,
         self._eigenvalues,
         eigenf,
         indices,
         opt) = compute_varprodmd_any(data,
                                      time,
                                      self._optargs,
                                      self._svd_rank,
                                      not self._exact,
                                      self._compression)

        # overwrite for lazy sorting
        if isinstance(self._sorted_eigs, bool):
            if self._sorted_eigs:
                self._sorted_eigs = "auto"

        if isinstance(self._sorted_eigs, str):
            if self._sorted_eigs == "auto":
                eigs_real = self._eigenvalues.real
                eigs_imag = self._eigenvalues.imag
                __eigs_abs = np.abs(self._eigenvalues)
                var_real = np.var(eigs_real)
                var_imag = np.var(eigs_imag)
                var_abs = np.var(__eigs_abs)
                __array = np.array([var_real, var_imag, var_abs])
                eigs_abs = (eigs_real, eigs_imag, __eigs_abs)[
                    np.argmax(__array)]

            elif self._sorted_eigs == "real":
                eigs_abs = np.abs(self._eigenvalues.real)

            elif self._sorted_eigs == "imag":
                eigs_abs = np.abs(self._eigenvalues.imag)

            elif self._sorted_eigs == "abs":
                eigs_abs = np.abs(self._eigenvalues)
            else:
                raise ValueError(f"{self._sorted_eigs} not supported!")

            idx = np.argsort(eigs_abs)[::-1]  # sort from biggest to smallest
            self._eigenvalues = self._eigenvalues[idx]
            self._modes = self._modes[:, idx]
            eigenf = eigenf[idx]

        return eigenf, opt, indices


class VarProDMD(DMDBase):
    """Variable Projection for DMD.
       Variable Projection is reformulated for SciPy's
       nonlinear least squares solver.
       Further simplifications avoid using sparse matrices.

    Args:
        DMDBase (DMDBase): DMDBase class
    """

    def __init__(self,
                 svd_rank: Union[float, int] = 0,
                 exact: bool = False,
                 sorted_eigs: Union[bool, str] = False,
                 compression: float = 0.,
                 optargs: Dict[str, Any] = None):
        """VarProDMD constructor

        Args:
            svd_rank (Union[float, int], optional):
                Rank for initial DMD computation. Defaults to 0.
                If rank :math: `r = 0`, the rank is chosen automatically,
                else desired rank is used.

            exact (bool, optional): 
                Compute exact VarProDMD (no projection) if True,
                else compute VarProDMD in low dimensional space.
                Defaults to False.

            sorted_eigs (Union[bool, str], optional): 
                Sort eigenvalues.
                If sorted_eigs is a string, supported modes are
                ["auto", "real", "imag", "abs"].
                If sorted_eigs is bool and True sorting is set to "auto",
                else no sorting is performed.
                Defaults to False.

            compression (float, optional):
                Library compression. If 0, no preselection is performed,
                else (1. - compression) samples are selected. Defaults to 0..

            optargs (Dict[str, Any], optional):
                Optimizer arguments for 
                Nonlinear Least Square Optmizer.
                Defaults to OPT_DEF_ARGS.
        """
        if optargs is None:
            optargs = OPT_DEF_ARGS

        self._Atilde = VarProOperator(
            svd_rank, exact, sorted_eigs, compression, optargs)
        self._optres: OptimizeResult = None
        self._snapshots_holder: Snapshots = None
        self._indices: np.ndarray = None
        self._modes_activation_bitmask_proxy = None

    def fit(self, data: np.ndarray, time: np.ndarray) -> object:
        """ Fit the eigenvalues, modes and amplitudes to data

        Args:
            data (np.ndarray): Data input
            time (np.ndarray): Measured timestamps.

        Returns:
            object: Reference to VarProDMD instance.
        """

        self._snapshots_holder = Snapshots(data)
        self._b, self._optres, self._indices = self._Atilde.compute_operator(
            data, time)
        self._original_time = time
        self._dmd_time = time[self._indices]

        return self

    def forecast(self, time: np.ndarray) -> np.ndarray:
        """ Forecast at given timesteps

        Args:
            time (np.ndarray): 1D time array.

        Raises:
            ValueError: If method fit was not called.

        Returns:
            np.ndarray: Forecast
        """
        if not self.fitted:
            raise ValueError("Nothing fitted yet!")

        return optdmd_predict(self._Atilde.modes,
                              self._Atilde.eigenvalues,
                              self._b, time)

    @property
    def ssr(self) -> float:
        """Sum squared resdiual

        Raises:
            ValueError: If method fit was not called.

        Returns:
            float: Sum Squared Residual (SSR), the smaller the better.
        """
        if not self.fitted:
            raise ValueError("Nothing fitted yet!")

        rho_flat_real = self._optres.fun
        rho_flat_imag = np.zeros(
            (rho_flat_real.size // 2,), dtype=np.complex128)
        rho_flat_imag.real = rho_flat_real[:rho_flat_real.size // 2]
        rho_flat_imag.imag = rho_flat_real[rho_flat_real.size // 2:]

        sigma = np.linalg.norm(rho_flat_imag)
        denom = max(self._original_time.size -
                    self._optres.jac.shape[0] // 2
                    - self._optres.jac.shape[1] // 2, 1)
        ssr = sigma / np.sqrt(float(denom))

        return ssr

    @property
    def selected_samples(self) -> np.ndarray:
        """Return indices OptDMD for creating the library

        Raises:
            ValueError: If method fit was not called.

        Returns:
            np.ndarray: Indices
        """
        if not self.fitted:
            raise ValueError("Nothing fitted yet!")

        return self._indices

    @property
    def opt_stats(self) -> OptimizeResult:
        """Return optimization statistics

        Raises:
            ValueError: If method fit was not called.

        Returns:
            OptimizeResult: Result of optimization.
        """
        if not self.fitted:
            raise ValueError("Nothing fitted yet!")

        return self._optres

    @property
    def dynamics(self):
        """
        Get the time evolution of each mode.

        :return: matrix that contains all the time evolution, stored by row.
        :rtype: numpy.ndarray
        """
        t_omega = np.exp(np.outer(self.eigs, self._original_time))
        return self.amplitudes.reshape(-1, 1) * t_omega

    @property
    def frequency(self):
        """
        Get the amplitude spectrum.

        :return: the array that contains the frequencies of the eigenvalues.
        :rtype: numpy.ndarray
        """
        return self.eigs.imag / (2 * np.pi)

    @property
    def growth_rate(self):
        """
        Get the growth rate values relative to the modes.

        :return: the Floquet values
        :rtype: numpy.ndarray
        """
        return self.eigs.real
