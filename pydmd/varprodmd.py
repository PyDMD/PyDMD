"""
Variable Projection for DMD. Reformulation of original paper
(https://epubs.siam.org/doi/abs/10.1137/M1124176) s.t. sparse matrix computation
is substiuted by outer products. Further the optimization is
reformulated s.t. SciPy's nonlinear least squares optimizer
can handle "complex" parameters.

Default optimizer arguments:
    OPT_DEF_ARGS =
        {"method": "trf",
        "tr_solver": "exact",
        "loss": "linear",
        "x_scale": "jac",
        "gtol": 1e-8,
        "xtol": 1e-8,
        "ftol": 1e-8}
"""

import warnings
from types import MappingProxyType
from typing import Any, Dict, Tuple, Union

import numpy as np
from scipy.linalg import qr
from scipy.optimize import OptimizeResult, least_squares

from .dmd import DMDBase
from .dmdoperator import DMDOperator
from .snapshots import Snapshots
from .utils import compute_svd

OPT_DEF_ARGS = MappingProxyType(
    {
        "method": "trf",
        "tr_solver": "exact",
        "loss": "linear",
        "x_scale": "jac",
        "gtol": 1e-8,
        "xtol": 1e-8,
        "ftol": 1e-8,
    }
)


def _compute_dmd_ev(
    x_current: np.ndarray,
    x_next: np.ndarray,
    rank: Union[float, int] = 0,
) -> np.ndarray:
    r"""
    Compute DMD eigenvalues.

    :param x_current: Observables :math:`\boldsymbol{X}`
    :type x_current: np.ndarray
    :param x_next: Observables :math:`\boldsymbol{X}'`
    :type x_current: np.ndarray
    :param rank: Desired rank. If rank :math:`r = 0`, the optimal rank is
        determined automatically. If rank is a float s.t. :math: `0 < r < 1`,
        the cumulative energy of the singular values is used
        to determine the optimal rank. If rank is an integer and :math:`r > 0`,
        the desired rank is used iff possible. Defaults to 0.
    :type rank: Union[float, int], optional
    :return: Diagonal matrix of eigenvalues
        :math:`\boldsymbol{\Lambda}` as 1d array.
    :rtype: np.ndarray
    """

    u_x, sigma_x, v_x = compute_svd(x_current, rank)

    # columns of v need to be multiplicated with inverse sigma
    sigma_inv_approx = np.reciprocal(sigma_x)

    a_approx = np.linalg.multi_dot(
        [u_x.conj().T, x_next, sigma_inv_approx[None] * v_x]
    )

    return np.linalg.eigvals(a_approx)


class _OptimizeHelper:  # pylint: disable=too-few-public-methods
    """
    Helper Class to store intermediate results during the optimization.
    """

    __slots__ = ["phi", "phi_inv", "u_svd", "s_inv", "v_svd", "b_matrix", "rho"]

    def __init__(self, l_in: int, m_in: int, n_in: int):
        self.phi = np.empty((m_in, l_in), dtype=np.complex128)
        self.u_svd = np.empty((m_in, l_in), dtype=np.complex128)
        self.s_inv = np.empty((l_in,), dtype=np.complex128)
        self.v_svd = np.empty((l_in, l_in), dtype=np.complex128)
        self.b_matrix = np.empty((l_in, n_in), dtype=np.complex128)
        self.rho = np.empty((m_in, n_in), dtype=np.complex128)


def _compute_dmd_rho(
    alphas: np.ndarray,
    time: np.ndarray,
    data: np.ndarray,
    opthelper: _OptimizeHelper,
) -> np.ndarray:
    r"""
    Compute the real residual vector :math:`\boldsymbol{\rho}` for
    Levenberg-Marquardt update.

    :param alphas: DMD eigenvalues to optimize,
        where :math:`\alpha \in \mathbb{C}^l`,
        but here :math:`\alpha \in \mathbb{R}^{2l}`,
        since optimizer cannot deal with complex numbers.
    :type alphas: np.ndarray
    :param time: 1D time array.
    :type time: np.ndarray
    :param data: data :math:`\boldsymbol{Y} \n C^{m \times n}`.
        For DMD computation we set :math:`\boldsymbol{Y} = \boldsymbol{X}^T`.
    :type data: np.ndarray
    :param opthelper: Optimization helper to speed up computations
        mainly for Jacobian.
    :type opthelper: _OptimizeHelper
    :return: 1d residual vector for Levenberg-Marquardt update
        :math:`\boldsymbol{\rho} \in \mathbb{R}^{2mn}`.
    :rtype: np.ndarray
    """

    _alphas = np.zeros((alphas.shape[-1] // 2,), dtype=np.complex128)
    _alphas.real = alphas[: alphas.shape[-1] // 2]
    _alphas.imag = alphas[alphas.shape[-1] // 2 :]

    phi = np.exp(np.outer(time, _alphas))
    u_phi, s_phi, v_phi_t = np.linalg.svd(phi, full_matrices=False)
    idx = np.where(s_phi.real != 0.0)[0]
    s_phi_inv = np.zeros_like(s_phi)
    s_phi_inv[idx] = np.reciprocal(s_phi[idx])

    rho = data - np.linalg.multi_dot([u_phi, u_phi.conj().T, data])
    rho_flat = np.ravel(rho)
    rho_out = np.zeros((2 * rho_flat.shape[-1],), dtype=np.float64)
    rho_out[: rho_flat.shape[-1]] = rho_flat.real
    rho_out[rho_flat.shape[-1] :] = rho_flat.imag

    opthelper.phi = phi
    opthelper.u_svd = u_phi
    opthelper.s_inv = s_phi_inv
    opthelper.v_svd = v_phi_t.conj().T
    opthelper.rho = rho
    opthelper.b_matrix = np.linalg.multi_dot(
        [
            opthelper.v_svd * s_phi_inv[None],
            opthelper.u_svd.conj().T,
            data,
        ]
    )
    return rho_out


def _compute_dmd_jac(
    alphas: np.ndarray,
    time: np.ndarray,
    data: np.ndarray,
    opthelper: _OptimizeHelper,
) -> np.ndarray:
    r"""
    Compute the real Jacobian.
    SciPy's nonlinear least squares optimizer requires real entities.
    Therefore, complex and real parts are split.

    :param alphas: DMD eigenvalues to optimize,
        where :math:`\alpha \in \mathbb{C}^l`,
        but here :math:`\alpha \in \mathbb{R}^{2l}` since optimizer cannot
        deal with complex numbers.
    :type alphas: np.ndarray
    :param time: 1D time array.
    :type time: np.ndarray
    :param data: data :math: `\boldsymbol{Y} \n C^{m \times n}`.
        For DMD computation we set :math:`\boldsymbol{Y} = \boldsymbol{X}^T`.
    :type data: np.ndarray
    :param opthelper: Optimization helper to speed up computations
        mainly for Jacobian. The entities are computed in `_compute_dmd_rho`.
    :type opthelper: _OptimizeHelper
    :return: Jacobian :math:`\boldsymbol{J} \in \mathbb{R}^{mn \times 2l}`.
    :rtype: np.ndarray
    """

    _alphas = np.zeros((alphas.shape[-1] // 2,), dtype=np.complex128)
    _alphas.real = alphas[: alphas.shape[-1] // 2]
    _alphas.imag = alphas[alphas.shape[-1] // 2 :]
    jac_out = np.zeros((2 * np.prod(data.shape), alphas.shape[-1]))

    for j in range(_alphas.shape[-1]):
        d_phi_j = time * opthelper.phi[:, j]
        outer = np.outer(d_phi_j, opthelper.b_matrix[j])
        a_j = outer - np.linalg.multi_dot(
            [opthelper.u_svd, opthelper.u_svd.conj().T, outer]
        )
        g_j = np.linalg.multi_dot(
            [
                opthelper.u_svd * opthelper.s_inv[None],
                np.outer(
                    opthelper.v_svd[j].conj(), d_phi_j.conj() @ opthelper.rho
                ),
            ]
        )
        # Compute the jacobian J_mat_j = - (A_j + G_j).
        jac = -a_j - g_j
        jac_flat = np.ravel(jac)

        # construct the overall jacobian for optimized
        # J_real = |Re{J} -Im{J}|
        #          |Im{J}  Re{J}|

        # construct real part for optimization
        jac_out[: jac_out.shape[0] // 2, j] = jac_flat.real
        jac_out[jac_out.shape[0] // 2 :, j] = jac_flat.imag

        # construct imaginary part for optimization
        jac_out[
            : jac_out.shape[0] // 2, _alphas.shape[-1] + j
        ] = -jac_flat.imag
        jac_out[jac_out.shape[0] // 2 :, _alphas.shape[-1] + j] = jac_flat.real

    return jac_out


def _varpro_preprocessing(
    data: np.ndarray,
    time: np.ndarray,
    rank: Union[float, int] = 0.0,
    use_proj: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Preprocess data for Variable Projection: Calculate
    :math:`\boldsymbol{Y}, \boldsymbol{Z}` for trapezoidal derivative
    approximation. If desired input data is projected to
    low-dimensional space.

    :param data: data matrix s.t. :math:`X \n C^{n \times m}`.
    :type data: np.ndarray
    :param time: 1d array of timestamps.
    :type time: np.ndarray
    :param optargs: Arguments for 'least_squares' optimizer.
    :type optargs: Dict[str, Any]
    :param rank: Desired rank. If rank :math:`r = 0`, the optimal rank is
        determined automatically. If rank is a float s.t. :math:`0 < r < 1`,
        the cumulative energy of the singular values is used
        to determine the optimal rank. If rank is an integer
        and :math:`r > 0`, the desired rank is used iff possible.
        Defaults to 0.
    :type rank: Union[float, int], optional
    :param use_proj: Perform variable projection in
        low dimensional space if `use_proj=True`, else in the original space.
        Defaults to True.
    :type use_proj: bool, optional
    :return: Derivative :math:`\boldsymbol{Y},\boldsymbol{Z}`, (projected) data,
         rank reduced projection matrix :math:`\boldsymbol{U}_r`.
    :rtype: Tuple[np.ndarray,
                  np.ndarray,
                  np.ndarray,
                  np.ndarray]
    """

    u_r, s_r, v_r = compute_svd(data, rank)
    data_out = v_r.conj().T * s_r[:, None] if use_proj else data

    # trapezoidal derivative approximation
    y_out = (data_out[:, :-1] + data_out[:, 1:]) / 2.0
    dt_in = time[1:] - time[:-1]
    z_out = (data_out[:, 1:] - data_out[:, :-1]) / dt_in[None]

    return z_out, y_out, data_out, u_r


def _compute_dmd_varpro(
    alphas_init: np.ndarray,
    time: np.ndarray,
    data: np.ndarray,
    opthelper: _OptimizeHelper,
    **optargs,
) -> OptimizeResult:
    r"""
    Compute Variable Projection (VarPro) for DMD using SciPy's
    nonlinear least squares optimizer.

    :type alphas_init: np.ndarray
    :param time: 1d time array.
    :type time: np.ndarray
    :param data: data :math:`\boldsymbol{Y} \n C^{m \times n}`.
        For DMD computation we set :math:`\boldsymbol{Y} = \boldsymbol{X}^T`.
    :type data: np.ndarray
    :param opthelper: Optimization helper to speed up computations
        mainly for Jacobian. The entities are computed in `_compute_dmd_rho`
        and are used in `_compute_dmd_jac`.
    :type opthelper: _OptimizeHelper
    :return: Optimization result.
    :rtype: OptimizeResult
    """

    return least_squares(
        _compute_dmd_rho,
        alphas_init,
        _compute_dmd_jac,
        **optargs,
        args=[time, data, opthelper],
    )


def select_best_samples_fast(data: np.ndarray, comp: float = 0.9) -> np.ndarray:
    r"""
    Select library samples using QR decomposition with column pivoting.

    :param data: Data matrix :math:`\boldsymbol{X} \in \mathbb{C}^{n \times m}`.
    :type data: np.ndarray
    :param comp: Library compression :math:`c`, where :math:`0 < c < 1`.
        The best fitting :math:`\lfloor \left(1 - c\right)m\rfloor` samples
        are selected. Defaults to 0.9.
    :type comp: float, optional
    :raises ValueError: ValueError is raised if data matrix is not a
        2d array.
    :raises ValueError: ValueError is raised of compression is not in required
        interval (:math:`0 < r < 1`).
    :return: Indices of selected samples as 1d array.
    :rtype: np.ndarray
    """

    if len(data.shape) != 2:
        raise ValueError("Expected 2D array!")

    if not 0 < comp < 1:
        raise ValueError("Compression must be in (0, 1)]")

    n_samples = int(data.shape[-1] * (1.0 - comp))
    pcolumn = qr(data, mode="economic", pivoting=True)[-1]

    return pcolumn[:n_samples]


def compute_varprodmd_any(  # pylint: disable=unused-variable
    data: np.ndarray,
    time: np.ndarray,
    optargs: Dict[str, Any],
    rank: Union[float, int] = 0.0,
    use_proj: bool = True,
    compression: float = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, OptimizeResult]:
    r"""
    Compute DMD given arbitary timesteps.

    :param data: data matrix s.t. :math:`X \n C^{n \times m}`.
    :type data: np.ndarray
    :param time: 1d array of timestamps.
    :type time: np.ndarray
    :param optargs: Arguments for 'least_squares' optimizer.
    :type optargs: Dict[str, Any]
    :param rank: Desired rank. If rank :math:`r = 0`, the optimal rank is
        determined automatically. If rank is a float s.t. :math:`0 < r < 1`,
        the cumulative energy of the singular values is used
        to determine the optimal rank. If rank is an integer
        and :math:`r > 0`, the desired rank is used iff possible.
        Defaults to 0.
    :type rank: Union[float, int], optional
    :param use_proj: Perform variable projection in
        low dimensional space if `use_proj=True`, else in the original space.
        Defaults to True.
    :type use_proj: bool, optional
    :param compression: If libary compression :math:`c = 0`,
        all samples are used. If :math:`0 < c < 1`, the best
        fitting :math:`\lfloor \left(1 - c\right)m\rfloor` samples
        are selected.
    :type compression: float, optional
    :raises ValueError: ValueError is raised if data matrix is not a
        2d array.
    :raises ValueError: ValueError is raised if time is not a
        1d array.
    :return: DMD modes :math:`\boldsymbol{\Phi}`, continuous DMD eigenvalues
        :math: `\boldsymbol{\Omega}` as 1d array,
        DMD eigenfunctions or amplitudes :math:`\boldsymbol{\varphi}`,
        optimization results of SciPy's nonlinear least squares optimizer.
    :rtype: Tuple[np.ndarray,
                  np.ndarray,
                  np.ndarray,
                  np.ndarray,
                  OptimizeResult]
    """

    if len(data.shape) != 2:
        raise ValueError("data needs to be 2D array")

    if len(time.shape) != 1:
        raise ValueError("time needs to be a 1D array")

    #  y_in, z_in, data_in, u_r
    res = _varpro_preprocessing(data, time, rank, use_proj)
    omegas = _compute_dmd_ev(res[0], res[1], res[-1].shape[-1])

    if compression > 0:
        indices = select_best_samples_fast(res[2], compression)

        if indices.size <= 1:
            indices = np.arange(res[2])

    else:
        indices = np.arange(res[2].shape[-1])

    if res[2].shape[-1] < omegas.shape[-1]:
        warnings.warn(
            " ".join(
                (
                    "Attempting to solve underdetermined system.",
                    "Decrease desired rank or compression!",
                )
            )
        )

    opthelper = _OptimizeHelper(res[-1].shape[-1], *res[2].shape)
    opt = _compute_dmd_varpro(
        np.concatenate([omegas.real, omegas.imag]),
        time[indices],
        res[2][:, indices].T,
        opthelper,
        **optargs,
    )
    omegas.real = opt.x[: opt.x.shape[-1] // 2]
    omegas.imag = opt.x[opt.x.shape[-1] // 2 :]
    xi = res[-1] @ opthelper.b_matrix.T if use_proj else opthelper.b_matrix.T
    eigenf = np.linalg.norm(xi, axis=0)
    return xi / eigenf[None], omegas, eigenf, indices, opt


def varprodmd_predict(
    phi: np.ndarray,
    omegas: np.ndarray,
    eigenf: np.ndarray,
    time: np.ndarray,
) -> np.ndarray:
    r"""
    Perform DMD prediction using computed modes, continuous eigenvalues
    and eigenfunctions/amplitudes.

    :param phi: DMD modes
        :math:`\boldsymbol{\Phi} \in \mathbb{C}^{n \times \left(m-1\right)}`.
    :type phi: np.ndarray
    :param omegas: Continuous diagonal matrix of eigenvalues
        :math:`\boldsymbol{\Omega} \in
            \mathbb{C}^{\left(m-1\right) \times \left(m-1\right)}`
        as 1d array.
    :type omegas: np.ndarray
    :param eigenf: Eigenfunctions or amplitudes
        :math:`\boldsymbol{\varphi} \in \mathbb{C}^{m - 1}`.
    :type eigenf: np.ndarray
    :param time: 1d array of timestamps.
    :type time: np.ndarray
    :return: Reconstructed/predicted state :math:`\hat{\boldsymbol{X}}`.
    :rtype: np.ndarray
    """

    return phi @ (np.exp(np.outer(omegas, time)) * eigenf[:, None])


class VarProOperator(DMDOperator):
    """
    Variable Projection Operator for DMD.
    """

    def __init__(
        self,
        svd_rank: Union[float, int],
        exact: bool,
        sorted_eigs: Union[bool, str],
        compression: float,
        optargs: Dict[str, Any],
    ):
        r"""
        VarProOperator constructor.

        :param svd_rank: Desired SVD rank.
            If rank :math:`r = 0`, the optimal rank is
            determined automatically. If rank is a float s.t.
            :math:`0 < r < 1`, the cumulative energy
            of the singular values is used to determine
            the optimal rank.
            If rank is an integer and :math:`r > 0`,
            the desired rank is used iff possible.
        :type svd_rank: Union[float, int]
        :param exact: Perform computations in original state space
            if `exact=True`, else perform optimization
            in projected (low dimensional) space.
        :type exact: bool
        :param sorted_eigs: Sort eigenvalues.
            If `sorted_eigs=True`, the variance of the absolute values
            of the complex eigenvalues
            :math:`\sqrt{\omega_i \cdot \bar{\omega}_i}`,
            the variance absolute values of the real parts
            :math:`\left|\Re\{{\omega_i}\}\right|`
            and the variance of the absolute values
            of the imaginary parts :math:`\left|\Im\{{\omega_i}\}\right|`
            is computed.
            The eigenvalues are then sorted according
            to the highest variance (from highest to lowest).
            If `sorted_eigs=False`, no sorting is performed.
            If the parameter is a string and set to sorted_eigs='auto',
            the eigenvalues are sorted accoring to the variances
            of previous mentioned quantities.
            If `sorted_eigs='real'` the eigenvalues are sorted
            w.r.t. the absolute values of the real parts
            of the eigenvalues (from highest to lowest).
            If `sorted_eigs='imag'` the eigenvalues are sorted
            w.r.t. the absolute values of the imaginary parts
            of the eigenvalues (from highest to lowest).
            If `sorted_eigs='abs'` the eigenvalues are sorted
            w.r.t. the magnitude of the eigenvalues
            :math:`\left(\sqrt{\omega_i \cdot \bar{\omega}_i}\right)`
            (from highest to lowest).
        :type sorted_eigs: Union[bool, str]
        :param compression: If libary compression :math:`c = 0`,
            all samples are used. If :math:`0 < c < 1`, the best
            fitting :math:`\lfloor \left(1 - c\right)m\rfloor` samples
            are selected.
        :type compression: float
        :param optargs: Arguments for 'least_squares' optimizer.
            Use `OPT_DEF_ARGS` as starting point.
        :type optargs: Dict[str, Any]
        """

        super().__init__(svd_rank, exact, False, None, sorted_eigs, False)
        self._sorted_eigs = sorted_eigs
        self._svd_rank = svd_rank
        self._exact = exact
        self._optargs: Dict[str, Any] = optargs
        self._compression: float = compression
        self._modes: np.ndarray = None
        self._eigenvalues: np.ndarray = None

    def compute_operator(
        self, X: np.ndarray, Y: np.ndarray
    ) -> Tuple[np.ndarray, OptimizeResult, np.ndarray]:
        r"""
        Perform Variable Projection for DMD using SciPy's
        nonlinear least squares optimizer.

        :param X: Measurement :math:`\boldsymbol{X} \in \mathbb{C}^{n \times m}`
        :type X: np.ndarray
        :param Y: 1d array of timestamps where individual
            measurements :math:`\boldsymbol{x}_i \in \mathbb{C}^n`
            where taken.
        :type Y: np.ndarray
        :raises ValueError: If `sorted_eigs` from constructor
            was set to an invalid string.
        :return: Eigenfunctions/amplitudes :math:`\boldsymbol{\varphi}^{m-1}`,
            OptimizeResult from SciPy's optimizer
            (optimal parameters and statistics),
            indices of selected measurements. If no compression (:math:`c = 0`)
            is used, all indices are returned, else the indices of the selected
            samples are used.
        :rtype: Tuple[np.ndarray, OptimizeResult, np.ndarray]
        """
        (
            self._modes,
            self._eigenvalues,
            eigenf,
            indices,
            opt,
        ) = compute_varprodmd_any(
            X,
            Y,
            self._optargs,
            self._svd_rank,
            not self._exact,
            self._compression,
        )

        # overwrite for lazy sorting
        if isinstance(self._sorted_eigs, bool):
            if self._sorted_eigs:
                self._sorted_eigs = "auto"

        if isinstance(self._sorted_eigs, str):
            if self._sorted_eigs == "auto":
                eigs_real = self._eigenvalues.real
                eigs_imag = self._eigenvalues.imag
                _eigs_abs = np.abs(self._eigenvalues)
                var_real = np.var(eigs_real)
                var_imag = np.var(eigs_imag)
                var_abs = np.var(_eigs_abs)
                array = np.array([var_real, var_imag, var_abs])
                eigs_abs = (eigs_real, eigs_imag, _eigs_abs)[np.argmax(array)]

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
    """
    Variable Projection for DMD using SciPy's
    nonlinear least squares solver. The original problem
    is reformulated s.t. complex residuals and Jacobians, which are used
    by the Levenberg-Marquardt algorithm, are transormed to real numbers.
    Further simplifications (outer products) avoids using sparse matrices.
    """

    def __init__(  # pylint: disable=super-init-not-called
        self,
        svd_rank: Union[float, int] = 0,
        exact: bool = False,
        sorted_eigs: Union[bool, str] = False,
        compression: float = 0.0,
        optargs: Dict[str, Any] = None,
    ):
        r"""
        VarProDMD constructor.

        :param svd_rank: Desired SVD rank.
            If rank :math:`r = 0`, the optimal rank is
            determined automatically. If rank is a float s.t. :math:`0 < r < 1`,
            the cumulative energy of the singular values is used
            to determine the optimal rank.
            If rank is an integer and :math:`r > 0`,
            the desired rank is used iff possible. Defaults to 0.
        :type svd_rank: Union[float, int], optional
        :param exact: Perform variable projection in
            low dimensional space if `exact=False`.
            Else the optimization is performed
            in the original space.
            Defaults to False.
        :type exact: bool, optional
        :param sorted_eigs: Sort eigenvalues.
            If `sorted_eigs=True`, the variance of the absolute values
            of the complex eigenvalues
            :math:`\left(\sqrt{\omega_i \cdot \bar{\omega}_i}\right)`,
            the variance absolute values of the real parts
            :math:`\left|\Re\{{\omega_i}\}\right|`
            and the variance of the absolute values of the imaginary parts
            :math:`\left|\Im\{{\omega_i}\}\right|` is computed.
            The eigenvalues are then sorted according
            to the highest variance (from highest to lowest).
            If `sorted_eigs=False`, no sorting is performed.
            If the parameter is a string and set to sorted_eigs='auto',
            the eigenvalues are sorted accoring to the variances
            of previous mentioned quantities.
            If `sorted_eigs='real'` the eigenvalues are sorted
            w.r.t. the absolute values of the real parts
            of the eigenvalues (from highest to lowest).
            If `sorted_eigs='imag'` the eigenvalues are sorted
            w.r.t. the absolute values of the imaginary parts
            of the eigenvalues (from highest to lowest).
            If `sorted_eigs='abs'` the eigenvalues are sorted
            w.r.t. the magnitude of the eigenvalues
            :math:`\left(\sqrt{\omega_i \cdot \bar{\omega}_i}\right)`
            (from highest to lowest).
            Defaults to False.
        :type sorted_eigs: Union[bool, str], optional
        :param compression: If libary compression :math:`c = 0`,
            all samples are used. If :math:`0 < c < 1`, the best
            fitting :math:`\lfloor \left(1 - c\right)m\rfloor` samples
            are selected.
        :type compression: float, optional
        :param optargs: Arguments for 'least_squares' optimizer.
            If set to None, `OPT_DEF_ARGS` are used as default parameters.
            Defaults to None.
        :type optargs: Dict[str, Any], optional
        """

        # super constructor not called
        # as most of the attributes are
        # not used.

        if optargs is None:
            optargs = OPT_DEF_ARGS

        self._Atilde = VarProOperator(
            svd_rank, exact, sorted_eigs, compression, optargs
        )
        self._optres: OptimizeResult = None
        self._snapshots_holder: Snapshots = None
        self._indices: np.ndarray = None
        self._modes_activation_bitmask_proxy = None

    def fit(self, X: np.ndarray, time: np.ndarray) -> object:
        r"""
        Fit the eigenvalues, modes and eigenfunctions/amplitudes
        to measurements.

        :param X: Measurements
            :math:`\boldsymbol{X} \in \mathbb{C}^{n \times m}`.
        :type X: np.ndarray
        :param time: 1d array of timestamps where measurements were taken.
        :type time: np.ndarray
        :return: VarProDMD instance.
        :rtype: object
        """

        self._snapshots_holder = Snapshots(X)
        (self._b, self._optres, self._indices) = self._Atilde.compute_operator(
            self._snapshots_holder.snapshots.astype(np.complex128), time
        )
        self._original_time = time
        self._dmd_time = time[self._indices]

        return self

    def forecast(self, time: np.ndarray) -> np.ndarray:
        r"""
        Forecast measurements at given timestamps `time`.

        :param time: Desired times for forcasting as 1d array.
        :type time: np.ndarray
        :raises ValueError: If method `fit(X, time)` was not called.
        :return: Predicted measurements :math:`\hat{\boldsymbol{X}}`.
        :rtype: np.ndarray
        """

        if not self.fitted:
            raise ValueError("Nothing fitted yet. Call fit-method first!")

        return varprodmd_predict(
            self._Atilde.modes, self._Atilde.eigenvalues, self._b, time
        )

    @property
    def ssr(self) -> float:
        """
        Compute the square root of sum squared residual (SSR) taken from
        https://link.springer.com/article/10.1007/s10589-012-9492-9.
        The SSR gives insight w.r.t. signal qualities.
        A low SSR is desired. If SSR is high the model may be inaccurate.

        :raises ValueError: ValueError is raised if method
            `fit(X, time)` was not called.
        :return: SSR.
        :rtype: float
        """

        if not self.fitted:
            raise ValueError("Nothing fitted yet!")

        rho_flat_real = self._optres.fun
        rho_flat_imag = np.zeros(
            (rho_flat_real.size // 2,), dtype=np.complex128
        )
        rho_flat_imag.real = rho_flat_real[: rho_flat_real.size // 2]
        rho_flat_imag.imag = rho_flat_real[rho_flat_real.size // 2 :]

        sigma = np.linalg.norm(rho_flat_imag)
        denom = max(
            self._original_time.size
            - self._optres.jac.shape[0] // 2
            - self._optres.jac.shape[1] // 2,
            1,
        )
        ssr = sigma / np.sqrt(float(denom))

        return ssr

    @property
    def selected_samples(self) -> np.ndarray:
        r"""
        Return indices for creating the library.

        :raises ValueError: ValueError is raised if method
            `fit(X, time)` was not called.
        :return: Indices of the selected samples.
            If no compression was performed :math:`\left(c = 0\right)`,
            all indices are returned, else indices of the
            library selection scheme using QR-Decomposition
            with column pivoting.
        :rtype: np.ndarray
        """

        if not self.fitted:
            raise ValueError("Nothing fitted yet. Call fit-method first!")

        return self._indices

    @property
    def opt_stats(self) -> OptimizeResult:
        """
        Return optimization statistics of the Variable Projection
        optimization.

        :raises ValueError: ValueError is raised if method `fit(X, time)`
            was not called.
        :return: Optimization results including optimal weights
            (continuous eigenvalues) and number of iterations.
        :rtype: OptimizeResult
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
        return self.amplitudes[:, None] * t_omega

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
