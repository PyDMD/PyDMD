"""
Test module for VarProDMD
"""
import numpy as np
import pytest

from pydmd import VarProDMD
from pydmd.varprodmd import (
    OPT_DEF_ARGS,
    _compute_dmd_jac,
    _compute_dmd_rho,
    _OptimizeHelper,
    compute_varprodmd_any,
    varprodmd_predict,
    select_best_samples_fast,
)


def signal(x_loc: np.ndarray, time: np.ndarray) -> np.ndarray:
    """
    construct complex spatio temporal signal for testing.
    :param x_loc: 1d x-coordinate.
    :type x_loc: np.ndarray
    :param time: 1d time array.
    :type time: np.ndarray
    :return: Spatiotemporal signal.
    :rtype: np.ndarray
    """
    f_1 = 1.0 / np.cosh(x_loc + 3) * np.exp(1j * 2.3 * time)
    f_2 = 2.0 / np.cosh(x_loc) * np.tanh(x_loc) * np.exp(1j * 2.8 * time)
    return f_1 + f_2


def test_varprodmd_rho():
    """
    Unit test for residual vector :math: `\boldsymbol{\rho}`.
    """
    data = np.eye(2, 2).astype(np.complex128)
    time = np.array([0.0, 1.0], np.float64)
    alphas = np.array([1.0 + 0j, 1.0 + 0j], np.complex128)
    alphas_in = np.array([1.0, 1.0, 0.0, 0.0], np.float64)
    phi = np.exp(np.outer(time, alphas))
    U_svd, s_svd, V_svd_t = np.linalg.svd(
        phi, hermitian=False, full_matrices=False
    )
    idx = np.where(s_svd.real != 0.0)[0]
    s_inv = np.zeros_like(s_svd)
    s_inv[idx] = np.reciprocal(s_svd[idx])

    res = data - np.linalg.multi_dot([U_svd, U_svd.conj().T, data])
    res_flat = np.ravel(res)
    res_flat_reals = np.zeros((2 * res_flat.shape[-1]))
    res_flat_reals[: res_flat_reals.shape[-1] // 2] = res_flat.real
    res_flat_reals[res_flat_reals.shape[-1] // 2 :] = res_flat.imag
    opthelper = _OptimizeHelper(2, *data.shape)
    rho_flat_out = _compute_dmd_rho(alphas_in, time, data, opthelper)

    assert np.array_equal(rho_flat_out, res_flat_reals)
    assert np.array_equal(U_svd, opthelper.u_svd)
    assert np.array_equal(s_inv, opthelper.s_inv)
    assert np.array_equal(V_svd_t.conj().T, opthelper.v_svd)
    assert np.array_equal(phi, opthelper.phi)


def test_varprodmd_jac():  # pylint: disable=too-many-locals,too-many-statements
    """
    Test Jacobian computation (real vs. complex).
    """
    data = np.eye(2, 2).astype(np.complex128)
    time = np.array([0.0, 1.0])
    alphas = np.array([-1.0 + 0.0j, -2.0 + 0.0j], np.complex128)
    alphas_in = np.array([-1.0, -2.0, 0.0, 0.0], np.float64)
    phi = np.exp(np.outer(time, alphas))
    d_phi_1 = np.zeros((2, 2), dtype=np.complex128)
    d_phi_2 = np.zeros((2, 2), dtype=np.complex128)
    d_phi_1[:, 0] = time * phi[:, 0]
    d_phi_2[:, 1] = time * phi[:, 1]

    U_svd, s_svd, __v = np.linalg.svd(phi, hermitian=False, full_matrices=False)
    idx = np.where(s_svd.real != 0.0)[0]
    s_inv = np.zeros_like(s_svd)
    s_inv[idx] = np.reciprocal(s_svd[idx])
    phi_inv = (__v.conj().T * s_inv.reshape((1, -1))) @ U_svd.conj().T

    opthelper = _OptimizeHelper(2, *data.shape)
    opthelper.u_svd = U_svd
    opthelper.v_svd = __v.conj().T
    opthelper.s_inv = s_inv
    opthelper.phi = phi
    opthelper.phi_inv = phi_inv
    opthelper.b_matrix = phi_inv @ data
    opthelper.rho = data - phi @ opthelper.b_matrix
    rho_flat = np.ravel(opthelper.rho)
    rho_real = np.zeros((2 * rho_flat.shape[0]))
    rho_real[: rho_flat.shape[0]] = rho_flat.real
    rho_real[rho_flat.shape[0] :] = rho_flat.imag
    A_1 = d_phi_1 @ opthelper.b_matrix - np.linalg.multi_dot(
        [U_svd, U_svd.conj().T, d_phi_1, opthelper.b_matrix]
    )

    A_2 = d_phi_2 @ opthelper.b_matrix - np.linalg.multi_dot(
        [U_svd, U_svd.conj().T, d_phi_2, opthelper.b_matrix]
    )

    G_1 = np.linalg.multi_dot(
        [phi_inv.conj().T, d_phi_1.conj().T, opthelper.rho]
    )
    G_2 = np.linalg.multi_dot(
        [phi_inv.conj().T, d_phi_2.conj().T, opthelper.rho]
    )
    J_1 = -A_1 - G_1
    J_2 = -A_2 - G_2
    J_1_flat = np.ravel(J_1)
    J_2_flat = np.ravel(J_2)
    JAC_IMAG = np.zeros((J_1_flat.shape[0], 2), dtype=np.complex128)
    JAC_IMAG[:, 0] = J_1_flat
    JAC_IMAG[:, 1] = J_2_flat
    JAC_REAL = np.zeros((2 * J_1_flat.shape[-1], 4), dtype=np.float64)
    JAC_REAL[: J_1_flat.shape[-1], 0] = J_1_flat.real
    JAC_REAL[J_1_flat.shape[-1] :, 0] = J_1_flat.imag
    JAC_REAL[: J_2_flat.shape[-1], 1] = J_2_flat.real
    JAC_REAL[J_2_flat.shape[-1] :, 1] = J_2_flat.imag
    JAC_REAL[: J_1_flat.shape[-1], 2] = -J_1_flat.imag
    JAC_REAL[J_1_flat.shape[-1] :, 2] = J_1_flat.real
    JAC_REAL[: J_2_flat.shape[-1], 3] = -J_2_flat.imag
    JAC_REAL[J_2_flat.shape[-1] :, 3] = J_2_flat.real
    JAC_OUT_REAL = _compute_dmd_jac(alphas_in, time, data, opthelper)

    GRAD_REAL = JAC_REAL.T @ rho_real
    GRAD_OUT_REAL = JAC_OUT_REAL.T @ rho_real
    GRAD_IMAG = JAC_IMAG.conj().T @ rho_flat

    assert np.linalg.norm(JAC_REAL - JAC_OUT_REAL) < 1e-12
    assert np.linalg.norm(GRAD_REAL - GRAD_OUT_REAL) < 1e-12

    imag2real = np.zeros_like(GRAD_REAL)
    imag2real[: imag2real.shape[-1] // 2] = GRAD_IMAG.real
    imag2real[imag2real.shape[-1] // 2 :] = GRAD_IMAG.imag

    rec_grad = np.zeros_like(GRAD_IMAG)
    rec_grad.real = GRAD_REAL[: GRAD_REAL.shape[-1] // 2]
    rec_grad.imag = GRAD_REAL[GRAD_REAL.shape[-1] // 2 :]

    # funny numerical errors leads to
    # np.array_equal(GRAD_IMAG, __rec_grad) to fail
    assert np.linalg.norm(GRAD_IMAG - rec_grad) < 1e-9

    rec_grad = np.zeros_like(GRAD_IMAG)
    rec_grad.real = GRAD_OUT_REAL[: GRAD_OUT_REAL.shape[-1] // 2]
    rec_grad.imag = GRAD_OUT_REAL[GRAD_OUT_REAL.shape[-1] // 2 :]

    # funny numerical errors leads to
    # np.array_equal(GRAD_IMAG, __rec_grad) to fail
    assert np.linalg.norm(GRAD_IMAG - rec_grad) < 1e-9


def test_varprodmd_any():
    """
    Test Variable Projection function for DMD (at any timestep).
    """
    time = np.linspace(0, 4 * np.pi, 100)
    x_loc = np.linspace(-10, 10, 1024)

    z = signal(*np.meshgrid(x_loc, time)).T

    idx = select_best_samples_fast(z, 0.6)

    z_sub = z[:, idx]
    t_sub = time[idx]

    with pytest.raises(ValueError):
        compute_varprodmd_any(
            z_sub[:, 0],
            t_sub,
            OPT_DEF_ARGS,
            rank=0.0,
        )

    with pytest.raises(ValueError):
        compute_varprodmd_any(
            z_sub, t_sub.reshape((-1, 1)), OPT_DEF_ARGS, rank=0.0
        )

    phi, lambdas, eigenf, _, _ = compute_varprodmd_any(
        z_sub, t_sub, OPT_DEF_ARGS, rank=0.0
    )
    pred = varprodmd_predict(phi, lambdas, eigenf, time)
    diff = np.abs(pred - z)
    mae = np.sum(np.sum(diff, axis=0), axis=-1) / z.shape[0] / z.shape[-1]

    assert mae < 1.0

    phi, lambdas, eigenf, _, _ = compute_varprodmd_any(
        z_sub, t_sub, OPT_DEF_ARGS, rank=0.0, use_proj=False
    )
    pred = varprodmd_predict(phi, lambdas, eigenf, time)
    diff = np.abs(pred - z)
    mae = np.sum(np.sum(diff, axis=0), axis=-1) / z.shape[0] / z.shape[-1]

    assert mae < 1.0


def test_varprodmd_class():
    """
    Test VarProDMD class.
    """
    time = np.linspace(0, 4 * np.pi, 100)
    x_loc = np.linspace(-10, 10, 1024)

    z = signal(*np.meshgrid(x_loc, time)).T
    dmd = VarProDMD(0, False, False, 0)

    with pytest.raises(ValueError):
        _ = dmd.forecast(time)

    with pytest.raises(ValueError):
        _ = dmd.ssr

    with pytest.raises(ValueError):
        _ = dmd.selected_samples

    with pytest.raises(ValueError):
        _ = dmd.opt_stats

    dmd.fit(z, time)
    assert dmd.fitted
    assert dmd.eigs.size > 0
    assert len(dmd.modes.shape) == 2
    assert dmd.amplitudes.size > 0
    assert len(dmd.dynamics.shape) == 2
    assert dmd.amplitudes.size == dmd.frequency.size
    assert dmd.growth_rate.size == dmd.amplitudes.size
    assert dmd.eigs.size == dmd.amplitudes.size

    pred = dmd.forecast(time)

    diff = np.abs(pred - z)
    mae = np.sum(np.sum(diff, axis=0), axis=-1) / z.shape[0] / z.shape[-1]

    assert mae < 1
    assert dmd.ssr < 1e-3

    dmd = VarProDMD(0, False, "unkown_sort", 0.8)

    with pytest.raises(ValueError):
        dmd.fit(z, time)

    sort_args = ["auto", "real", "imag", "abs", True, False]

    for arg in sort_args:
        dmd = VarProDMD(0, False, arg, 0.6)
        dmd.fit(z, time)
        pred = dmd.forecast(time)
        diff = np.abs(pred - z)
        mae = np.sum(np.sum(diff, axis=0), axis=-1) / z.shape[0] / z.shape[-1]
        assert dmd.selected_samples.size == int((1 - 0.6) * 100)
        assert mae < 1.0
