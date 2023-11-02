"""
Test module for VarProDMD
"""
import numpy as np
import pytest
from pydmd.utils import compute_rank
from pydmd.varprodmd import (
    __OptimizeHelper,
    __compute_dmd_rho,
    __compute_dmd_jac,
    __compute_rank,
    __svht,
    compute_varprodmd_any,
    optdmd_predict,
    select_best_samples_fast,
    OPT_DEF_ARGS,
    VarProDMD,
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
    __f_1 = 1.0 / np.cosh(x_loc + 3) * np.exp(1j * 2.3 * time)
    __f_2 = 2.0 / np.cosh(x_loc) * np.tanh(x_loc) * np.exp(1j * 2.8 * time)
    return __f_1 + __f_2


def test_rank():
    """
    Test SVHT rank (no duplicate SVD computation is performed).
    """
    time = np.linspace(0, 4 * np.pi, 100)
    x_loc = np.linspace(-10, 10, 1024)
    __x, __time = np.meshgrid(x_loc, time)
    z = signal(__x, __time).T
    s = np.linalg.svd(z, full_matrices=False)[1]
    assert __compute_rank(s, z.shape[0], z.shape[1], 0) == compute_rank(z, 0)
    assert __compute_rank(s, z.shape[0], z.shape[1], 4) == 4
    assert __compute_rank(s, z.shape[0], z.shape[1], 0.8) == compute_rank(
        z, 0.8
    )
    assert __svht(s, z.shape[0], z.shape[1]) == compute_rank(z, 0)


def test_varprodmd_rho():
    """
    Unit test for residual vector :math: `\boldsymbol{\rho}`.
    """
    data = np.eye(2, 2).astype(np.complex128)
    time = np.array([0.0, 1.0], np.float64)
    alphas = np.array([1.0 + 0j, 1.0 + 0j], np.complex128)
    alphas_in = np.array([1.0, 1.0, 0.0, 0.0], np.float64)
    phi = np.exp(np.outer(time, alphas))
    __u, __s, __v_t = np.linalg.svd(phi, hermitian=False, full_matrices=False)
    __idx = np.where(__s.real != 0.0)[0]
    __s_inv = np.zeros_like(__s)
    __s_inv[__idx] = np.reciprocal(__s[__idx])

    res = data - np.linalg.multi_dot([__u, __u.conj().T, data])
    res_flat = np.ravel(res)
    res_flat_reals = np.zeros((2 * res_flat.shape[-1]))
    res_flat_reals[: res_flat_reals.shape[-1] // 2] = res_flat.real
    res_flat_reals[res_flat_reals.shape[-1] // 2 :] = res_flat.imag
    opthelper = __OptimizeHelper(2, *data.shape)
    rho_flat_out = __compute_dmd_rho(alphas_in, time, data, opthelper)
    assert np.array_equal(rho_flat_out, res_flat_reals)
    assert np.array_equal(__u, opthelper.u_svd)
    assert np.array_equal(__s_inv, opthelper.s_inv)
    assert np.array_equal(__v_t.conj().T, opthelper.v_svd)

    assert np.array_equal(phi, opthelper.phi)


def test_varprodmd_jac():
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

    __u, __s, __v = np.linalg.svd(phi, hermitian=False, full_matrices=False)
    __idx = np.where(__s.real != 0.0)[0]
    __s_inv = np.zeros_like(__s)
    __s_inv[__idx] = np.reciprocal(__s[__idx])
    phi_inv = (__v.conj().T * __s_inv.reshape((1, -1))) @ __u.conj().T

    opthelper = __OptimizeHelper(2, *data.shape)
    opthelper.u_svd = __u
    opthelper.v_svd = __v.conj().T
    opthelper.s_inv = __s_inv
    opthelper.phi = phi
    opthelper.phi_inv = phi_inv
    opthelper.b_matrix = phi_inv @ data
    opthelper.rho = data - phi @ opthelper.b_matrix
    rho_flat = np.ravel(opthelper.rho)
    rho_real = np.zeros((2 * rho_flat.shape[0]))
    rho_real[: rho_flat.shape[0]] = rho_flat.real
    rho_real[rho_flat.shape[0] :] = rho_flat.imag
    A_1 = d_phi_1 @ opthelper.b_matrix - np.linalg.multi_dot(
        [__u, __u.conj().T, d_phi_1, opthelper.b_matrix]
    )

    A_2 = d_phi_2 @ opthelper.b_matrix - np.linalg.multi_dot(
        [__u, __u.conj().T, d_phi_2, opthelper.b_matrix]
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
    __JAC_OUT_REAL = __compute_dmd_jac(alphas_in, time, data, opthelper)

    GRAD_REAL = JAC_REAL.T @ rho_real
    __GRAD_REAL = __JAC_OUT_REAL.T @ rho_real
    GRAD_IMAG = JAC_IMAG.conj().T @ rho_flat

    assert np.linalg.norm(JAC_REAL - __JAC_OUT_REAL) < 1e-12
    assert np.linalg.norm(GRAD_REAL - __GRAD_REAL) < 1e-12

    __imag2real = np.zeros_like(GRAD_REAL)
    __imag2real[: __imag2real.shape[-1] // 2] = GRAD_IMAG.real
    __imag2real[__imag2real.shape[-1] // 2 :] = GRAD_IMAG.imag

    __rec_grad = np.zeros_like(GRAD_IMAG)
    __rec_grad.real = GRAD_REAL[: GRAD_REAL.shape[-1] // 2]
    __rec_grad.imag = GRAD_REAL[GRAD_REAL.shape[-1] // 2 :]

    # funny numerical errors leads to
    # np.array_equal(GRAD_IMAG, __rec_grad) to fail
    assert np.linalg.norm(GRAD_IMAG - __rec_grad) < 1e-9

    __rec_grad = np.zeros_like(GRAD_IMAG)
    __rec_grad.real = __GRAD_REAL[: __GRAD_REAL.shape[-1] // 2]
    __rec_grad.imag = __GRAD_REAL[__GRAD_REAL.shape[-1] // 2 :]

    # funny numerical errors leads to
    # np.array_equal(GRAD_IMAG, __rec_grad) to fail
    assert np.linalg.norm(GRAD_IMAG - __rec_grad) < 1e-9


def test_varprodmd_any():
    """
    Test Variable Projection function for DMD (at any timestep).
    """
    time = np.linspace(0, 4 * np.pi, 100)
    x_loc = np.linspace(-10, 10, 1024)
    __x, __time = np.meshgrid(x_loc, time)

    z = signal(__x, __time).T

    __idx = select_best_samples_fast(z, 0.6)

    __z_sub = z[:, __idx]
    __t_sub = time[__idx]

    with pytest.raises(ValueError):
        compute_varprodmd_any(
            __z_sub[:, 0],
            __t_sub,
            OPT_DEF_ARGS,
            rank=0.0,
        )

    with pytest.raises(ValueError):
        compute_varprodmd_any(
            __z_sub, __t_sub.reshape((-1, 1)), OPT_DEF_ARGS, rank=0.0
        )

    phi, lambdas, eigenf, _, _ = compute_varprodmd_any(
        __z_sub, __t_sub, OPT_DEF_ARGS, rank=0.0
    )
    __pred = optdmd_predict(phi, lambdas, eigenf, time)
    __diff = np.abs(__pred - z)
    __mae_0 = np.sum(np.sum(__diff, axis=0), axis=-1) / z.shape[0] / z.shape[-1]

    assert __mae_0 < 1.0

    phi, lambdas, eigenf, _, _ = compute_varprodmd_any(
        __z_sub, __t_sub, OPT_DEF_ARGS, rank=0.0, use_proj=False
    )
    __pred = optdmd_predict(phi, lambdas, eigenf, time)
    __diff = np.abs(__pred - z)
    __mae_0 = np.sum(np.sum(__diff, axis=0), axis=-1) / z.shape[0] / z.shape[-1]

    assert __mae_0 < 1.0


def test_varprodmd_class():
    """
    Test VarProDMD class.
    """
    time = np.linspace(0, 4 * np.pi, 100)
    x_loc = np.linspace(-10, 10, 1024)
    __x, __time = np.meshgrid(x_loc, time)

    z = signal(__x, __time).T
    dmd = VarProDMD(0, False, False, 0)

    with pytest.raises(ValueError):
        __ = dmd.forecast(time)

    with pytest.raises(ValueError):
        __ = dmd.ssr

    with pytest.raises(ValueError):
        __ = dmd.selected_samples

    with pytest.raises(ValueError):
        __ = dmd.opt_stats

    dmd.fit(z, time)
    assert dmd.fitted
    assert dmd.eigs.size > 0
    assert len(dmd.modes.shape) == 2
    assert dmd.amplitudes.size > 0
    assert len(dmd.dynamics.shape) == 2
    assert dmd.amplitudes.size == dmd.frequency.size
    assert dmd.growth_rate.size == dmd.amplitudes.size
    assert dmd.eigs.size == dmd.amplitudes.size

    __pred = dmd.forecast(time)

    __diff = np.abs(__pred - z)
    __mae = np.sum(np.sum(__diff, axis=0), axis=-1) / z.shape[0] / z.shape[-1]

    assert __mae < 1
    assert dmd.ssr < 1e-3

    dmd = VarProDMD(0, False, "unkown_sort", 0.8)

    with pytest.raises(ValueError):
        dmd.fit(z, time)

    sort_args = ["auto", "real", "imag", "abs", True, False]

    for arg in sort_args:
        dmd = VarProDMD(0, False, arg, 0.6)
        dmd.fit(z, time)
        __pred = dmd.forecast(time)
        __diff = np.abs(__pred - z)
        __mae = (
            np.sum(np.sum(__diff, axis=0), axis=-1) / z.shape[0] / z.shape[-1]
        )
        assert dmd.selected_samples.size == int((1 - 0.6) * 100)
        assert __mae < 1.0
