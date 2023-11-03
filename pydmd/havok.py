"""
Module for the Hankel alternative view of Koopman (HAVOK) analysis.

References:
- S. L. Brunton, B. W. Brunton, J. L. Proctor, E. Kaiser, and J. N. Kutz,
Chaos as an intermittently forced linear system, Nature Communications, 8
(2017), pp. 1-9.
- S. M. Hirsh, S. M. Ichinaga, S. L. Brunton, J. N. Kutz, and B. W. Brunton,
Structured time-delay models for dynamical systems with connections to
frenet-serret frame, Proceedings of the Royal Society A, 477
(2021). art. 20210097.
"""

import warnings
import numpy as np
from scipy.signal import lsim, StateSpace

from .bopdmd import BOPDMD
from .dmdbase import DMDBase
from .utils import compute_svd, differentiate


class HAVOK:
    """
    Hankel alternative view of Koopman (HAVOK) analysis.

    :param svd_rank: the rank for the truncation; if 0, the method computes the
        optimal rank and uses it for the truncation; if positive integer, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute a truncation.
    :type svd_rank: int or float
    :param delays: the number of consecutive time-shifted copies of the
        data to use when building Hankel matrices. Note that if examining an
        n-dimensional data set, this means that the resulting Hankel matrix
        will contain n * `delays` rows.
    :type delays: int
    :param lag: the number of time steps between each time-shifted copy of
        data in the Hankel matrix.
    :type lag: int
    :param num_chaos: the number of forcing terms to use in the HAVOK model.
    :type num_chaos: int
    :param structured: whether to perform standard HAVOK or structured HAVOK
        (sHAVOK). If `True`, sHAVOK is performed, otherwise HAVOK is performed.
        Note that sHAVOK cannot be performed with a `BOPDMD` model.
    :type structured: bool
    :param lstsq: method used for computing the HAVOK operator if a DMD method
        is not provided. If True, least-squares is used, otherwise the pseudo-
        inverse is used. This parameter is ignored if `dmd` is provided.
    :type lstsq: bool
    :param dmd: DMD instance used to compute the HAVOK operator. If `None`,
        least-squares or the pseudo-inverse is used depending on `lstsq`.
    :type dmd: DMDBase
    """

    def __init__(
        self,
        svd_rank=0,
        delays=10,
        lag=1,
        num_chaos=1,
        structured=False,
        lstsq=True,
        dmd=None,
    ):
        self._svd_rank = svd_rank
        self._delays = delays
        self._lag = lag
        self._num_chaos = num_chaos
        self._structured = structured
        self._lstsq = lstsq
        self._dmd = dmd

        # Keep track of the original data and Hankel matrix.
        self._snapshots = None
        self._ho_snapshots = None
        self._time = None

        # Keep track of SVD information.
        self._singular_vecs = None
        self._singular_vals = None
        self._delay_embeddings = None

        # Keep track of the full HAVOK operator.
        self._havok_operator = None
        self._eigenvalues = None

    @property
    def snapshots(self):
        """
        Get the input data (time-series or space-flattened).

        :return: the matrix that contains the original input data.
        :rtype: numpy.ndarray
        """
        if self._snapshots is None:
            raise ValueError("You need to call fit().")
        return self._snapshots

    @property
    def ho_snapshots(self):
        """
        Get the time-delay data matrix.

        :return: the matrix that contains the time-delayed data.
        :rtype: numpy.ndarray
        """
        if self._ho_snapshots is None:
            raise ValueError("You need to call fit().")
        return self._ho_snapshots

    @property
    def linear_dynamics(self):
        """
        Get the HAVOK embeddings that are governed by linear dynamics.
        Coordinates are stored as columns of the returned matrix.

        :return: matrix containing the linear HAVOK embeddings.
        :rtype: numpy.ndarray
        """
        if self._delay_embeddings is None:
            raise ValueError("You need to call fit().")
        return self._delay_embeddings[:, : -self._num_chaos]

    @property
    def forcing(self):
        """
        Get the HAVOK embeddings that force the linear dynamics.

        :return: matrix containing the chaotic forcing terms.
        :rtype: numpy.ndarray
        """
        if self._delay_embeddings is None:
            raise ValueError("You need to call fit().")
        return self._delay_embeddings[:, -self._num_chaos :]

    @property
    def A(self):
        """
        Get the matrix A in the HAVOK relationship dv/dt = Av + Bu, where v
        denotes the linear HAVOK embeddings and u denotes the forcing terms.

        :return: linear dynamics matrix A.
        :rtype: numpy.ndarray
        """
        if self._havok_operator is None:
            raise ValueError("You need to call fit().")
        return self._havok_operator[: -self._num_chaos, : -self._num_chaos]

    @property
    def B(self):
        """
        Get the matrix B in the HAVOK relationship dv/dt = Av + Bu, where v
        denotes the linear HAVOK embeddings and u denotes the forcing terms.

        :return: forcing dynamics matrix B.
        :rtype: numpy.ndarray
        """
        if self._havok_operator is None:
            raise ValueError("You need to call fit().")
        return self._havok_operator[: -self._num_chaos, -self._num_chaos :]

    @property
    def r(self):
        """
        Get the number of HAVOK embeddings utilized by the HAVOK model.

        :return: rank of the HAVOK model.
        :rtype: int
        """
        if self._havok_operator is None:
            raise ValueError("You need to call fit().")
        return len(self._havok_operator)

    @property
    def eigs(self):
        """
        Get the eigenvalues of the linear HAVOK operator.
        """
        if self._eigenvalues is None:
            raise ValueError("You need to call fit().")
        return self._eigenvalues

    def fit(self, X, t):
        """
        Perform HAVOK analysis.

        :param X:
        :type X:
        :param t:
        :type t:
        """

        # Confirm that delays, lag, and num_chaos are positive integers.
        for x in [self._delays, self._lag, self._num_chaos]:
            if not isinstance(x, int) or x < 1:
                msg = "delays, lag, and num_chaos must be positive integers."
                raise ValueError("num_chaos must be a positive integer.")

        # Confirm that dmd is a child of DMDBase, if provided.
        if self._dmd is not None and not isinstance(self._dmd, DMDBase):
            raise ValueError("dmd must be None or a pydmd.DMDBase object.")

        # Confirm that the input data is a 1D time-series or a 2D data matrix.
        X = np.squeeze(np.array(X))
        if X.ndim > 2:
            msg = "Please ensure that input data is a 1D or 2D array."
            raise ValueError(msg)
        if X.ndim == 1:
            X = X[None]
        n_samples = X.shape[-1]

        # Check that the input data contains enough observations.
        if n_samples < self._delays * self._lag:
            msg = (
                "Not enough snapshots provided for {} delays and a lag of {}."
                "Please provide at least {} snapshots."
            )
            raise ValueError(
                msg.format(self._delays, self._lag, self._delays * self._lag)
            )

        # Check the input time information and set the time vector.
        if isinstance(t, (int, float)) and t > 0.0:
            time = np.arange(n_samples) * t
        elif isinstance(t, (list, np.ndarray)):
            time = np.squeeze(np.array(t))

            # Throw error if the time vector is not 1D or the correct length.
            if time.ndim != 1 or len(time) != n_samples:
                msg = "Please provide a 1D array of {} time values."
                raise ValueError(msg.format(n_samples))

            # Generate warning if the times are not uniformly-spaced.
            if not np.allclose(
                time[1:] - time[:-1],
                (time[1] - time[0]) * np.ones(len(time) - 1),
            ):
                msg = (
                    "Input snapshots are unevenly-spaced in time. "
                    "Note that unexpected results may occur because of this. "
                )
                warnings.warn(msg)
        else:
            msg = (
                "t must either be a single positive time step "
                "or a 1D array of {} time values."
            )
            raise ValueError(msg.format(n_samples))

        # Set the time step - this is ignored if using BOP-DMD.
        dt = time[1] - time[0]

        # We have enough data - compute the Hankel matrix.
        hankel_matrix = self._hankel(X)

        # Perform structured HAVOK (sHAVOK).
        if self._structured:
            U, s, V = compute_svd(hankel_matrix[:, 1:-1], self._svd_rank)
            V1 = compute_svd(hankel_matrix[:, :-2], len(s))
            V2 = compute_svd(hankel_matrix[:, 2:], len(s))
            V_dot = (V2 - V1) / (2 * dt)

        # Perform standard HAVOK.
        else:
            U, s, V = compute_svd(hankel_matrix, self._svd_rank)
            V_dot = differentiate(V.T, dt).T

        # Generate an error if too few HAVOK embeddings are being used.
        if len(s) < self._num_chaos + 1:
            msg = (
                "HAVOK model is attempting to use r = {} embeddings when r "
                "should be at least {}. Try increasing the number of delays "
                "and/or providing a positive integer argument for svd_rank."
            )
            raise ValueError(msg.format(len(s), self._num_chaos + 1))

        # Use lstsq or pinv to compute the HAVOK operator.
        if self._dmd is None:
            if self._lstsq:
                havok_operator = np.linalg.lstsq(V, V_dot, rcond=None)[0].T
            else:
                havok_operator = np.linalg.pinv(V).dot(V_dot).T

        # Use the provided DMDBase object to compute the operator.
        else:
            if isinstance(self._dmd, BOPDMD):
                self._dmd.fit(V.T, time)
            else:
                self._dmd.fit(V.T, V_dot.T)

            # Compute the full system matrix.
            havok_operator = np.linalg.multi_dot(
                [
                    self._dmd.modes,
                    np.diag(self._dmd.eigs),
                    np.linalg.pinv(self._dmd.modes),
                ]
            )

        # Set the input data information.
        self._snapshots = X
        self._ho_snapshots = hankel_matrix
        self._time = time

        # Set the SVD information.
        self._singular_vecs = U
        self._singular_vals = s
        self._delay_embeddings = V

        # Save the full HAVOK operator.
        self._havok_operator = havok_operator
        self._eigenvalues = np.linalg.eig(
            havok_operator[: -self._num_chaos, : -self._num_chaos]
        )

        return self

    @property
    def reconstructed_embeddings(self):
        """
        Get the reconstructed data.
        """
        # Build a system with the following form:
        #   dx/dt = Ax + Bu
        #   y = Cx + Du
        C = np.eye(len(self.A))
        D = 0.0 * self.B
        havok_system = StateSpace(self.A, self.B, C, D)

        # Reconstruct the linear dynamics using the HAVOK system.
        linear_recon = lsim(
            havok_system,
            U=self.forcing,
            T=self._time[: len(self.forcing)],
            X0=self.linear_dynamics[0],
        )[1]

        return linear_recon

    @property
    def reconstructed_data(self):
        """
        Get the reconstructed data.

        :return: the matrix that contains the reconstructed snapshots.
        :rtype: numpy.ndarray
        """
        # Reconstruct the linear dynamics using the HAVOK system.
        linear_recon = self.reconstructed_embeddings

        # Send the reconstructions back to the space of the original data.
        hankel_matrix_recon = np.linalg.multi_dot(
            [
                self._singular_vecs[:, : -self._num_chaos],
                np.diag(self._singular_vals[: -self._num_chaos]),
                linear_recon.conj().T,
            ]
        )

        return self._dehankel(hankel_matrix_recon)

    def _hankel(self, X):
        """
        Given a data matrix X as a 2D numpy.ndarray, uses the `_delays`
        and `_lag` attributes to return the data as a Hankel matrix.
        """
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError("Please ensure that input data is a 2D array.")
        n, m = X.shape
        num_cols = m - ((self._delays - 1) * self._lag)
        H = np.empty((n * self._delays, num_cols))
        for i in range(self._delays):
            H[i * n : (i + 1) * n] = X[
                :, i * self._lag : i * self._lag + num_cols
            ]
        return H

    def _dehankel(self, H):
        """
        Given a Hankel matrix H as a 2D numpy.ndarray, uses the `_delays`
        and `_lag` attributes to unravel the data in the Hankel matrix.
        """
        if not isinstance(H, np.ndarray) or H.ndim != 2:
            raise ValueError("Please ensure that input data is a 2D array.")
        n = int(H.shape[0] / self._delays)
        X = np.hstack([H[:n], H[n:, -1].reshape(n, -1, order="F")])
        return X
