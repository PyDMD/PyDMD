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

from .dmdbase import DMDBase
from .bopdmd import BOPDMD
from .utils import compute_svd, pseudo_hankel_matrix, differentiate


class HAVOK:
    """
    Hankel alternative view of Koopman (HAVOK) analysis.

    :param delays:
    :param svd_rank:
    :param num_chaos:
    :param lstsq:
    :param dmd:
    :param structured:
    """

    def __init__(
        self,
        delays=10,
        svd_rank=0,
        num_chaos=1,
        lstsq=True,
        dmd=None,
        structured=False,
    ):
        self._delays = delays
        self._svd_rank = svd_rank
        self._num_chaos = num_chaos
        self._lstsq = lstsq
        self._dmd = dmd
        self._structured = structured

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
        return self._snapshots

    @property
    def ho_snapshots(self):
        """
        Get the time-delay data matrix.

        :return: the matrix that contains the time-delayed data.
        :rtype: numpy.ndarray
        """
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
        # Confirm that num_chaos is a positive integer.
        if not isinstance(self._num_chaos, int) or self._num_chaos < 1:
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
        if n_samples < self._delays:
            msg = (
                "The number of snapshots provided is not enough for {} delays."
                "Expected at least {} snapshots."
            )
            raise ValueError(msg.format(self._delays, self._delays))

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
            ) and not isinstance(self._dmd, BOPDMD):
                msg = (
                    "Input snapshots are unevenly-spaced in time. "
                    "Consider using pydmd.BOPDMD as your dmd method."
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
        hankel_matrix = pseudo_hankel_matrix(X, self._delays)

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
        linear_recon = self.reconstructed_embeddings()

        # Send the reconstructions back to the space of the original data.
        hankel_matrix_recon = np.linalg.multi_dot(
            [
                self._singular_vecs[:, :-self._num_chaos],
                np.diag(self._singular_vals[:-self._num_chaos]),
                linear_recon.conj().T,
            ]
        )

        return self._dehankel(hankel_matrix_recon)

    @staticmethod
    def _dehankel(X):
        """
        Given a hankel matrix X as a 2-D numpy.ndarray,
        returns the data as a 1-D time-series.
        """
        return np.concatenate((X[0, :], X[1:, -1]))
