"""
Derived module from hankeldmd.py for HAVOK.

Reference:
- S. L. Brunton, B. W. Brunton, J. L. Proctor,Eurika Kaiser, and J. N. Kutz.
Chaos as an intermittently forced linear system.
Nature Communications, 8, 2017.
"""

import numpy as np
from scipy import signal

from .hankeldmd import HankelDMD
from .snapshots import Snapshots
from .utils import compute_svd
from pydmd.linalg import no_torch, build_linalg_module


class HAVOK(HankelDMD):
    """
    Hankel alternative view of Koopman (HAVOK) analysis
    """

    def __init__(
        self,
        svd_rank=0,
        tlsq_rank=0,
        exact=False,
        opt=False,
        rescale_mode=None,
        forward_backward=False,
        sorted_eigs=False,
        tikhonov_regularization=None,
        d=10,
    ):
        super().__init__(
            svd_rank=svd_rank,
            tlsq_rank=tlsq_rank,
            exact=exact,
            opt=opt,
            rescale_mode=rescale_mode,
            forward_backward=forward_backward,
            sorted_eigs=sorted_eigs,
            tikhonov_regularization=tikhonov_regularization,
            d=d,
        )

        self._svd_rank = svd_rank

        self._embeddings = None
        self._A = None
        self._B = None
        self._r = None

    @property
    def linear_embeddings(self):
        """
        Get the time-delay embeddings of the data that are governed by linear
        dynamics. Emeddings are stored as columns of the returned matrix.

        :return: matrix containing the linear time-delay embeddings.
        :rtype: numpy.ndarray
        """
        if self._embeddings is None:
            raise RuntimeError("fit() not called")
        return self._embeddings[:, :-1]

    @property
    def forcing_input(self):
        """
        Get the time-delay embedding that forces the linear embeddings.

        :return: array containing the chaotic forcing term.
        :rtype: numpy.ndarray
        """
        if self._embeddings is None:
            raise RuntimeError("fit() not called")
        return self._embeddings[:, -1]

    @property
    def A(self):
        """
        Get the matrix A in the relationship dV/dt = AV + Bu, where V denotes
        the linear embeddings and u denotes the forcing input.

        :return: linear dynamics matrix A.
        :rtype: numpy.ndarray
        """
        if self._A is None:
            raise RuntimeError("fit() not called")
        return self._A

    @property
    def B(self):
        """
        Get the vector B in the relationship dV/dt = AV + Bu, where V denotes
        the linear embeddings and u denotes the forcing input.

        :return: forcing dynamics vector B.
        :rtype: numpy.ndarray
        """
        if self._B is None:
            raise RuntimeError("fit() not called")
        return self._B

    @property
    def r(self):
        """
        Number of time-delay embeddings utilized by the HAVOK model.
        Note: d is the same as svd_rank if svd_rank is a positive integer.
        """
        if self._r is None:
            raise RuntimeError("fit() not called")
        return self._r

    def _dehankel(self, X):
        """
        Given a hankel matrix X as a 2-D numpy.ndarray,
        returns the data as a 1-D time-series.
        """
        return np.concatenate((X[0, :], X[1:, -1]))

    def reconstructions_of_timeindex(self, timeindex=None):
        raise NotImplementedError("This function is not compatible with HAVOK.")

    @property
    def reconstructed_data(self):
        # Build a continuous-time system of the following form, where
        # x, u, and y represent the states, inputs, and outputs respectively.
        #
        # dx/dt = Ax + Bu
        # y = Cx + Du
        #
        # Note that we define the output y to be the state x.
        havok_system = signal.StateSpace(
            self.A, self.B, np.eye(self._r - 1), 0 * self.B
        )

        # Using the system defined above, compute the output of the system,
        # where the system input is the forcing input, the times of forcing
        # are given by self.dmd_timesteps, and the initial state values are
        # given by the initial values of the linear time-delay embeddings.
        # This yields a reconstruction of V from the SVD of the hankel matrix.
        reconstructed_v = signal.lsim(
            havok_system,
            U=self.forcing_input,
            T=self.dmd_timesteps[: len(self.forcing_input)],
            X0=self.linear_embeddings[0, :],
        )[1]

        # Compute a reconstruction of the original data x by first recomputing
        # the hankel matrix with the reconstructed V matrix and then recovering
        # a 1-D time-series from the computed hankel matrix.
        reconstructed_hankel_matrix = np.linalg.multi_dot(
            [
                self._svd_modes[:, :-1],
                np.diag(self._svd_amps[:-1]),
                reconstructed_v.conj().T,
            ]
        )

        return self._dehankel(reconstructed_hankel_matrix)

    def fit(self, x, dt):
        """
        Perform HAVOK analysis on 1-D time-series data x given the size of
        the time step dt separating the observations in x.
        """
        self._reset()
        no_torch(x)

        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("Input data must be a 1-D time series.")
        self._snapshots_holder = Snapshots(x[None])

        # Get number of data points
        n_samples = self.snapshots.shape[-1]

        # Check that the input time-series contains enough observations
        if n_samples < self._d:
            msg = """The number of snapshots provided is not enough for d={}.
Expected at least d."""
            raise ValueError(msg.format(self._d))

        # Compute hankel matrix for the input data
        linalg_module = build_linalg_module(self.snapshots)
        hankel_matrix = linalg_module.pseudo_hankel_matrix(
            self.snapshots, self._d
        )

        # Take SVD of the hankel matrix
        # Save the resulting U, s, and V for future reconstructions
        self._svd_modes, self._svd_amps, self._embeddings = compute_svd(
            hankel_matrix, self._svd_rank
        )

        # Record the number of time-delay embeddings being used.
        # Throw an error if less than 2 embeddings are being used.
        self._r = len(self._svd_amps)
        if self._r < 2:
            msg = f"""HAVOK model is attempting to use r = {self._r} embeddings
when r should be at least 2. Try increasing the number of delays d and/or 
providing a positive integer argument for svd_rank."""
            raise RuntimeError(msg)

        # Perform DMD on the time-delay embeddings
        self._sub_dmd.fit(self._embeddings.T)

        # Compute the full (discrete-time) system matrix
        w, e = self._sub_dmd.modes, self._sub_dmd.eigs
        regression_model_discrete = np.linalg.multi_dot(
            [w, np.diag(e), np.linalg.pinv(w)]
        )

        # Roughly approximate the continuous-time system matrix
        regression_model_continuous = (
            regression_model_discrete - np.eye(self._r)
        ) / dt

        # Save A matrix and B vector
        self._A = regression_model_continuous[:-1, :-1]
        self._B = regression_model_continuous[:-1, -1, None]

        # Set timesteps
        self._set_initial_time_dictionary(
            {"t0": 0, "tend": (n_samples - 1) * dt, "dt": dt}
        )

        return self
