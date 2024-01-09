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
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from scipy.signal import lsim, StateSpace
from scipy.stats import norm

from .bopdmd import BOPDMD
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
        data in the Hankel matrix. This means that each row of the Hankel
        matrix will be separated by a time-step of dt * `lag`.
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
        self._r = None

    @property
    def delays(self):
        """
        Get the number of delays used when building Hankel matrices.

        :return: the number of Hankel matrix delays.
        :rtype: int
        """
        return self._delays

    @property
    def lag(self):
        """
        Get the lag used when building Hankel matrices.

        :return: the number of time-steps used for Hankel matrix lag.
        :rtype: int
        """
        return self._lag

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
        Get the time-delay data matrix (i.e. the Hankel matrix).

        :return: the matrix that contains the time-delayed data.
        :rtype: numpy.ndarray
        """
        if self._ho_snapshots is None:
            raise ValueError("You need to call fit().")
        return self._ho_snapshots

    @property
    def time(self):
        """
        Get the times of the input data.

        :return: the vector that contains the times of the input data.
        :rtype: numpy.ndarray
        """
        if self._time is None:
            raise ValueError("You need to call fit().")
        return self._time

    @property
    def modes(self):
        """
        Get the U matrix from the SVD of the Hankel matrix. Note that the
        columns of this matrix are referred to as the eigen-time-delay modes.

        :return: matrix containing the eigen-time-delay modes.
        :rtype: numpy.ndarray
        """
        if self._singular_vecs is None:
            raise ValueError("You need to call fit().")
        return self._singular_vecs

    @property
    def singular_vals(self):
        """
        Get the singular value spectrum of the Hankel matrix.

        :return: the singular values of the Hankel matrix.
        :rtype: numpy.ndarray
        """
        if self._singular_vals is None:
            raise ValueError("You need to call fit().")
        return self._singular_vals

    @property
    def delay_embeddings(self):
        """
        Get all of the HAVOK embeddings (linear dynamics and forcing).
        Coordinates are stored as columns of the returned matrix.
        Note that this is the V matrix from the SVD of the Hankel matrix.

        :return: matrix containing all of the HAVOK embeddings.
        :rtype: numpy.ndarray
        """
        if self._delay_embeddings is None:
            raise ValueError("You need to call fit().")
        return self._delay_embeddings

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
        Coordinates are stored as columns of the returned matrix.

        :return: matrix containing the chaotic forcing terms.
        :rtype: numpy.ndarray
        """
        if self._delay_embeddings is None:
            raise ValueError("You need to call fit().")
        return self._delay_embeddings[:, -self._num_chaos :]

    @property
    def operator(self):
        """
        Get the full HAVOK regression model,
        which contains A, B, and the bad fit.

        :return: the full HAVOK regression model.
        :rtype: numpy.ndarray
        """
        if self._havok_operator is None:
            raise ValueError("You need to call fit().")
        return self._havok_operator

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
    def eigs(self):
        """
        Get the eigenvalues of the linear HAVOK operator A.

        :return: the eigenvalues of the operator A.
        :rtype: numpy.ndarray
        """
        if self._eigenvalues is None:
            raise ValueError("You need to call fit().")
        return self._eigenvalues

    @property
    def r(self):
        """
        Get the number of HAVOK embeddings utilized by the HAVOK model.
        Note that this is essentially the integer rank truncation used.

        :return: rank of the HAVOK model.
        :rtype: int
        """
        if self._r is None:
            raise ValueError("You need to call fit().")
        return self._r

    def fit(self, X, t):
        """
        Perform the HAVOK analysis.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        :param t: the input time vector or uniform time-step between snapshots.
        :type t: {numpy.ndarray, iterable} or {int, float}
        """

        # Confirm that delays, lag, and num_chaos are positive integers.
        for x in [self._delays, self._lag, self._num_chaos]:
            if not isinstance(x, int) or x < 1:
                raise ValueError(
                    "delays, lag, and num_chaos must be positive integers."
                )

        # Confirm that the input data is 1-D or 2-D.
        X = np.squeeze(np.array(X))
        if X.ndim > 2:
            raise ValueError("Input data must be a 1-D or 2-D array.")
        if X.ndim == 1:
            X = X[None]
        n_samples = X.shape[-1]

        # Check that the input data contains enough observations.
        if n_samples < self._delays * self._lag:
            raise ValueError(
                "Not enough snapshots provided for "
                f"{self._delays} delays and lag {self._lag}. Please "
                f"provide at least {self._delays * self._lag} snapshots."
            )

        # Check the input time information and set the time vector.
        if isinstance(t, (int, float)):
            time = np.arange(n_samples) * t
        else:
            time = np.squeeze(np.array(t))

            # Throw error if the time vector is not 1D or the correct length.
            if time.ndim != 1 or len(time) != n_samples:
                raise ValueError(
                    f"Please provide a 1-D array of {n_samples} time values."
                )

            # Generate warning if the times are not uniformly-spaced.
            if not np.allclose(time[1:] - time[:-1], time[1] - time[0]):
                warnings.warn(
                    "Input snapshots are unevenly-spaced in time. "
                    "Unexpected results may occur because of this."
                )

        # Set the time step - this is ignored if using BOP-DMD.
        dt = time[1] - time[0]

        # We have enough data - compute the Hankel matrix.
        hankel_matrix = self._hankel(X)

        # Perform structured HAVOK (sHAVOK).
        if self._structured:
            U, s, V = compute_svd(hankel_matrix[:, 1:-1], self._svd_rank)
            self._r = len(s)
            V1 = compute_svd(hankel_matrix[:, :-2], self._r)[-1]
            V2 = compute_svd(hankel_matrix[:, 2:], self._r)[-1]
            V_dot = (V2 - V1) / (2 * dt)

        # Perform standard HAVOK.
        else:
            U, s, V = compute_svd(hankel_matrix, self._svd_rank)
            self._r = len(s)
            V_dot = differentiate(V.T, dt).T

        # Generate an error if too few HAVOK embeddings are being used.
        if self._r < self._num_chaos + 1:
            raise ValueError(
                f"HAVOK is attempting to use r = {self._r} embeddings "
                f"when r should be at least {self._num_chaos + 1}. "
                "Try increasing the number of delays or providing "
                "a positive integer argument for svd_rank."
            )

        # Use lstsq or pinv to compute the HAVOK operator.
        if self._dmd is None:
            if self._lstsq:
                havok_operator = np.linalg.lstsq(V, V_dot, rcond=None)[0].T
            else:
                havok_operator = np.linalg.pinv(V).dot(V_dot).T

        # Use the provided DMDBase object to compute the operator.
        else:
            if isinstance(self._dmd, BOPDMD):
                self._dmd.fit(V.T, time[: len(V)])

                if self._structured:
                    warnings.warn(
                        "Structured HAVOK cannot be performed with BOP-DMD. "
                        "Performing normal HAVOK instead..."
                    )
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
        )[0]

        return self

    def predict(self, forcing, time, V0):
        """
        Use a custom forcing input to make system predictions.

        :param forcing: (m, `num_chaos`) array of forcing inputs.
        :type forcing: numpy.ndarray
        :param time: (m,) array that contains the times that correspond with
            the provided forcing inputs. These will also be the times at which
            system predictions are computed.
        :type time: numpy.ndarray
        :param V0: (`r` - `num_chaos`,) array that contains the initial
            condition of the linear dynamics. This array should contain the
            linear dynamics evaluated at the first time in the `time` array.
        :type V0: numpy.ndarray
        :return: system predictions evaluated at the times in `time`.
        :rtype: numpy.ndarray
        """
        return self._embeddings_to_original(
            self._compute_embeddings(forcing, time, V0)
        )

    @property
    def reconstructed_embeddings(self):
        """
        Get the reconstructed time-delay embeddings.

        :return: the matrix that contains the reconstructed embeddings.
        :rtype: numpy.ndarray
        """
        return self._compute_embeddings(
            self.forcing,
            self._time[: len(self.forcing)],
            self.linear_dynamics[0],
        )

    @property
    def reconstructed_data(self):
        """
        Get the reconstructed data.

        :return: the matrix that contains the reconstructed snapshots.
        :rtype: numpy.ndarray
        """
        return np.squeeze(
            self._embeddings_to_original(self.reconstructed_embeddings)
        )

    @staticmethod
    def compute_threshold(forcing, p=0.01, bins=50, plot=False, ylim=None):
        """
        Use the distribution of forcing terms to determine a threshold at which
        the absolute value of the forcing is large enough to be considered
        "active". This method uses a histogram of the forcing signal values
        and a forcing event probability in order to estimate this threshold.

        :param forcing: (m,) array of forcing inputs to be thresholded.
        :type forcing: numpy.ndarray
        :param p: desired approximate probability that a forcing event occurs.
            Note that `p` must be a float between 0.0 and 1.0, and that smaller
            values of `p` will result in larger threshold values. If `p` lies
            outside of this range, the forcing term histogram's intersection
            with a fitted Gaussian distribution is used as the threshold.
        :type p: int or float
        :param bins: `bins` input to the `numpy.histogram` function.
        :type bins: int or sequence of scalars or str
        :param plot: whether or not to plot the computed histogram of forcing
            values and the computed threshold. A Gaussian distribution fitted
            to the computed histogram is also plotted if `plot=True`.
        :type plot: bool
        :param ylim: set the y-axis limit of the histogram plot.
        :type ylim: iterable
        :return: active threshold for the absolute value of the forcing terms.
        :rtype: float
        """
        # Compute histogram of the forcing values.
        hy, hx = np.histogram(forcing, bins=bins, density=True)
        hx = 0.5 * (hx[:-1] + hx[1:])  # get bin centers
        hy /= hy.sum()

        # Fit a Gaussian to the forcing values.
        mu, std = norm.fit(forcing)
        gauss = norm.pdf(hx, mu, std)
        gauss /= np.sum(gauss)

        if p <= 0.0 or p >= 1.0:
            # Use Gaussian intersection.
            a = gauss - hy
            ind_sign_change = np.where(np.sign(a[:-1]) - np.sign(a[1:]) != 0)[0]
            thres_1 = np.abs(hx[ind_sign_change])
            thres_2 = np.abs(hx[ind_sign_change + 1])
            threshold_candidates = np.sort(0.5 * (thres_1 + thres_2))
            threshold = threshold_candidates[p]
        else:
            # Use probability p of switching.
            ind1 = np.where(np.cumsum(hy) > 0.5 * p)[0][0]
            ind2 = np.where(1 - np.cumsum(hy) < 0.5 * p)[0][0]
            threshold = 0.5 * (abs(hx[ind1]) + abs(hx[ind2]))

        if plot:
            # Plot the histogram, fitted Gaussian, and the computed threshold.
            plt.figure(figsize=(5, 4))
            plt.plot(hx, hy, c="tab:red", label="Forcing", lw=2)
            plt.plot(hx, gauss, c="k", label="Gaussian", lw=2, ls="--")
            plt.axvline(x=threshold, lw=2, label="Threshold")
            plt.axvline(x=-threshold, lw=2)
            plt.xlabel("vr")
            plt.ylabel("p", rotation=0)
            plt.semilogy()
            plt.legend()
            if ylim is not None:
                plt.ylim(ylim)
            plt.show()

        return threshold

    def plot_summary(
        self,
        num_plot=None,
        index_linear=(0, 1, 2),
        index_forcing=0,
        forcing_threshold=None,
        min_jump_dist=None,
        true_switch_indices=None,
        figsize=(20, 4),
        dpi=200,
        filename=None,
    ):
        """
        Generate a 5-element summarizing plot that contains the following:
        - the time-series used to apply HAVOK
        - the full linear operator, which contains A, B, and the bad fit
        - the first linear embedding term and the first forcing term
        - the HAVOK embeddings, along with active forcing times
        - the HAVOK reconstruction of the embeddings.

        :param num_plot: The number of time points to plot across all subplots.
            By default, all available data points are plotted.
        :type num_plot: int
        :param index_linear: Tuple of indices of the linear embeddings to be
            plotted. May contain either 2 or 3 valid indices. The final two
            subplots will be plotted in 2D or 3D depending on the number of
            indices provided. Also note that the first index in this tuple
            will determine the embedding plotted in the third subplot.
        :type index_linear: tuple
        :param index_forcing: Index of the forcing term to be plotted. Note
            that this index refers to indices of the forcing term itself rather
            than the full matrix of time-delay embeddings. Hence if 0, the
            first forcing term will be plotted, and so on.
        :type index_forcing: int
        :param forcing_threshold: Threshold value at which the absolute value
            of the forcing signal is considered large enough to be "active".
        :type forcing_threshold: float
        :param min_jump_dist: The minimum number of indices used to separate
            distinct forcing events. Decreasing this parameter will lead to
            many short forcing events, while increasing this parameter will
            lead to fewer longer forcing events.
        :type min_jump_dist: int
        :param true_switch_indices: Optional vector that contains the indices
            at which true chaotic bursting occurs. If provided, true bursting
            times are plotted on top of the forcing term.
        :type true_switch_indices: numpy.ndarray or iterable
        :param figsize: Tuple in inches defining the figure size.
        :type figsize: tuple(int, int)
        :param dpi: Figure resolution.
        :type dpi: int
        :param filename: If specified, the plot is saved at `filename`.
        :type filename: str
        """
        if self._havok_operator is None:
            raise ValueError("You need to call fit().")

        # Plot as many time points as possible.
        if num_plot is None:
            num_plot = len(self._delay_embeddings)

        # Compute a threshold based on the distribution of all forcing terms.
        if forcing_threshold is None:
            forcing_threshold = self.compute_threshold(
                self.forcing[:, index_forcing]
            )

        # Use the time step to estimate a reasonable jump.
        if min_jump_dist is None:
            min_jump_dist = int(0.5 / (self._time[1] - self._time[0]))

        # Get index slices at which the forcing is considered active.
        forcing = self.forcing[:num_plot, index_forcing]
        active_indices = np.arange(num_plot)[
            np.abs(forcing) > forcing_threshold
        ]
        active_slices = self._get_index_slices(active_indices, min_jump_dist)

        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = GridSpec(2, 5, figure=fig)
        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[:, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 2])
        if len(index_linear) == 3:
            ax5 = fig.add_subplot(gs[:, 3], projection="3d")
            ax6 = fig.add_subplot(gs[:, 4], projection="3d")
        else:
            ax5 = fig.add_subplot(gs[:, 3])
            ax6 = fig.add_subplot(gs[:, 4])

        # (1) plot the time-series data (first coordinate).
        ax1.set_title("Time series")
        ax1.plot(self._time[:num_plot], self._snapshots[0, :num_plot], c="k")
        ax1.set_xlabel("Time")

        # (2) plot the HAVOK operator.
        ax2.set_title("Regression model")
        vmax = np.abs(self._havok_operator).max()
        fig.colorbar(
            ax2.imshow(
                self._havok_operator,
                vmax=vmax,
                vmin=-vmax,
                cmap="PuOr",
            ),
            fraction=0.046,
            pad=0.04,
        )
        a = len(self._havok_operator) - self._num_chaos - 0.5
        ax2.plot([a, a], [-0.5, a], c="k", lw=1)
        ax2.axhline(y=a, c="k", lw=1)
        ax2.set_xticks([])
        ax2.set_yticks([])

        # (3) plot the linear HAVOK embeddings (first coordinate).
        ax3.set_title("Linear dynamics")
        ax3.plot(
            self._time[:num_plot],
            self.linear_dynamics[:num_plot, index_linear[0]],
            c="tab:blue",
        )
        ax3.set_xticks([])
        ax3.set_yticks([])

        # (4) plot the HAVOK forcing term with activation times.
        ax4.set_title("Forcing")
        ax4.plot(self._time[:num_plot], forcing, c="gray")
        ax4.set_xticks([])
        ax4.set_yticks([])

        for ind1, ind2 in active_slices:
            ax4.plot(self._time[ind1:ind2], forcing[ind1:ind2], c="r")

        if true_switch_indices is not None:
            # Remove indices that fall outside of the plotting range.
            outside_indices = np.where(true_switch_indices >= num_plot)[0]
            if len(outside_indices) > 0:
                true_switch_indices = true_switch_indices[: outside_indices[0]]
            ax4.plot(
                self._time[:num_plot][true_switch_indices],
                np.zeros(len(true_switch_indices)),
                "*",
                mec="k",
                mfc="y",
                ms=12,
            )

        # (5) plot the embedded attractor with activation.
        ax5.set_title("Embedded attractor")
        linear_dynamics = [
            self.linear_dynamics[:num_plot, i] for i in index_linear
        ]
        ax5.plot(*linear_dynamics, c="gray")
        for ind1, ind2 in active_slices:
            active_dynamics = [
                self.linear_dynamics[ind1:ind2, i] for i in index_linear
            ]
            ax5.plot(*active_dynamics, c="r")
        ax5.set_axis_off()

        # (6) plot the reconstructed attractor.
        ax6.set_title("Reconstructed attractor")
        reconstruction = [
            self.reconstructed_embeddings[:num_plot, i] for i in index_linear
        ]
        ax6.plot(*reconstruction, c="tab:blue")
        ax6.set_axis_off()
        plt.tight_layout(pad=0.1)

        # Save plot if filename is provided.
        if filename:
            plt.savefig(filename)
            plt.close(fig)
        else:
            plt.show()

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

    def _compute_embeddings(self, forcing, time, V0):
        """
        Helper function that uses the fitted HAVOK model to reconstruct the
        time-delay embeddings for a generic forcing term, set of times, and
        initial condition for the time-delay embeddings.
        """
        # Build a system with the following form:
        #   dx/dt = Ax + Bu
        #   y = Cx + Du
        C = np.eye(len(self.A))
        D = 0.0 * self.B
        havok_system = StateSpace(self.A, self.B, C, D)

        # Reconstruct the linear dynamics using the HAVOK system.
        embeddings = lsim(
            havok_system,
            U=forcing,
            T=time,
            X0=V0,
        )[1]

        return embeddings

    def _embeddings_to_original(self, V):
        """
        Helper function that uses SVD and Hankel parameter information stored
        in the HAVOK model to convert data in time-delay embedding space back
        to the space of the original input data.
        """
        U = self._singular_vecs[:, : V.shape[-1]]
        s = self._singular_vals[: V.shape[-1]]
        H = np.linalg.multi_dot([U, np.diag(s), V.conj().T])
        return self._dehankel(H)

    @staticmethod
    def _get_index_slices(x, min_jump_dist):
        """
        Helper function that, given an array x of indices at which to plot,
        computes and returns the beginning and ending index for each
        consecutive set of indices.

        :Example:
            >>> a = np.array([2, 3, 4, 5, 10, 11, 12, 25, 26, 28])
            >>> _get_index_slices(a, min_jump_dist=2)
            [(2, 5), (10, 12), (25, 28)]
        """
        # Get the locations within x where a significant jump occurs.
        jumps = x[1:] - x[:-1] > min_jump_dist
        jump_starts = np.insert(x[1:][jumps], 0, x[0])
        jump_ends = np.append(x[:-1][jumps], x[-1])
        index_slices = list(zip(jump_starts, jump_ends))

        return index_slices
