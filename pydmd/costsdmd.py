import numpy as np
from pydmd.bopdmd import BOPDMD
import scipy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class CostsDMD:
    """Coherent Spatio-Temporal Scale Separation with DMD

    :param window_length: Length of the analysis window in number of time steps.
    :type window_length: int
    :param step_size: Number of time steps to slide each CSM-DMD window.
    :type step_size: int
    :param n_components: Number of independent frequency bands for this window length.
    :type n_components: int
    :param svd_rank: The rank of the BOPDMD fit.
    :type svd_rank: int
    :param global_svd: Flag indicating whether to find the proj_basis and initial
        values using the entire dataset instead of individually for each window.
        Generally using the global_svd speeds up the fitting process by not finding a
        new initial value for each window. Default is True.
    :type global_svd: bool
    :param initialize_artificially: Flag indicating whether to initialize the DMD using
        imaginary eigenvalues (i.e., the imaginary component of the cluster results from a
        previous iteration) through the `cluster_centroids` keyword. Default is False.
    :type initialize_artificially: bool
    :param pydmd_kwargs: Keyword arguments to pass onto the BOPDMD object.
    :type global_svd: dict
    :param cluster_centroids: Cluster centroids from a previous fitting iteration to
        use for the initial guess of the eigenvalues. Should only be the imaginary
        component.
    :type cluster_centroids: numpy array
    :param reset_alpha_init: Flag indicating whether the initial guess for the BOPDMD
        eigenvalues should be reset for each window. Resetting the initial value increases
        the computation time due to finding a new initial guess. Default is False.
    :type reset_alpha_init: bool
    :param force_even_eigs: Flag indicating whether an even svd_rank should be forced
        when not specifying the svd_rank directly (i.e., svd_rank=0). Default is True.
    :type global_svd: bool
    :param max_rank: Maximum svd_rank allowed when the svd_rank is found through rank
        truncation (i.e., svd_rank=0).
    :type max_rank: int
    :param use_kmean_freqs:
    :type use_kmean_freqs: bool
    :param init_alpha:
    :type init_alpha: numpy array
    """

    def __init__(
        self,
        window_length=None,
        step_size=None,
        svd_rank=None,
        global_svd=True,
        initialize_artificially=False,
        use_last_freq=False,
        use_kmean_freqs=False,
        init_alpha=None,
        pydmd_kwargs=None,
        cluster_centroids=None,
        reset_alpha_init=False,
        force_even_eigs=True,
        max_rank=None,
        n_components=None,
    ):
        self._n_components = n_components
        self._step_size = step_size
        self._window_length = window_length
        self._svd_rank = svd_rank
        self._global_svd = global_svd
        self._initialize_artificially = initialize_artificially
        self._use_last_freq = use_last_freq
        self._use_kmean_freqs = use_kmean_freqs
        self._init_alpha = init_alpha
        self._cluster_centroids = cluster_centroids
        self._reset_alpha_init = reset_alpha_init
        self._force_even_eigs = force_even_eigs
        self._max_rank = max_rank

        # Initialize variables that are defined in fitting.
        self._n_data_vars = None
        self._n_time_steps = None
        self._window_length = None
        self._n_slides = None
        self._time_array = None
        self._modes_array = None
        self._omega_array = None
        self._time_means_array = None
        self._amplitudes_array = None
        self._t_starts_array = None

        if pydmd_kwargs is None:
            self._pydmd_kwargs = {
                "eig_sort": "imag",
                "proj_basis": None,
                "use_proj": False,
            }
        else:
            self._pydmd_kwargs = pydmd_kwargs
            self._pydmd_kwargs["eig_sort"] = pydmd_kwargs.get(
                "eig_sort", "imag"
            )
            self._pydmd_kwargs["proj_basis"] = pydmd_kwargs.get(
                "proj_basis", None
            )
            self._pydmd_kwargs["use_proj"] = pydmd_kwargs.get("use_proj", False)

    def _compute_svd_rank(self, data, svd_rank=None):
        def omega(x):
            return 0.56 * x**3 - 0.95 * x**2 + 1.82 * x + 1.43

        U, s, _ = np.linalg.svd(data, full_matrices=False)

        if svd_rank == 0:
            beta = np.divide(*sorted(data.shape))
            tau = np.median(s) * omega(beta)
            svd_rank = np.sum(s > tau)
        elif 0 < svd_rank < 1:
            cumulative_energy = np.cumsum(s**2 / (s**2).sum())
            svd_rank = np.searchsorted(cumulative_energy, svd_rank) + 1
        elif svd_rank >= 1 and isinstance(svd_rank, (int, np.integer)):
            svd_rank = min(svd_rank, self._n_data_vars)
        else:
            svd_rank = self._n_data_vars

        return svd_rank

    def _build_proj_basis(self, data, svd_rank=None):
        self._svd_rank = self._compute_svd_rank(data, svd_rank=svd_rank)
        # Recover the first r modes of the global svd
        u, _, _ = scipy.linalg.svd(data, full_matrices=False)
        return u[:, : self._svd_rank]

    def _build_initizialization(self):
        """Method for making initial guess of DMD eigenvalues."""

        # If not initial values are provided return None by default.
        init_alpha = None
        # User provided initial eigenvalues.
        if self._initialize_artificially and self._init_alpha is not None:
            init_alpha = self._init_alpha
        # Initial eigenvalue guesses from kmeans clustering.
        elif (
            self._initialize_artificially
            and self._init_alpha is None
            and self._cluster_centroids is not None
        ):
            init_alpha = np.repeat(
                np.sqrt(self._cluster_centroids) * 1j,
                int(self._svd_rank / self._n_components),
            )
            init_alpha = init_alpha * np.tile(
                [1, -1], int(self._svd_rank / self._n_components)
            )
        # The user accidentally provided both methods of initializing the eigenvalues.
        elif (
            self._initialize_artificially
            and self._init_alpha is not None
            and self._cluster_centroids is not None
        ):
            raise ValueError(
                "Only one of `init_alpha` and `cluster_centroids` can be provided"
            )

        return init_alpha

    @staticmethod
    def build_windows(data, window_length, step_size, integer_windows=False):
        """Calculate how many times to slide the window across the data."""

        if integer_windows:
            n_split = np.floor(data.shape[1] / window_length).astype(int)
        else:
            n_split = data.shape[1] / window_length

        n_steps = int((window_length * n_split))

        # Number of sliding-window iterations
        n_slides = np.floor((n_steps - window_length) / step_size).astype(int)

        return n_slides + 1

    @staticmethod
    def calculate_lv_kern(window_length, corner_sharpness=None):
        """Calculate the kerning window for suppressing real eigenvalues."""

        # Higher = sharper corners
        if corner_sharpness is None:
            corner_sharpness = 16

        lv_kern = (
            np.tanh(
                corner_sharpness
                * np.arange(1, window_length + 1)
                / window_length
            )
            - np.tanh(
                corner_sharpness
                * (np.arange(1, window_length + 1) - window_length)
                / window_length
            )
            - 1
        )

        return lv_kern

    def _data_shape(self, data):
        n_time_steps = np.shape(data)[1]
        n_data_vars = np.shape(data)[0]
        return n_time_steps, n_data_vars

    @property
    def svd_rank(self):
        """
        :return: the rank used for the svd truncation.
        :rtype: int or float
        """
        return self._svd_rank

    @property
    def modes_array(self):
        if not hasattr(self, "_modes_array"):
            raise ValueError("You need to call fit before")
        return self._modes_array

    @property
    def amplitudes_array(self):
        if not hasattr(self, "_amplitudes_array"):
            raise ValueError("You need to call fit first.")
        return self._amplitudes_array

    @property
    def omega_array(self):
        if not hasattr(self, "_omega_array"):
            raise ValueError("You need to call fit first.")
        return self._omega_array

    @property
    def time_array(self):
        if not hasattr(self, "_time_array"):
            raise ValueError("You need to call fit first.")
        return self._time_array

    @property
    def window_means_array(self):
        if not hasattr(self, "_window_means_array"):
            raise ValueError("You need to call fit first.")
        return self._window_means_array

    @property
    def t_starts_array(self):
        if not hasattr(self, "_t_starts_array"):
            raise ValueError("You need to call fit first.")
        return self._t_starts_array

    @property
    def time_means_array(self):
        if not hasattr(self, "_time_means_array"):
            raise ValueError("You need to call fit first.")
        return self._time_means_array

    @property
    def n_components(self):
        if not hasattr(self, "_n_components"):
            raise ValueError("You need to call `cluster_omega()` first.")
        return self._n_components

    def fit(
        self,
        data,
        time,
        window_length,
        step_size,
        verbose=False,
        corner_sharpness=None,
    ):
        # Prepare window and data properties.
        self._window_length = window_length
        self._step_size = step_size
        self._n_time_steps, self._n_data_vars = self._data_shape(data)
        self._n_slides = self.build_windows(
            data, self._window_length, self._step_size
        )

        # If the window size and step size do not span the data in an integer
        # number of slides, we add one last window that has a smaller step spacing
        # relative to the other window spacings.
        n_slide_last_window = self._n_time_steps - (
            self._step_size * (self._n_slides - 1) + self._window_length
        )
        if n_slide_last_window > 0:
            self._n_slides += 1
        self._n_slide_last_window = n_slide_last_window

        # Build the projection basis if using a global svd.
        if self._global_svd:
            u = self._build_proj_basis(data, svd_rank=self._svd_rank)
            self._pydmd_kwargs["proj_basis"] = u
            self._pydmd_kwargs["use_proj"] = self._pydmd_kwargs.get(
                "use_proj", False
            )
            self._svd_rank = self._compute_svd_rank(
                data, svd_rank=self._svd_rank
            )
            svd_rank_pre_allocate = self._svd_rank
        elif not self._global_svd and self._svd_rank > 0:
            svd_rank_pre_allocate = self._compute_svd_rank(
                data, svd_rank=self._svd_rank
            )
        # If not using a global svd or a specified svd_rank, local u from each window is
        # used instead. The optimal svd_rank may change when using the locally optimal
        # svd_rank. To deal with this situation in the pre-allocation we give the
        # maximally allowed svd_rank for pre-allocation.
        elif self._max_rank is not None:
            svd_rank_pre_allocate = self._max_rank
        else:
            svd_rank_pre_allocate = self._n_data_vars

        # Pre-allocate all elements for the sliding window DMD.
        self._time_array = np.zeros((self._n_slides, self._window_length))
        self._modes_array = np.zeros(
            (self._n_slides, self._n_data_vars, svd_rank_pre_allocate),
            np.complex128,
        )
        self._omega_array = np.zeros(
            (self._n_slides, svd_rank_pre_allocate), np.complex128
        )
        self._amplitudes_array = np.zeros(
            (self._n_slides, svd_rank_pre_allocate), np.complex128
        )
        self._window_means_array = np.zeros((self._n_slides, self._n_data_vars))
        self._t_starts_array = np.zeros(self._n_slides)
        self._time_means_array = np.zeros(self._n_slides)

        # Get initial values for the eigenvalues.
        self._init_alpha = self._build_initizialization()

        # Initialize the BOPDMD object.
        optdmd = BOPDMD(
            svd_rank=self._svd_rank,
            init_alpha=self._init_alpha,
            **self._pydmd_kwargs,
        )

        # Round the corners of the window to shrink real components.
        lv_kern = self.calculate_lv_kern(
            self._window_length, corner_sharpness=corner_sharpness
        )

        # Perform the sliding window DMD fitting.
        for k in range(self._n_slides):
            if verbose:
                if k // 50 == k / 50:
                    print("{} of {}".format(k, self._n_slides))

            # Get the window indices and data.
            sample_start = self._step_size * k
            if k == self._n_slides - 1 and self._n_slide_last_window > 0:
                sample_slice = slice(-self._window_length, None)
            else:
                sample_slice = slice(
                    sample_start, sample_start + self._window_length
                )
            data_window = data[:, sample_slice]
            original_time_window = time[:, sample_slice]
            time_window_mean = np.mean(original_time_window)

            # All windows are fit with the time array reset to start at t=0.
            t_start = original_time_window[:, 0]
            time_window = original_time_window - t_start

            # Subtract off the time mean before rounding corners.
            c = np.mean(data_window, 1, keepdims=True)
            data_window = data_window - c

            # Round the corners of the window.
            data_window = data_window * lv_kern

            # Reset optdmd between iterations
            if not self._global_svd:
                # Get the svd rank for this window. Uses rank truncation when svd_rank is
                # not fixed, i.e. svd_rank = 0, otherwise uses the specified rank.
                _svd_rank = self._compute_svd_rank(
                    data_window, svd_rank=self._svd_rank
                )
                # Force svd rank to be even to allow for conjugate pairs.
                if _svd_rank % 2:
                    _svd_rank += 1
                # Force svd rank to not exceed a user specified amount.
                if self._max_rank is not None:
                    optdmd.svd_rank = min(_svd_rank, self._max_rank)
                else:
                    optdmd.svd_rank = _svd_rank
                optdmd._proj_basis = self._pydmd_kwargs["proj_basis"]

            # Fit the window using the optDMD.
            optdmd.fit(data_window, time_window)

            # Assign the results from this window.
            self._modes_array[k, :, : optdmd.modes.shape[-1]] = optdmd.modes
            self._omega_array[k, : optdmd.eigs.shape[0]] = optdmd.eigs
            self._amplitudes_array[
                k, : optdmd.eigs.shape[0]
            ] = optdmd.amplitudes
            self._window_means_array[k, :] = c.flatten()
            self._time_array[k, :] = original_time_window
            self._time_means_array[k] = time_window_mean
            self._t_starts_array[k] = t_start

            # Reset optdmd between iterations
            if not self._global_svd:
                # The default behavior is to reset the optdmd object to use the default
                # initial value (None) or the user provided values.
                if not self._use_last_freq:
                    optdmd.init_alpha = self._init_alpha
                # Otherwise use the eigenvalues from this window to seed the next window.
                elif self._use_last_freq:
                    optdmd.init_alpha = optdmd.eigs

    def cluster_omega(self, omega_array, n_components, kmeans_kwargs=None):
        self._n_components = n_components

        # Reshape the omega array into a 1d array
        n_slides = self._n_slides
        if self._svd_rank == 0:
            svd_rank = self._n_data_vars
        else:
            svd_rank = self._svd_rank
        omega_rshp = omega_array.reshape(n_slides * svd_rank)

        omega_squared = (np.conj(omega_rshp) * omega_rshp).astype("float")

        if kmeans_kwargs is None:
            random_state = 0
            kmeans_kwargs = {
                "n_init": "auto",
                "random_state": random_state,
            }
        kmeans = KMeans(n_clusters=n_components, **kmeans_kwargs)
        omega_classes = kmeans.fit_predict(np.atleast_2d(omega_squared).T)
        omega_classes = omega_classes.reshape(n_slides, svd_rank)
        cluster_centroids = kmeans.cluster_centers_.flatten()

        return omega_classes, omega_squared, cluster_centroids

    @staticmethod
    def plot_omega_squared_histogram(omega_squared, cluster_centroids):
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig, ax = plt.subplots(1, 1)
        ax.hist(omega_squared, bins=64)
        ax.set_xlabel("$|\omega|^{2}$")
        ax.set_ylabel("Count")
        ax.set_title("$|\omega|^2$ Spectrum & k-Means Centroids")
        [
            ax.axvline(c, color=colors[nc % len(colors)])
            for nc, c in enumerate(cluster_centroids)
        ]

        return fig, ax

    def plot_omega_squared_time_series(self, omega_squared, omega_classes):
        fig, ax = plt.subplots(1, 1)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # Reshape the omega array into a 1d array
        n_slides = self._n_slides
        if self._svd_rank == 0:
            svd_rank = self._n_data_vars
        else:
            svd_rank = self._svd_rank

        for ncomponent, component in enumerate(range(self._n_components)):
            ax.plot(
                self._time_means_array,
                np.where(
                    omega_classes == component,
                    omega_squared.reshape((n_slides, svd_rank)),
                    np.nan,
                ),
                color=colors[ncomponent % len(colors)],
            )
        ax.set_ylabel("$|\omega|^{2}$")
        ax.set_xlabel("Time")
        ax.set_title("$|\omega|^2$ Spectrum (Moving Window)")

        return fig, ax

    def global_reconstruction(
        self,
    ):
        # Container for the reconstructed time series
        glbl_reconstruction = np.zeros(
            (self._n_data_vars, self._n_time_steps)
        ).astype("complex128")

        # Count the number of windows contributing to each step
        xn = np.zeros(self._n_time_steps)

        for k in range(self._n_slides):
            # Extract out the DMD fit for this window.
            w = self._modes_array[k]
            b = self._amplitudes_array[k]
            omega = np.atleast_2d(self._omega_array[k]).T
            c = np.atleast_2d(self._window_means_array[k]).T

            # Compute each segment starting at t=0
            t = self._time_array[k]
            t_start = self._t_starts_array[k]
            t = t - t_start

            # Perform the global reconstruction.
            recon_window = (
                np.linalg.multi_dot([w, np.diag(b), np.exp(omega * t)]) + c
            )

            if k == self._n_slides - 1 and self._n_slide_last_window > 0:
                window_indices = slice(-self._window_length, None)
            else:
                window_indices = slice(
                    k * self._step_size,
                    k * self._step_size + self._window_length,
                )
            glbl_reconstruction[:, window_indices] += recon_window
            xn[window_indices] += 1

        # Weight xr so all steps are on equal footing
        glbl_reconstruction = glbl_reconstruction / xn

        return glbl_reconstruction

    def scale_reconstruction(
        self,
        omega_classes,
        cluster_centroids,
        suppress_growth=True,
        include_means=True,
    ):
        """Reconstruct the sliding mrDMD into the constituent components.

        The reconstructed data are convolved with a guassian filter since
        points near the middle of the window are more reliable than points
        at the edge of the window. Note that this will leave the beginning
        and end of time series prone to larger errors. A best practice is
        to cut off `window_length` from each end before further analysis.


        suppress_growth:
        Kill positive real components of frequencies
        """

        # Each individual reconstructed window
        xr_sep = np.zeros(
            (self._n_components, self._n_data_vars, self._n_time_steps)
        )

        # Track the total contribution from all windows to each time step
        xn = np.zeros(self._n_time_steps)

        # Convolve each windowed reconstruction with a gaussian filter.
        # Std dev of gaussian filter
        recon_filter_sd = self._window_length / 8
        recon_filter = np.exp(
            -(
                (np.arange(self._window_length) - (self._window_length + 1) / 2)
                ** 2
            )
            / recon_filter_sd**2
        )

        for k in range(self._n_slides):
            w = self._modes_array[k]
            b = self._amplitudes_array[k]
            omega = np.atleast_2d(self._omega_array[k]).T
            classification = omega_classes[k]

            if suppress_growth:
                omega[omega.real > 0] = 1j * omega[omega.real > 0].imag

            c = np.atleast_2d(self._window_means_array[k]).T

            # Compute each segment of xr starting at "t = 0"
            t = self._time_array[k]
            t_start = self._t_starts_array[k]
            t = t - t_start

            xr_sep_window = np.zeros(
                (self._n_components, self._n_data_vars, self._window_length)
            )
            for j in np.unique(omega_classes):
                xr_sep_window[j, :, :] = np.linalg.multi_dot(
                    [
                        w[:, classification == j],
                        np.diag(b[classification == j]),
                        np.exp(omega[classification == j] * t),
                    ]
                )

                # Add the constant offset to the lowest frequency cluster.
                if include_means and (j == np.argmin(cluster_centroids)):
                    xr_sep_window[j, :, :] += c
                xr_sep_window[j, :, :] = xr_sep_window[j, :, :] * recon_filter

                if k == self._n_slides - 1 and self._n_slide_last_window > 0:
                    window_indices = slice(-self._window_length, None)
                else:
                    window_indices = slice(
                        k * self._step_size,
                        k * self._step_size + self._window_length,
                    )
                xr_sep[j, :, window_indices] = (
                    xr_sep[j, :, window_indices] + xr_sep_window[j, :, :]
                )

            xn[window_indices] += recon_filter

        xr_sep = xr_sep / xn

        return xr_sep
