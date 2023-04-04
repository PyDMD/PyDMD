import numpy as np
from pydmd.bopdmd import BOPDMD
import scipy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class multi_res_discovery:
    def __init__(
            self,
            window_length=None,
            step_size=None,
            n_components=None,
            svd_rank=None,
            global_svd=True,
            initialize_artificially=False,
            use_last_freq=False,
            use_kmean_freqs=False,
            init_alpha=None,
            pydmd_kwargs=None,
            threshhold_percent=1,
            cluster_centroids=None,
    ):
        self._n_components = n_components
        self._step_size = step_size
        self._svd_rank = svd_rank
        self._window_length = window_length
        self._global_svd = global_svd
        self._initialize_artificially = initialize_artificially
        self._use_last_freq = use_last_freq
        self._use_kmean_freqs = use_kmean_freqs
        self._init_alpha = init_alpha
        self._cluster_centroids = cluster_centroids
        if pydmd_kwargs is None:
            self._pydmd_kwargs = {
                'eig_sort': 'imag',
            }
        else:
            self._pydmd_kwargs = pydmd_kwargs

        # Set the threshold for the number of frequencies to use in clustering,
        # where the frequencies have been sorted in order of magnitude.
        # Note: this value is not currently implemented (it was always 1 in examples).
        self._threshhold_percent = threshhold_percent

    def _compute_svd_rank(self, data, svd_rank=None):
        # rank to fit w/ optdmd
        _, n_data_vars = self._data_shape(data)

        if svd_rank is None:
            svd_rank = n_data_vars
        elif svd_rank > n_data_vars:
            raise ValueError('svd_rank is greater than the number of spatial dimensions.')
        return svd_rank

    def _build_proj_basis(self, data, global_svd=True, svd_rank=None):

        if svd_rank is None:
            self._svd_rank = self._compute_svd_rank(data, svd_rank=svd_rank)

        # Use global SVD modes for each DMD rather than individual SVD on
        # each window.
        if global_svd:
            # Recover the first r modes of the global svd
            u, _, _ = scipy.linalg.svd(data, full_matrices=False)
            return u[:, :self._svd_rank]

    def _build_initizialization(self):
        """ Method for making initial guess of DMD eigenvalues.
        """

        # User provided initial eigenvalues.
        init_alpha = None
        if self._initialize_artificially and self._init_alpha is not None:
            init_alpha = self._init_alpha
        # Initial eigenvalue guesses from kmeans clustering.
        elif self._initialize_artificially and self._init_alpha is None:
            if self._use_kmean_freqs:
                init_alpha = np.repeat(np.sqrt(self._cluster_centroids) * 1j,
                                       int(self._svd_rank / self._n_components))
                init_alpha = init_alpha * np.tile([1, -1],
                                                  int(self._svd_rank / self._n_components))

        return init_alpha

    def build_windows(self, data, integer_windows=False):
        """Calculate how many times to slide the window across the data.

        """

        if integer_windows:
            n_split = np.floor(data.shape[1] / self._window_length).astype(int)
        else:
            n_split = data.shape[1] / self._window_length

        n_steps = int((self._window_length * n_split))

        # Number of sliding-window iterations
        n_slides = np.floor((n_steps - self._window_length) / self._step_size).astype(int)

        return n_slides

    def calculate_lv_kern(self, window_length, corner_sharpness=None):
        """Calculate the kerning window for suppressing real eigenvalues.

        """

        # Higher = sharper corners
        if corner_sharpness is None:
            corner_sharpness = 16

        lv_kern = (
                np.tanh(
                    corner_sharpness * np.arange(1, window_length + 1) / window_length
                )
                - np.tanh(
            corner_sharpness * (
                    np.arange(1, window_length + 1) - window_length) / window_length) - 1
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
            raise ValueError("You need to call fit before")
        return self._amplitudes_array

    @property
    def omega_array(self):
        if not hasattr(self, "_omega_array"):
            raise ValueError("You need to call fit before")
        return self._omega_array

    @property
    def omega_array(self):
        if not hasattr(self, "_omega_array"):
            raise ValueError("You need to call fit before")
        return self._omega_array

    @property
    def time_array(self):
        if not hasattr(self, "_time_array"):
            raise ValueError("You need to call fit before")
        return self._time_array

    @property
    def window_means_array(self):
        if not hasattr(self, "_window_means_array"):
            raise ValueError("You need to call fit before")
        return self._window_means_array

    @property
    def t_starts_array(self):
        if not hasattr(self, "_t_starts_array"):
            raise ValueError("You need to call fit before")
        return self._t_starts_array

    @property
    def time_means_array(self):
        if not hasattr(self, "_time_means_array"):
            raise ValueError("You need to call fit before")
        return self._time_means_array

    def fit(self, data, time, window_length, step_size, verbose=False,
            corner_sharpness=None):
        self._window_length = window_length
        self._step_size = step_size

        self._n_time_steps, self._n_data_vars = self._data_shape(data)
        self._n_slides = self.build_windows(data)
        self._svd_rank = self._compute_svd_rank(data, svd_rank=self._svd_rank)

        # Check dimensionality/shape of all
        # Each element calculate for a window is returned to the user in these array.
        data_array = np.zeros((self._n_slides, self._n_data_vars, self._window_length))
        self._time_array = np.zeros((self._n_slides, self._window_length))
        self._modes_array = np.zeros((self._n_slides, self._n_data_vars, self._svd_rank),
                                     np.complex128)
        self._omega_array = np.zeros((self._n_slides, self._svd_rank), np.complex128)
        self._amplitudes_array = np.zeros((self._n_slides, self._svd_rank), np.complex128)
        self._window_means_array = np.zeros((self._n_slides, self._n_data_vars))
        self._t_starts_array = np.zeros(self._n_slides)
        self._time_means_array = np.zeros(self._n_slides)

        # Round the corners of the window to shrink real components.
        lv_kern = self.calculate_lv_kern(self._window_length,
                                         corner_sharpness=corner_sharpness)

        # Build the projection basis if using a global svd. If not provided local u is used instead.
        if self._global_svd:
            u = self._build_proj_basis(data, global_svd=self._global_svd,
                                       svd_rank=self._svd_rank)
            self._pydmd_kwargs['proj_basis'] = u
            self._pydmd_kwargs['use_proj'] = False

        # Initialize the DMD class.
        if self._initialize_artificially:
            self._init_alpha = self._build_initizialization()
            optdmd = BOPDMD(
                svd_rank=self._svd_rank, num_trials=0, init_alpha=self._init_alpha,
                **self._pydmd_kwargs
            )
        else:
            optdmd = BOPDMD(
                svd_rank=self._svd_rank, num_trials=0, **self._pydmd_kwargs,
            )

        for k in range(self._n_slides):
            if verbose:
                if k // 50 == k / 50:
                    print('{} of {}'.format(k, self._n_slides))

            sample_start = self._step_size * k
            sample_steps = np.arange(sample_start, sample_start + self._window_length)

            data_window = data[:, sample_steps]
            time_window = time[:, sample_steps]
            data_array[k, :, :] = data_window
            self._time_array[k, :] = time_window
            self._time_means_array[k] = np.mean(time_window)

            t_start = time_window[:, 0]
            time_window = time_window - t_start
            self._t_starts_array[k] = t_start

            # Subtract off mean before rounding corners
            # https://stackoverflow.com/questions/32030343/
            # subtracting-the-mean-of-each-row-in-numpy-with-broadcasting
            c = np.mean(data_window, 1, keepdims=True)
            data_window = data_window - c

            # Round corners of the window
            data_window = data_window * lv_kern

            # Fit with the desired DMD class
            optdmd.fit(data_window, time_window)

            # if use_last_freq == 1:
            #     e_init = e

            # Assign the results from this window
            self._modes_array[k, ::] = optdmd.modes
            self._omega_array[k, :] = optdmd.eigs
            self._amplitudes_array[k, :] = optdmd.amplitudes
            self._window_means_array[k, :] = c.flatten()

    def cluster_omega(self, omega_array, n_components, kmeans_kwargs=None):
        self._n_components = n_components
        omega_rshp = omega_array.reshape(self._n_slides * self._svd_rank)
        omega_squared = (np.conj(omega_rshp) * omega_rshp).astype('float')

        if kmeans_kwargs is None:
            random_state = 0
            kmeans_kwargs = {
                "n_init": "auto",
                "random_state": random_state,
            }
        kmeans = KMeans(n_clusters=self._n_components, **kmeans_kwargs)
        y_pred = kmeans.fit_predict(np.atleast_2d(omega_squared).T)
        omega_classes = y_pred.reshape(self._n_slides, self._svd_rank)
        cluster_centroids = kmeans.cluster_centers_.flatten()

        return omega_classes, omega_squared, cluster_centroids

    def plot_omega_squared_histogram(self, omega_squared, cluster_centroids):
        fig, ax = plt.subplots(1, 1)
        ax.hist(omega_squared, bins=64)
        ax.set_xlabel('$|\omega|^{2}$')
        ax.set_ylabel('Count')
        ax.set_title('$|\omega|^2$ Spectrum & k-Means Centroids')
        [ax.axvline(c, color='r') for c in cluster_centroids]

        return fig, ax

    def plot_omega_squared_time_series(self, omega_squared, omega_classes):
        fig, ax = plt.subplots(1, 1)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for ncomponent, component in enumerate(range(self._n_components)):
            ax.plot(
                self._time_means_array,
                np.where(
                    omega_classes == component,
                    omega_squared.reshape((self._n_slides, self._svd_rank)), np.nan
                ),
                color=colors[ncomponent % len(colors)]
            )
        ax.set_ylabel('$|\omega|^{2}$')
        ax.set_xlabel('Time')
        ax.set_title('$|\omega|^2$ Spectrum (Moving Window)')

        return fig, ax

    def global_reconstruction(self, ):
        # Container for the reconstructed time series
        glbl_reconstruction = np.zeros(
            (self._n_data_vars, self._n_time_steps)).astype('complex128')

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
            recon_window = np.linalg.multi_dot([w, np.diag(b), np.exp(omega * t)]) + c

            window_indices = slice(k * self._step_size,
                                   k * self._step_size + self._window_length)
            glbl_reconstruction[:, window_indices] += recon_window
            xn[window_indices] += 1

        # Weight xr so all steps are on equal footing
        glbl_reconstruction = glbl_reconstruction / xn

        return glbl_reconstruction

    def scale_reconstruction(self, omega_classes, cluster_centroids, suppress_growth=True,
                             include_means=True):
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
        xr_sep = np.zeros((self._n_components, self._n_data_vars, self._n_time_steps))

        # Track the total contribution from all windows to each time step
        xn = np.zeros(self._n_time_steps)

        # Convolve each windowed reconstruction with a gaussian filter.
        # Std dev of gaussian filter
        recon_filter_sd = self._window_length / 8
        recon_filter = np.exp(
            -(
                     np.arange(self._window_length)
                     - (self._window_length + 1) / 2
             ) ** 2 / recon_filter_sd ** 2)

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
                (self._n_components, self._n_data_vars, self._window_length))
            for j in np.unique(omega_classes):
                xr_sep_window[j, :, :] = np.linalg.multi_dot(
                    [
                        w[:, classification == j],
                        np.diag(b[classification == j]),
                        np.exp(omega[classification == j] * t)
                    ]
                )

                # Add the constant offset to the lowest frequency cluster.
                if include_means and (j == np.argmin(cluster_centroids)):
                    xr_sep_window[j, :, :] += c
                xr_sep_window[j, :, :] = xr_sep_window[j, :, :] * recon_filter
                window_indices = slice(k * self._step_size,
                                       k * self._step_size + self._window_length)
                xr_sep[j, :, window_indices] = xr_sep[j, :,
                                               window_indices] + xr_sep_window[j, :, :]

            xn[window_indices] += recon_filter

        xr_sep = xr_sep / xn

        return xr_sep

