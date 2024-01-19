import numpy as np
from pydmd.bopdmd import BOPDMD
from .utils import compute_rank, compute_svd
import copy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import xarray as xr


class COSTS:
    """Coherent Spatio-Temporal Scale Separation with DMD.

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
    :type pydmd_kwargs: dict
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
    :param use_kmean_freqs: Flag specifying if the BOPDMD fit should use initial values
        taken from cluster centroids, e.g., from a previoius iteration.
    :type use_kmean_freqs: bool
    :param init_alpha: Initial guess for the eigenvalues provided to BOPDMD. Must be equal
        to the `svd_rank`.
    :type init_alpha: numpy array
    :param max_rank: Maximum allowed `svd_rank`. Overrides the optimal rank truncation if
        `svd_rank=0`.
    :type max_rank: int
    :param n_components: Number of frequency bands to use for clustering.
    :type n_components: int
    :param force_even_eigs: Flag specifying if the `svd_rank` should be forced to be even.
    :type force_even_eigs: bool
    :param reset_alpha_init: Flag specifying if the initial eigenvalue guess should be reset
        between windows.
    :type reset_alpha_init: bool
    """

    def __init__(
        self,
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
        self._hist_kwargs = None
        self._omega_label = None
        self._step_size = None
        self._window_length = None
        self._n_components = n_components
        self._svd_rank = svd_rank
        self._global_svd = global_svd
        self._initialize_artificially = initialize_artificially
        self._use_last_freq = use_last_freq
        self._init_alpha = init_alpha
        self._cluster_centroids = cluster_centroids
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
        self._amplitudes_array = None
        self._cluster_centroids = None
        self._omega_classes = None
        self._transform_method = None
        self._window_means_array = None
        self._non_integer_n_slide = None

        # Specify default keywords to hand to BOPDMD.
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

    @property
    def svd_rank(self):
        """
        :return: the rank used for the svd truncation.
        :rtype: int or float
        """
        return self._svd_rank

    @property
    def global_svd(self):
        """
        :return: If a global svd was used for the BOP-DMD fit.
        :rtype: int or float
        """
        return self._global_svd

    @property
    def window_length(self):
        """
        :return: the length of the windows used for this decomposition level.
        :rtype: int or float
        """
        return self._window_length

    @property
    def step_size(self):
        """
        :return: the length of the windows used for this decomposition level.
        :rtype: int or float
        """
        return self._step_size

    @property
    def n_slides(self):
        """
        :return: number of window slides for this decomposition level.
        :rtype: int
        """
        return self._n_slides

    @property
    def modes_array(self):
        """
        :return: Modes for each window
        :rtype: numpy.ndarray
        """
        if not hasattr(self, "_modes_array"):
            raise ValueError("You need to call fit before")
        return self._modes_array

    @property
    def amplitudes_array(self):
        """
        :return: amplitudes of each window
        :rtype: numpy.ndarray
        """
        if not hasattr(self, "_amplitudes_array"):
            raise ValueError("You need to call fit first.")
        return self._amplitudes_array

    @property
    def omega_array(self):
        """
        :return: omega (a.k.a eigenvalues or time dynamics) for each window
        :rtype: numpy.ndarray
        """
        if not hasattr(self, "_omega_array"):
            raise ValueError("You need to call fit first.")
        return self._omega_array

    @property
    def time_array(self):
        """
        :return: time values for each fit window
        :rtype: numpy.ndarray
        """
        if not hasattr(self, "_time_array"):
            raise ValueError("You need to call fit first.")
        return self._time_array

    @property
    def window_means_array(self):
        """
        :return: Time mean of the data in each window
        :rtype: numpy.ndarray
        """
        if not hasattr(self, "_window_means_array"):
            raise ValueError("You need to call fit first.")
        return self._window_means_array

    @property
    def n_components(self):
        """
        :return: Number of frequency bands fit in the kmeans clustering
        :rtype: int
        """
        if not hasattr(self, "_n_components"):
            raise ValueError("You need to call `cluster_omega()` first.")
        return self._n_components

    @property
    def cluster_centroids(self):
        """
        :return: Centroids of the frequency bands
        :rtype: numpy.ndarray
        """
        if not hasattr(self, "_cluster_centroids"):
            raise ValueError("You need to call `cluster_omega()` first.")
        return self._cluster_centroids

    @property
    def omega_classes(self):
        """
        :return: Frequency band classifications, corresponds to omega_array
        :rtype: numpy.ndarray
        """
        if not hasattr(self, "_omega_classes"):
            raise ValueError("You need to call `cluster_omega()` first.")
        return self._omega_classes

    def periods(self):
        """
        :return: Time dynamics converted to periods
        :rtype: numpy.ndarray
        """
        frequencies = np.abs(
            self._omega_array[self._omega_classes > 0].imag.flatten()
        )
        return 2 * np.pi / frequencies

    @staticmethod
    def relative_error(x_est, x_true):
        """Helper function for calculating the relative error."""
        return np.linalg.norm(x_est - x_true) / np.linalg.norm(x_true)

    @staticmethod
    def build_windows(data, window_length, step_size, integer_windows=False):
        """How many times integer slides fit the data for a given step and window size.

        :param data: 1D snapshots for fitting
        :type data: numpy.ndarray
        :param window_length: Length of the fitting window in units of time steps
        :type window_length: int
        :param step_size:  Distance to slide each window.
        :type step_size: int
        :param integer_windows: Whether to force an integer number of windows
        :type integer_windows: bool
        :return:
        """

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

    @staticmethod
    def build_kern(window_length):
        """Build the convolution kernel for the window reconstruction.

        :param window_length: Length of the data window in units of time
        :return:
        """
        recon_filter_sd = window_length / 8
        recon_filter = np.exp(
            -((np.arange(window_length) - (window_length + 1) / 2) ** 2)
            / recon_filter_sd**2
        )
        return recon_filter

    @staticmethod
    def _data_shape(data):
        """Returns the shape of the data."""
        n_time_steps = np.shape(data)[1]
        n_data_vars = np.shape(data)[0]
        return n_time_steps, n_data_vars

    def _build_proj_basis(self, data, svd_rank=None):
        """Build the projection basis."""
        self._svd_rank = compute_rank(data, svd_rank=svd_rank)
        # Recover the first r modes of the global svd
        u, _, _ = compute_svd(data, svd_rank=svd_rank)
        return u

    def _build_initizialization(self):
        """Method for making initial guess of DMD eigenvalues.

        :return: First guess of eigenvalues
        :rtype: numpy.ndarray or None
        """
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

    def fit(
        self,
        data,
        time,
        window_length,
        step_size,
        verbose=False,
        corner_sharpness=None,
    ):
        """Fit COherent SpatioTemporal Scale separation (COSTS).

        :param data: the input data to decompose (1D snapshots). Dimensions space vs time.
        :type data: numpy.ndarray
        :param time: time series labeling the 1D snapshots
        :type time: numpy.ndarray
        :param window_length: decomposition window length
        :type window_length: int
        :param step_size: how far to slide each window from the previous window.
        :type step_size: int
        :param verbose: notifies progress for fitting. Default is False.
        :type verbose: bool
        :param corner_sharpness: See `calculate_lv_kern`
        :type corner_sharpness: float or int
        """

        # Prepare window and data properties.
        self._window_length = window_length
        self._step_size = step_size
        self._n_time_steps, self._n_data_vars = self._data_shape(data)
        self._n_slides = self.build_windows(
            data, self._window_length, self._step_size
        )

        if self._window_length > self._n_time_steps:
            raise ValueError(
                "Window length (n={}) is larger than the time dimension (n={})".format(
                    self._window_length, self._n_time_steps
                )
            )

        # If the window size and step size do not span the data in an integer
        # number of slides, we add one last window that has a smaller step spacing
        # relative to the other window spacings.
        n_slide_last_window = self._n_time_steps - (
            self._step_size * (self._n_slides - 1) + self._window_length
        )
        if n_slide_last_window > 0:
            self._n_slides += 1
            self._non_integer_n_slide = True
        else:
            self._non_integer_n_slide = False

        # Build the projection basis if using a global svd.
        if self._global_svd:
            u = self._build_proj_basis(data, svd_rank=self._svd_rank)
            self._pydmd_kwargs["proj_basis"] = u
            self._pydmd_kwargs["use_proj"] = self._pydmd_kwargs.get(
                "use_proj", False
            )
            self._svd_rank = compute_rank(data, svd_rank=self._svd_rank)
            svd_rank_pre_allocate = self._svd_rank
        elif not self._global_svd and self._svd_rank > 0:
            if self._force_even_eigs and self._svd_rank % 2:
                raise ValueError(
                    "svd_rank is odd, but force_even_eigs is True."
                )
            if self._svd_rank > self._n_data_vars:
                raise ValueError(
                    "Rank is larger than the data spatial dimension."
                )
            svd_rank_pre_allocate = compute_rank(data, svd_rank=self._svd_rank)
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

        # Get initial values for the eigenvalues.
        self._init_alpha = self._build_initizialization()

        # Initialize the BOPDMD object.
        optdmd = BOPDMD(
            svd_rank=self._svd_rank,
            init_alpha=self._init_alpha,
            **self._pydmd_kwargs,
        )

        # Round the corners of the windowed data towards zero, which shrinks
        # the real components of the fitted eigenvalues away from unrealistic
        # exponential growth.
        lv_kern = self.calculate_lv_kern(
            self._window_length, corner_sharpness=corner_sharpness
        )

        # Perform the sliding window DMD fitting.
        for k in range(self._n_slides):
            if verbose:
                if k // 50 == k / 50:
                    print("{} of {}".format(k, self._n_slides))

            sample_slice = self.get_window_indices(k)
            data_window = data[:, sample_slice]
            original_time_window = time[:, sample_slice]

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
                # Get the svd rank for this window. Uses rank truncation when
                # svd_rank is not fixed, i.e. svd_rank = 0, otherwise uses the
                # specified rank.
                _svd_rank = compute_rank(data_window, svd_rank=self._svd_rank)
                # Force svd rank to be even to allow for conjugate pairs.
                if self._force_even_eigs and _svd_rank % 2:
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

            # Reset optdmd between iterations
            if not self._global_svd:
                # The default behavior is to reset the optdmd object to use the default
                # initial value (None) or the user provided values.
                if not self._use_last_freq:
                    optdmd.init_alpha = self._init_alpha
                # Otherwise use the eigenvalues from this window to seed the next window.
                elif self._use_last_freq:
                    optdmd.init_alpha = optdmd.eigs

    def get_window_indices(self, k):
        """Returns the window indices for slide `k`.

        Handles non-integer number of slides by making the last slide
        correspond to `slice(-window_length, None)`.

        :param k: Window to index
        :type k: int
        :return: slice indexing the given window
        :rtype: slice
        """
        # Get the window indices and data.
        sample_start = self._step_size * k
        if k == self._n_slides - 1 and self._non_integer_n_slide:
            sample_slice = slice(-self._window_length, None)
        else:
            sample_slice = slice(
                sample_start, sample_start + self._window_length
            )
        return sample_slice

    def cluster_omega(
        self,
        n_components,
        kmeans_kwargs=None,
        transform_method=None,
        method="KMeans",
    ):
        """Clusters fitted eigenvalues into frequency bands by the imaginary component.

        :param method: Choose clustering strategy: KMeans (default) or KMediods
            (requires sklearn_extras).
        :type method: str
        :param n_components: Hyperparameter for k-means clustering, number of clusters.
        :type n_components: int
        :param kmeans_kwargs: Arguments for KMeans clustering.
        :type kmeans_kwargs: dict
        :param transform_method: How to transform omega. See docstring for valid options.
        :type transform_method: str or NoneType
        :return:
        """
        # Reshape the omega array into a 1d array
        omega_array = self.omega_array
        n_slides = omega_array.shape[0]
        svd_rank = omega_array.shape[1]
        omega_rshp = omega_array.reshape(n_slides * svd_rank)
        omega_transform = self.transform_omega(
            omega_rshp, transform_method=transform_method
        )

        if kmeans_kwargs is None:
            random_state = 0
            kmeans_kwargs = {
                # "n_init": "auto",
                "random_state": random_state,
            }
        if method == "KMeans":
            clustering = KMeans(n_clusters=n_components, **kmeans_kwargs)
        elif method == "KMediods":
            from sklearn_extra.cluster import KMedoids

            clustering = KMedoids(n_clusters=n_components, **kmeans_kwargs)

        omega_classes = clustering.fit_predict(np.atleast_2d(omega_transform).T)
        omega_classes = omega_classes.reshape(n_slides, svd_rank)
        cluster_centroids = clustering.cluster_centers_.flatten()

        # Sort the clusters by the centroid magnitude.
        idx = np.argsort(cluster_centroids)
        lut = np.zeros_like(idx)
        lut[idx] = np.arange(n_components)
        omega_classes = lut[omega_classes]
        cluster_centroids = cluster_centroids[idx]

        # Assign the results to the object.
        self._cluster_centroids = cluster_centroids
        self._omega_classes = omega_classes
        self._transform_method = transform_method
        self._n_components = n_components

        return self

    def transform_omega(self, omega_array, transform_method=None):
        """Transform omega, primarily for clustering.
        Options for transforming omega are:
            "period": :math:`\\frac{1}{\\omega}`
            "log10": :math:`log10(\\omega)`
            "square_frequencies": :math:`\\omega^2`
            "absolute": :math:`|\\omega|`
        Default value is "absolute". All transformations and clustering are performed on
        the imaginary portion of omega.

        :param omega_array:
        :param transform_method:
        :return: transformed omega array
        :rtype: numpy.ndarray
        """
        # Apply a transformation to omega to improve frequency band separation
        if transform_method is None or transform_method == "absolute":
            omega_transform = np.abs(omega_array.imag.astype("float"))
            self._omega_label = r"$|\omega|$"
            self._hist_kwargs = {"bins": 64}
        # Outstanding question: should this be the complex conjugate or
        # the imaginary component squared?
        elif transform_method == "square_frequencies":
            omega_transform = (np.conj(omega_array) * omega_array).real.astype(
                "float"
            )
            self._omega_label = r"$|\omega|^{2}$"
            self._hist_kwargs = {"bins": 64}
        elif transform_method == "log10":
            omega_transform = np.log10(np.abs(omega_array.imag.astype("float")))
            # Impute log10(0) with the smallest non-zero values in log10(omega).
            zero_imputer = omega_transform[np.isfinite(omega_transform)].min()
            omega_transform[~np.isfinite(omega_transform)] = zero_imputer
            self._omega_label = r"$log_{10}(|\omega|)$"
            self._hist_kwargs = {
                "bins": np.linspace(
                    np.min(np.log10(omega_transform[omega_transform > 0])),
                    np.max(np.log10(omega_transform[omega_transform > 0])),
                    64,
                )
            }
        elif transform_method == "period":
            omega_transform = 1 / np.abs(omega_array.imag.astype("float"))
            self._omega_label = "Period"
            # @ToDo: Specify bins like in log10 transform
            self._hist_kwargs = {"bins": 64}
        else:
            raise ValueError(
                "Transform method {} not supported.".format(transform_method)
            )

        return omega_transform

    def cluster_hyperparameter_sweep(
        self, n_components_range=None, transform_method=None
    ):
        """Hyperparameter search for number of frequency bands.

        Searches for the optimal number of clusters to use in kmeans clustering
        separation of the frequency bands. To best separate frequency bands it may
        be necessary to transform omega. Scores clusters using the silhouette score
        which can be slow to compute.

        Options for transforming omega are:
            "period": :math:`\\frac{1}{\\omega}`
            "log10": :math:`log10(\\omega)`
            "square_frequencies": :math:`\\omega^2`
            "absolute": :math:`|\\omega|`
        Default value is "absolute". All transformations and clustering are performed on
        the imaginary portion of omega.

        :param n_components_range: Range of n_components for the sweep.
        :type n_components_range: numpy.ndarray of ints
        :param transform_method: How to transform the imaginary component of omega.
        :type transform_method: str
        :return: optimal value of `n_components` for clustering.
        """
        if n_components_range is None:
            n_components_range = np.arange(
                np.max((self.svd_rank // 4, 2)), self.svd_rank
            )
        score = np.zeros_like(n_components_range, float)

        # Reshape the omega array into a 1d array
        omega_array = self.omega_array
        n_slides = omega_array.shape[0]
        svd_rank = omega_array.shape[1]
        omega_rshp = omega_array.reshape(n_slides * svd_rank)

        # Apply the transformation
        omega_transform = self.transform_omega(
            omega_rshp, transform_method=transform_method
        )

        for nind, n in enumerate(n_components_range):
            _ = self.cluster_omega(
                n_components=n, transform_method=transform_method
            )

            classes_reshape = self.omega_classes.reshape(n_slides * svd_rank)

            score[nind] = silhouette_score(
                np.atleast_2d(omega_transform).T,
                np.atleast_2d(classes_reshape).T,
            )

        return n_components_range[np.argmax(score)]

    def plot_omega_histogram(self):
        """Histogram of fit frequencies.

        This plot is useful for assessing if the frequencies bands were well separated.
        A good choice of transformation and clustering will have clearly separated clusters.

        :return fig: Figure handle for the plot
        :return ax: Axes handle for the plot
        """
        # Reshape the omega array into a 1d array
        omega_array = self.omega_array
        n_slides = omega_array.shape[0]
        svd_rank = omega_array.shape[1]
        omega_rshp = omega_array.reshape(n_slides * svd_rank)

        # Apply the transformation to omega
        omega_transform = self.transform_omega(
            omega_rshp, transform_method=self._transform_method
        )

        label = self._omega_label
        hist_kwargs = self._hist_kwargs

        cluster_centroids = self._cluster_centroids

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig, ax = plt.subplots(1, 1)
        ax.hist(omega_transform, **hist_kwargs)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(r"$\omega$ Spectrum & k-Means Centroids")
        [
            ax.axvline(c, color=colors[nc % len(colors)])
            for nc, c in enumerate(cluster_centroids)
        ]

        return fig, ax

    def plot_omega_time_series(self):
        """Time series of transformed omega colored by frequency band.

        :return fig: figure handle for the plot
        :rtype fig: matplotlib.figure()
        :return ax: matplotlib subplot instances
        :rtype ax: matplotlib.Axes()
        """
        fig, ax = plt.subplots(1, 1)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # Reshape the omega array into a 1d array
        omega_array = self.omega_array
        n_slides = omega_array.shape[0]
        svd_rank = omega_array.shape[1]
        omega_rshp = omega_array.reshape(n_slides * svd_rank)

        # Apply the transformation to omega
        omega_transform = self.transform_omega(
            omega_rshp, transform_method=self._transform_method
        )
        label = self._omega_label

        for ncomponent, component in enumerate(range(self._n_components)):
            ax.plot(
                np.mean(self.time_array, axis=1),
                np.where(
                    self._omega_classes == component,
                    omega_transform.reshape((n_slides, svd_rank)),
                    np.nan,
                ),
                color=colors[ncomponent % len(colors)],
                ls="None",
                marker=".",
            )
        ax.set_ylabel(label)
        ax.set_xlabel("Time")
        ax.set_title("Time dynamics time series")

        return fig, ax

    def global_reconstruction(self, scale_reconstruction_kwargs=None):
        """Helper function for generating the global reconstruction.

        :param scale_reconstruction_kwargs: Arguments for the scale reconstruction.
        :type scale_reconstruction_kwargs: dict
        :return: Global reconstruction (sum of all frequency bands)
        :rtype: numpy.ndarray
        """
        if scale_reconstruction_kwargs is None:
            scale_reconstruction_kwargs = {}
        xr_sep = self.scale_reconstruction(**scale_reconstruction_kwargs)
        x_global_recon = xr_sep.sum(axis=0)
        return x_global_recon

    def scale_reconstruction(
        self,
        suppress_growth=True,
        include_means=True,
    ):
        """Reconstruct the spatiotemporal features for each frequency band.

        The reconstructed data are convolved with a guassian filter since
        points near the middle of the window are more reliable than points
        at the edge of the window. Note that this will leave the beginning
        and end of time series prone to larger errors. A best practice is
        to cut off `window_length` from each end before further analysis.

        :param suppress_growth: Not API stable
        :param include_means: Not API stable
        :return: Reconstruction for each frequency band with dimensions of:
            n_components x n_data_vars x n_time_steps
        :rtype: numpy.ndarray
        """

        # Each individual reconstructed window
        xr_sep = np.zeros(
            (self._n_components, self._n_data_vars, self._n_time_steps)
        )

        # Track the total contribution from all windows to each time step
        xn = np.zeros(self._n_time_steps)

        # Convolve each windowed reconstruction with a gaussian filter.
        # Weights points in the middle of the window and de-emphasizes the
        # edges of the window.
        recon_filter = self.build_kern(self._window_length)

        for k in range(self._n_slides):
            window_indices = self.get_window_indices(k)

            w = self._modes_array[k]
            b = self._amplitudes_array[k]
            # @ToDo: global flag for suppressing growth?
            omega = copy.deepcopy(np.atleast_2d(self._omega_array[k]).T)
            classification = self._omega_classes[k]

            if suppress_growth:
                omega[omega.real > 0] = 1j * omega[omega.real > 0].imag

            c = np.atleast_2d(self._window_means_array[k]).T

            # Compute each segment of the reconstructed data starting at "t = 0"
            t = self._time_array[k]
            t_start = t.min()
            t = t - t_start

            xr_sep_window = np.zeros(
                (self._n_components, self._n_data_vars, self._window_length)
            )
            for j in np.unique(self._omega_classes):
                xr_sep_window[j, :, :] = np.linalg.multi_dot(
                    [
                        w[:, classification == j],
                        np.diag(b[classification == j]),
                        np.exp(omega[classification == j] * t),
                    ]
                ).real

                # Add the constant offset to the lowest frequency cluster.
                if include_means and (j == np.argmin(self._cluster_centroids)):
                    xr_sep_window[j, :, :] += c
                xr_sep_window[j, :, :] = xr_sep_window[j, :, :] * recon_filter

                xr_sep[j, :, window_indices] = (
                    xr_sep[j, :, window_indices] + xr_sep_window[j, :, :]
                )

            xn[window_indices] += recon_filter

        xr_sep = xr_sep / xn

        return xr_sep

    def scale_separation(
        self,
        scale_reconstruction_kwargs=None,
    ):
        """Separate the lowest frequency band from the high frequency bands.

        The lowest frequency band should contain the window means and can be passed on
        as the data for the next decomposition level. The high frequencies should have
        frequencies shorter than 1 / window_length.

        :param scale_reconstruction_kwargs: Arguments passed to `scale_reconstruction`
        :return xr_low_frequency: Reconstruction of the low frequency component with
            dimensions of n_data_vars x n_time_steps.
        :rtype xr_low_frequency: numpy.ndarray
        :return xr_high_frequency: Sum of all high frequency bands with dimensions of
            n_data_vars x n_time_steps
        :rtype xr_high_frequency: numpy.ndarray
        """

        if scale_reconstruction_kwargs is None:
            scale_reconstruction_kwargs = {}

        xr_sep = self.scale_reconstruction(**scale_reconstruction_kwargs)
        xr_low_frequency = xr_sep[0, :, :]
        xr_high_frequency = xr_sep[1:, :, :].sum(axis=0)

        return xr_low_frequency, xr_high_frequency

    def plot_scale_separation(
        self,
        data,
        scale_reconstruction_kwargs=None,
        plot_residual=False,
        fig_kwargs=None,
        plot_kwargs=None,
        hf_plot_kwargs=None,
        plot_contours=False,
    ):
        """Plot the scale-separated low and high frequency bands.

        The reconstructions are plotted in a time-space diagram. The high frequency
        component is the sum of all high frequency bands except the low frequency
        band which is plotted separately.

        :param data: Data used for the decomposition. An array of 1D snapshots.
        :type data: numpy.ndarray
        :param scale_reconstruction_kwargs: Arguments for reconstructing the COSTS fit.
        :type scale_reconstruction_kwargs: dict
        :param plot_residual: If the error should be fit. Will plot
            `data - low frequency - high frequency` yielding the error in absolute units.
        :type plot_residual: bool
        :param fig_kwargs: Arguments for the figure creation.
        :type fig_kwargs: dict
        :param plot_kwargs: Arguments for plotting the low frequency and data.
        :type plot_kwargs: dict
        :param hf_plot_kwargs: Arguments for the high frequency plotting.
        :type hf_plot_kwargs: dict
        :param plot_contours: Indicates if contours should be plotted.
        :type plot_contours: bool
        :return fig: figure handle for the plot
        :rtype fig: matplotlib.figure()
        :return axes: matplotlib subplot instances
        :rtype fig: matplotlib.Axes()
        """
        if scale_reconstruction_kwargs is None:
            scale_reconstruction_kwargs = {}

        xr_low_frequency, xr_high_frequency = self.scale_separation(
            scale_reconstruction_kwargs
        )

        if fig_kwargs is None:
            fig_kwargs = {}
        fig_kwargs["sharex"] = fig_kwargs.get("sharex", True)
        fig_kwargs["figsize"] = fig_kwargs.get("figsize", (6, 4))

        if plot_kwargs is None:
            plot_kwargs = {}
        plot_kwargs["vmin"] = plot_kwargs.get("vmin", -np.abs(data).max())
        plot_kwargs["vmax"] = plot_kwargs.get("vmax", np.abs(data).max())
        plot_kwargs["cmap"] = plot_kwargs.get("cmap", "cividis")

        if hf_plot_kwargs is None:
            hf_plot_kwargs = {}
        hf_plot_kwargs["vmin"] = hf_plot_kwargs.get(
            "vmin", -np.abs(xr_high_frequency).max()
        )
        hf_plot_kwargs["vmax"] = hf_plot_kwargs.get(
            "vmax", np.abs(xr_high_frequency).max()
        )
        hf_plot_kwargs["cmap"] = hf_plot_kwargs.get("cmap", "RdBu_r")

        if plot_residual:
            fig, axes = plt.subplots(4, 1, **fig_kwargs)
        else:
            fig, axes = plt.subplots(3, 1, **fig_kwargs)

        ax = axes[0]
        ax.pcolormesh(data, **plot_kwargs)
        if plot_contours:
            ax.contour(data, colors=["k"])
        ax.set_title(
            "Input Data at decomposition window length = {}".format(
                self._window_length
            )
        )
        ax.set_ylabel("Space (-)")

        ax = axes[1]
        ax.set_title("Reconstruction, low frequency")
        ax.pcolormesh(xr_low_frequency, **plot_kwargs)
        if plot_contours:
            ax.contour(data, colors=["k"])
        ax.set_ylabel("Space (-)")

        ax = axes[2]
        ax.set_title("Reconstruction, high frequency")
        ax.pcolormesh(xr_high_frequency, **hf_plot_kwargs)
        ax.set_ylabel("Space (-)")

        if plot_residual:
            ax = axes[3]
            ax.set_title("Residual")
            ax.pcolormesh(
                data - xr_high_frequency - xr_low_frequency, **hf_plot_kwargs
            )
            ax.set_ylabel("Space (-)")

        axes[-1].set_xlabel("Time (-)")
        fig.tight_layout()

        return fig, axes

    def plot_reconstructions(
        self,
        data,
        plot_period=False,
        scale_reconstruction_kwargs=None,
        plot_residual=False,
        fig_kwargs=None,
        plot_kwargs=None,
        hf_plot_kwargs=None,
        plot_contours=False,
    ):
        """Time-space plots for each individual frequency band and the fitted data.

        :param data: Data used for the decomposition. An array of 1D snapshots.
        :type data: numpy.ndarray
        :param plot_period:
        :param scale_reconstruction_kwargs: Arguments for reconstructing the COSTS fit.
        :type scale_reconstruction_kwargs: dict
        :param plot_residual: Indicates if the residual of the fit should be plotted
        :type plot_residual: bool
        :param fig_kwargs: Arguments for the figure creation.
        :type fig_kwargs: dict
        :param plot_kwargs: Arguments for plotting the low frequency and data.
        :type plot_kwargs: dict
        :param hf_plot_kwargs: Arguments for the high frequency plotting.
        :type hf_plot_kwargs: dict
        :param plot_contours: Indicates if contours should be plotted.
        :type plot_contours: bool
        :return fig: figure handle for the plot
        :rtype fig: matplotlib.figure()
        :return axes: matplotlib subplot instances
        :rtype fig: matplotlib.Axes()
        """
        if scale_reconstruction_kwargs is None:
            scale_reconstruction_kwargs = {}

        xr_sep = self.scale_reconstruction(**scale_reconstruction_kwargs)

        if fig_kwargs is None:
            fig_kwargs = {}
        fig_kwargs["sharex"] = fig_kwargs.get("sharex", True)
        fig_kwargs["figsize"] = fig_kwargs.get(
            "figsize", (6, 1.5 * len(self._cluster_centroids) + 1)
        )

        # Low frequency and input data often require separate plotting parameters.
        if plot_kwargs is None:
            plot_kwargs = {}
        plot_kwargs["vmin"] = plot_kwargs.get("vmin", -np.abs(data).max())
        plot_kwargs["vmax"] = plot_kwargs.get("vmax", np.abs(data).max())
        plot_kwargs["cmap"] = plot_kwargs.get("cmap", "cividis")

        # High frequency components often require separate plotting parameters.
        if hf_plot_kwargs is None:
            hf_plot_kwargs = {}
        hf_plot_kwargs["vmin"] = hf_plot_kwargs.get(
            "vmin", -np.abs(xr_sep[1:, :, :]).max()
        )
        hf_plot_kwargs["vmax"] = hf_plot_kwargs.get(
            "vmax", np.abs(xr_sep[1:, :, :]).max()
        )
        hf_plot_kwargs["cmap"] = hf_plot_kwargs.get("cmap", "RdBu_r")

        # Determine the number of plotting elements, which changes depending on if the
        # residual is included.
        if plot_residual:
            num_plot_elements = len(self._cluster_centroids) + 2
        else:
            num_plot_elements = len(self._cluster_centroids) + 1
        fig, axes = plt.subplots(
            num_plot_elements,
            1,
            **fig_kwargs,
        )

        ax = axes[0]
        ax.pcolormesh(data.real, **plot_kwargs)
        if plot_contours:
            ax.contour(data.real, colors=["k"])
        ax.set_ylabel("Space (-)")
        ax.set_xlabel("Time (-)")
        ax.set_title(
            "Input Data at decomposition window length = {}".format(
                self._window_length
            )
        )
        for n_cluster, cluster in enumerate(self._cluster_centroids):
            if plot_period:
                x = 2 * np.pi / cluster
                title = "Reconstruction, central period={:.2f}"
            else:
                x = cluster
                title = "Reconstruction, central eig={:.2f}"

            ax = axes[n_cluster + 1]
            xr_scale = xr_sep[n_cluster, :, :]
            if n_cluster == 0:
                ax.pcolormesh(xr_scale, **plot_kwargs)
                if plot_contours:
                    ax.contour(xr_scale, colors=["k"])
            else:
                ax.pcolormesh(xr_scale, **hf_plot_kwargs)
            ax.set_ylabel("Space (-)")
            ax.set_title(title.format(x))

        if plot_residual:
            ax = axes[-1]
            ax.set_title("Residual")
            ax.pcolormesh(data - xr_sep.sum(axis=0), **hf_plot_kwargs)
            ax.set_ylabel("Space (-)")

        axes[-1].set_xlabel("Time (-)")
        fig.tight_layout()

        return fig, axes

    def plot_error(
        self, data, scale_reconstruction_kwargs=None, plot_kwargs=None
    ):
        """Plots the error for the COSTS fit

        Plots are a space-time diagram assuming a 1D spatial dimension.

        Determining the error requires the input data and the fit will be
        reconstructed.

        :param data: Data on which COSTS was fit
        :type data: numpy.ndarray
        :param scale_reconstruction_kwargs: Arguments for reconstructing the fit.
        :type scale_reconstruction_kwargs: dict
        :param plot_kwargs: Arguments passed to costs.plot_error().
        :type scale_reconstruction_kwargs: dict
        :return:
        """
        if scale_reconstruction_kwargs is None:
            scale_reconstruction_kwargs = {}
        if plot_kwargs is None:
            plot_kwargs = {}
        plot_kwargs["vmin"] = plot_kwargs.get("vmin", -3)
        plot_kwargs["vmax"] = plot_kwargs.get("vmax", 3)
        plot_kwargs["cmap"] = plot_kwargs.get("cmap", "RdBu_r")

        global_reconstruction = self.global_reconstruction(
            scale_reconstruction_kwargs=scale_reconstruction_kwargs
        )

        fig_glbl_r, ax_glbl_r = plt.subplots(
            1,
            1,
        )
        im = ax_glbl_r.pcolormesh(
            (global_reconstruction.real - data.real) / data.real * 100,
            **plot_kwargs,
        )

        cbar = fig_glbl_r.colorbar(im)
        cbar.set_label("% error")

        ax_glbl_r.set_xlabel("time (-)")
        ax_glbl_r.set_ylabel("space (-)")
        re = self.relative_error(global_reconstruction.real, data)
        ax_glbl_r.set_title("Error in Global Reconstruction = {:.2}".format(re))

    def plot_time_series(
        self,
        space_index,
        data,
        scale_reconstruction_kwargs=None,
        include_residual=False,
    ):
        """Plots CoSTS for a single spatial point.

        Includes the input data for decomposition, the low-frequency component for the next
        decomposition level, the residual of the high frequency component, and the
        reconstructions of the frequency bands for the point.

        :param space_index: Index of the point in space for the 1D snapshot.
        :type space_index: int
        :param data: Original data, only necessary for level=0.
        :type data: numpy.ndarray
        :param scale_reconstruction_kwargs: Arguments for reconstructing the fit.
        :type scale_reconstruction_kwargs: dict
        :param include_residual:
        :return:
        """
        ground_truth_mean = data.mean(axis=1)
        ground_truth = (data.T - ground_truth_mean).T
        ground_truth = ground_truth[space_index, :]
        ground_truth_mean = ground_truth_mean[space_index]

        if scale_reconstruction_kwargs is None:
            scale_reconstruction_kwargs = {}
        xr_sep = self.scale_reconstruction(**scale_reconstruction_kwargs)

        # ToDo: Make these kwargs adjustable inputs.
        fig, axes = plt.subplots(
            nrows=self.n_components + 2,
            sharex=True,
            sharey=True,
            figsize=(8, 1.5 * self.n_components),
        )

        ax = axes[0]
        ax.plot(ground_truth, color="k")
        ax.plot(
            xr_sep.sum(axis=0)[space_index, :] - ground_truth_mean,
            color="r",
            lw=0.5,
        )
        ax.set_title(
            "window={}, black=input data, red=reconstruction".format(
                self._window_length
            )
        )
        ax.set_ylabel("Amp.")
        ax.set_xlabel("Time")

        ax = axes[1]
        ax.plot(
            ground_truth - (xr_sep[1:, :, :].sum(axis=0))[space_index, :],
            color="k",
        )

        for n in range(self.n_components):
            ax = axes[n + 1]
            if n == 0:
                title = "blue = Low-frequency component, black = high frequency residual"
                ax.plot(xr_sep[n, space_index, :] - ground_truth_mean)
            else:
                title = "Band period = {:.0f}s".format(
                    2 * np.pi / self.cluster_centroids[n]
                )
                ax.plot(xr_sep[n, space_index, :])
            ax.set_title(title)
            ax.set_ylabel("Amp.")
            ax.set_xlabel("Time")

        ax = axes[-1]
        ax.plot(ground_truth, color="k", label="Smoothed data")
        ax.plot(
            (xr_sep[1:, :, :].sum(axis=0))[space_index, :],
            label="High-frequency",
        )
        ax.plot(
            xr_sep[0, space_index, :] - ground_truth_mean, label="Low-frequency"
        )
        if include_residual:
            ax.plot(
                ground_truth - xr_sep.sum(axis=0)[space_index, :],
                label="Residual",
            )
            ax.set_title(
                "black=input data, yellow=low-frequency, blue=high-frequency, red=residual"
            )
        else:
            ax.set_title(
                "black=input data, yellow=low-frequency, blue=high-frequency"
            )

        ax.set_ylabel("Amp.")
        ax.set_xlabel("Time")
        ax.set_xlim(0, self._n_time_steps)
        fig.tight_layout()

        return fig, axes

    def to_xarray(self):
        """Build an xarray dataset from the fitted CoSTS object.

        The CoSTS object is converted to an xarray dataset, which allows
        saving the computationally expensive results, e.g., between iterations.

        The reconstructed data are not included since their size can rapidly
        explode to unexpected sizes. e.g., a 30MB dataset, decomposed at 6
        levels with an average number of frequency bands across decomposition
        levels equal to 8 becomes 1.3GB once reconstructed for each band.

        The functions `to_xarray` and `from_xarray` should allow for a complete
        round trip of the COSTS object without alteration.

        :return: COSTS fit in xarray format
        :rtype: xarray.Dataset
        """
        ds = xr.Dataset(
            {
                "omega": (("window_time_means", "svd_rank"), self.omega_array),
                "omega_classes": (
                    ("window_time_means", "svd_rank"),
                    self.omega_classes,
                ),
                "amplitudes": (
                    ("window_time_means", "svd_rank"),
                    self.amplitudes_array,
                ),
                "modes": (
                    ("window_time_means", "space", "svd_rank"),
                    self.modes_array,
                ),
                "window_means": (
                    ("window_time_means", "space"),
                    self.window_means_array,
                ),
                "cluster_centroids": (
                    "frequency_band",
                    self._cluster_centroids,
                ),
            },
            coords={
                "window_time_means": np.mean(self.time_array, axis=1),
                "slide": ("window_time_means", np.arange(self._n_slides)),
                "svd_rank": np.arange(self.svd_rank),
                "space": np.arange(self._n_data_vars),
                "frequency_band": np.arange(self.n_components),
                "window_index": np.arange(self._window_length),
                "time": (
                    ("window_time_means", "window_index"),
                    self.time_array,
                ),
            },
            attrs={
                "svd_rank": self.svd_rank,
                "omega_transformation": self._transform_method,
                "n_slides": self._n_slides,
                "window_length": self._window_length,
                "num_frequency_bands": self.n_components,
                "n_data_vars": self._n_data_vars,
                "n_time_steps": self._n_time_steps,
                "step_size": self._step_size,
                "non_integer_n_slide": self._non_integer_n_slide,
                "global_svd": self._global_svd,
            },
        )

        for kw, kw_val in self._pydmd_kwargs.items():
            ds.attrs["pydmd_kwargs__{}".format(kw)] = self._xarray_sanitize(
                kw_val
            )

        return ds

    def from_xarray(self, ds):
        """Convert xarray Dataset into a fitted CoSTS object.

        The functions `to_xarray` and `from_xarray` should allow for a complete
        round trip of the COSTS object without alteration.

        :return: Previously fitted COSTS object.
        """

        self._omega_array = ds.omega.values
        self._omega_classes = ds.omega_classes.values
        self._amplitudes_array = ds.amplitudes.values
        self._modes_array = ds.modes.values
        self._window_means_array = ds.window_means.values
        self._cluster_centroids = ds.cluster_centroids.values
        self._time_array = ds.time.values
        self._n_slides = ds.attrs["n_slides"]
        self._svd_rank = ds.attrs["svd_rank"]
        self._n_data_vars = ds.attrs["n_data_vars"]
        self._n_time_steps = ds.attrs["n_time_steps"]
        self._n_components = ds.attrs["num_frequency_bands"]
        self._non_integer_n_slide = ds.attrs["non_integer_n_slide"]
        self._step_size = ds.attrs["step_size"]
        self._window_length = ds.attrs["window_length"]
        self._global_svd = ds.attrs["global_svd"]

        self._pydmd_kwargs = {}
        for attr in ds.attrs:
            if "pydmd_kwargs" in attr:
                new_attr_name = attr.replace("pydmd_kwargs__", "")
                self._pydmd_kwargs[new_attr_name] = self._xarray_unsanitize(
                    ds.attrs[attr]
                )

        return self

    @staticmethod
    def _xarray_sanitize(value):
        """Handle Nones in the pydmd_kwargs (i.e., used default values)

        Netcdf cannot handle NoneTypes. To allow the xarray DataSet to be
        saved to file we have to "sanitize" the NoneTypes. These two functions
        allow for a round trip recovery of `pydmd_kwargs`.

        :param value: Value to be stored in the attributes of an xarray Dataset.
        :return: value unaltered except if value NoneType.
        """
        if value is None:
            value = "None"
        return value

    @staticmethod
    def _xarray_unsanitize(value):
        """Handle Nones in the pydmd_kwargs (i.e., used default values)

        Netcdf cannot handle NoneTypes. To allow the xarray DataSet to be
        saved to file we have to "sanitize" the NoneTypes. These two functions
        allow for a round trip recovery of `pydmd_kwargs`.

        :param value: Value stored in the attributes of an xarray Dataset.
        :return: value unaltered except if value is the string "None".
        """
        # To handle the varying behavior between python versions when evaluating
        # a mixed type statement we have to try to catch cases when `value` is
        # an array (e.g., `proj_basis` kwarg). The except block should not be
        # triggered but is meant to catch edge cases when the user provides
        # an unexpected type (e.g., tuple).
        if not hasattr(value, "shape"):
            try:
                if value == "None":
                    return None
            except ValueError:
                return value
        return value
