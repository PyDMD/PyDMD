import numpy as np
import copy
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import xarray as xr
from pydmd.costs import COSTS
import os


class mrCOSTS:
    """Multi-resolution Coherent Spatio-Temporal Scale Separation (mrCOSTS) with DMD.

    :param window_length_array: Length of the analysis window in number of time steps.
    :type window_length_array: list of int
    :param step_size_array: Number of time steps to slide each CSM-DMD window.
    :type step_size_array: list of int
    :param n_components_array: Number of frequency bands to use for clustering this
        window length.
    :type n_components_array: list of int
    :param svd_rank_array: The rank of the BOPDMD fit.
    :type svd_rank_array: list of int
    :param global_svd: Flag indicating whether to find the proj_basis and initial
        values using the entire dataset instead of individually for each window.
        Generally using the global_svd speeds up the fitting process by not finding a
        new initial value for each window. Default is True.
    :type global_svd: list of bool
    :param initialize_artificially: Flag indicating whether to initialize the DMD
        using imaginary eigenvalues (i.e., the imaginary component of the cluster
        results from a previous iteration) through the `cluster_centroids` keyword.
        Default is False.
    :type initialize_artificially: list of bool
    :param pydmd_kwargs: Keyword arguments to pass onto the BOPDMD object.
    :type pydmd_kwargs: dict
    :param cluster_centroids: Cluster centroids from a previous fitting iteration to
        use for the initial guess of the eigenvalues. Should only be the imaginary
        component.
    :type cluster_centroids: numpy array
    :param reset_alpha_init: Flag indicating whether the initial guess for the BOPDMD
        eigenvalues should be reset for each window. Resetting the initial value
        increases the computation time due to finding a new initial guess. Default
        is False.
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
    :param init_alpha: Initial guess for the eigenvalues provided to BOPDMD.
        Must be equal to the `svd_rank`.
    :type init_alpha: numpy array
    :param max_rank: Maximum allowed `svd_rank`. Overrides the optimal rank truncation
        if `svd_rank=0`.
    :type max_rank: int
    :param force_even_eigs: Flag specifying if the `svd_rank` should be forced to
        be even.
    :type force_even_eigs: bool
    :param reset_alpha_init: Flag specifying if the initial eigenvalue guess should
        be reset between windows.
    :type reset_alpha_init: bool
    """

    def __init__(
        self,
        window_length_array=None,
        step_size_array=None,
        svd_rank_array=None,
        global_svd_array=None,
        initialize_artificially=False,
        use_last_freq=False,
        use_kmean_freqs=False,
        init_alpha=None,
        pydmd_kwargs=None,
        costs_recon_kwargs=None,
        cluster_centroids=None,
        reset_alpha_init=False,
        force_even_eigs=True,
        max_rank=None,
        n_components_array=None,
        cluster_sweep=False,
        transform_method=None,
        store_data=True,
    ):
        self._store_data = store_data
        self._n_components_array = n_components_array
        self._step_size_array = step_size_array
        self._window_length_array = window_length_array
        self._svd_rank_array = svd_rank_array
        self._global_svd_array = global_svd_array
        self._initialize_artificially = initialize_artificially
        self._use_last_freq = use_last_freq
        self._use_kmean_freqs = use_kmean_freqs
        self._init_alpha = init_alpha
        self._cluster_centroids = cluster_centroids
        self._reset_alpha_init = reset_alpha_init
        self._force_even_eigs = force_even_eigs
        self._max_rank = max_rank
        self._cluster_sweep = cluster_sweep
        self._transform_method = transform_method

        # ToDo: Check for cluster_sweep and n_components_array

        # Initialize variables that are defined in fitting.
        self._n_decompositions = None
        self._n_data_vars = None
        self._n_time_steps = None
        self._cluster_centroids = None
        self._omega_classes = None
        self._costs_array = None
        self._da_omega = None

        # Specify default keywords to hand to CoSTS's BOPDMD model.
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

        if costs_recon_kwargs is None:
            self._costs_recon_kwargs = {
                "suppress_growth": False,
            }
        else:
            self._costs_recon_kwargs = costs_recon_kwargs
            self._costs_recon_kwargs[
                "suppress_growth"
            ] = costs_recon_kwargs.get("suppress_growth", False)

    @property
    def store_data(self):
        """
        :return: If the low-frequency components were stored (True)
            or discarded (False).
        :rtype: bool
        """
        return self._store_data

    @property
    def costs_array(self):
        """
        :return: costs objects for each decomposition level.
        :rtype: list
        """
        return self._costs_array

    @costs_array.setter
    def costs_array(self, costs_array):
        self._costs_array = costs_array

    @property
    def svd_rank_array(self):
        """
        :return: the rank used for the svd truncation.
        :rtype: int or float
        """
        return self._svd_rank_array

    @property
    def window_length_array(self):
        """
        :return: the length of the windows used for each decomposition level.
        :rtype: list of int or float
        """
        return self._window_length_array

    @property
    def step_size_array(self):
        """
        :return: the length of the windows used for each decomposition level.
        :rtype: list of int or float
        """
        return self._step_size_array

    @property
    def n_decompositions(self):
        """
        :return: The number of multi-resolution decompositions to perform.
        :rtype: int
        """
        return self._n_decompositions

    @property
    def transform_method(self):
        """
        :return: How to transform the eigenvalues for clustering.
        :rtype: string
        """
        return self._transform_method

    @property
    def n_components_array(self):
        """
        :return: the number of frequency bands used for each decomposition level.
        :rtype: list of int or float
        """
        return self._n_components_array

    @property
    def modes_array(self):
        if not hasattr(self, "_modes_array"):
            raise ValueError("You need to call fit before")
        return self._modes_array

    @property
    def cluster_centroids(self):
        if not hasattr(self, "_cluster_centroids"):
            raise ValueError("You need to call `cluster_omega()` first.")
        return self._cluster_centroids

    @property
    def omega_classes(self):
        if not hasattr(self, "_omega_classes"):
            raise ValueError("You need to call `cluster_omega()` first.")
        return self._omega_classes

    @staticmethod
    def _data_shape(data):
        n_time_steps = np.shape(data)[1]
        n_data_vars = np.shape(data)[0]
        return n_time_steps, n_data_vars

    def fit(self, data, time, verbose=True):
        window_lengths = self._window_length_array
        step_sizes = self._step_size_array
        svd_ranks = self._svd_rank_array
        self._n_decompositions = len(window_lengths)
        n_decompositions = self._n_decompositions
        transform_method = self._transform_method

        # Set the global_svd flag if none was provided.
        if self._global_svd_array is None:
            self._global_svd_array = [True] * n_decompositions

        self._costs_array = []
        self._n_time_steps, self._n_data_vars = self._data_shape(data)

        if self._store_data:
            data_iter = np.zeros(
                (n_decompositions, self._n_data_vars, self._n_time_steps)
            )
            data_iter[0, :, :] = data
        else:
            data_iter = data

        for n_decomp, (window, step, rank) in enumerate(
            zip(window_lengths, step_sizes, svd_ranks)
        ):
            global_svd = self._global_svd_array[n_decomp]

            if self._store_data:
                x_iter = data_iter[n_decomp, :, :].squeeze()
            else:
                x_iter = data_iter.squeeze()

            mrd = COSTS(
                svd_rank=rank,
                global_svd=global_svd,
                pydmd_kwargs=self._pydmd_kwargs,
            )

            if verbose:
                print("_________________________________________________")
                print("Fitting window length = {}".format(window))
            mrd.fit(x_iter, np.atleast_2d(time), window, step, verbose=verbose)

            # Cluster the frequency bands
            if self._cluster_sweep or np.isnan(
                self._n_components_array[n_decomp]
            ):
                n_components = mrd.cluster_hyperparameter_sweep(
                    transform_method=transform_method
                )
                self._n_components_array[n_decomp] = n_components
            else:
                n_components = self._n_components_array[n_decomp]

            _ = mrd.cluster_omega(
                n_components=n_components, transform_method=transform_method
            )

            # Global reconstruction
            if verbose:
                global_reconstruction = mrd.global_reconstruction(
                    scale_reconstruction_kwargs=self._costs_recon_kwargs,
                )
                re = mrd.relative_error(global_reconstruction.real, x_iter)
                print("Error in Global Reconstruction = {:.2}".format(re))

            # Scale separation
            xr_low_frequency, xr_high_frequency = mrd.scale_separation(
                scale_reconstruction_kwargs=self._costs_recon_kwargs
            )

            # Pass the low frequency component to the next level of decomposition.
            if n_decomp < n_decompositions - 1:
                if self._store_data:
                    data_iter[n_decomp + 1, :, :] = xr_low_frequency
                else:
                    data_iter = xr_low_frequency

            # Save the object for later use.
            self._costs_array.append(copy.copy(mrd))

    def multi_res_interp(
        self,
    ):
        ds_list = [c.to_xarray() for c in self._costs_array]
        # Remove the low-frequency bands.
        da_to_concat = [
            ds_list[0].omega.where(ds_list[0].omega_classes > 0, drop=True)
        ]

        # Interpolate the larger decomposition levels to the timestep of the
        # smallest decomposition level.
        for ds in ds_list[1:]:
            da = ds.omega.where(ds.omega_classes > 0, drop=True)
            da = da.interp(
                window_time_means=ds_list[0].window_time_means,
                method="nearest",
                kwargs={"fill_value": "extrapolate"},
            )
            da_to_concat.append(da)

        da_omega = xr.concat(
            da_to_concat,
            dim="window_length",
        )
        da_omega.coords["window_length"] = self._window_length_array
        da_omega.coords["decomposition_level"] = (
            "window_length",
            np.arange(len(da_omega.window_length)),
        )

        # ToDo: Verify this can handle variable number of svd_ranks
        #   for each decomposition level

        self._da_omega = da_omega

    def from_xarray(self, file_list):
        """Create an mrcosts object from saved files."""
        mrd_list = []
        # ToDo: Check for file existence.
        for f in file_list:
            mrd_list.append(xr.load_dataset(f, engine="h5netcdf"))

        # Sort by window length
        mrd_list = sorted(mrd_list, key=lambda mrd: mrd.window_length)

        # Convert to an array of costs objects.
        mrd_list = [COSTS().from_xarray(mrd) for mrd in mrd_list]

        window_length_array = [mrd.window_length for mrd in mrd_list]
        step_size_array = [mrd.step_size for mrd in mrd_list]
        svd_rank_array = [mrd.svd_rank for mrd in mrd_list]
        # ToDo: Don't access protected information.
        n_components_array = [mrd._n_components for mrd in mrd_list]
        global_svd_array = [mrd._global_svd for mrd in mrd_list]

        # ToDo: Find a more robust way of handling these cases.
        # A simple stop-gap to use the kwargs simply from the first element.
        pydmd_kwargs = mrd_list[0]._pydmd_kwargs
        # costs_recon_kwargs = mrd_list[0]._costs_recon_kwargs

        # Initialize the mrcosts object.
        self.__init__(
            window_length_array=window_length_array,
            step_size_array=step_size_array,
            svd_rank_array=svd_rank_array,
            global_svd_array=global_svd_array,
            pydmd_kwargs=pydmd_kwargs,
            # costs_recon_kwargs=costs_recon_kwargs,
            n_components_array=n_components_array,
            store_data=False,
        )

        # Initialize variables that are defined in fitting.
        self._n_decompositions = len(mrd_list)
        self.costs_array = mrd_list

    def scale_separation(self, level, data=None, plot_kwargs=None):
        # Plotting reconstructions is a bit complicated because it requires
        # indexing the desired level and the previous level.

        if level == 0 and data is None:
            raise ValueError(
                "Data must be provided when plotting the first decomposition level"
            )
        elif level == 0 and data is not None:
            x_iter = data
        else:
            x_iter, _ = self.costs_array[level - 1].scale_separation()

        if plot_kwargs is None:
            plot_kwargs = {}

        _ = self.costs_array[level].plot_scale_separation(
            x_iter, plot_kwargs=plot_kwargs
        )

    def cluster_hyperparameter_sweep(
        self, n_components_range, method=None, score_method=None
    ):
        """
        Hyperparameter search for n_components for kmeans clustering.

        score_method: str or None
            Valid scoring methods are 'silhouette' and 'calinski-harabasz'. Default is
            the silhouette score, which can be slow for large numbers of samples.
        """
        score = np.zeros_like(n_components_range, float)

        for nind, n in enumerate(n_components_range):
            print("fitting n_components = {}".format(n))
            cluster_centroids, omega_classes, omega = self.cluster_omega(
                n, method=method
            )

            print("scoring")
            if score_method is None or score_method == "silhouette":
                score[nind] = silhouette_score(
                    omega.reshape(-1, 1),
                    omega_classes.reshape(-1, 1),
                    n_jobs=-2,
                )
            elif score_method == "calinski-harabasz":
                score[nind] = calinski_harabasz_score(
                    omega.reshape(-1, 1),
                    omega_classes.reshape(-1, 1),
                )

        return score, n_components_range[np.argmax(score)]

    def cluster_omega(self, n_components, method=None, kmeans_kwargs=None):
        omega_array = self._da_omega.values
        # This step flattens the array, which is desirable. We want to cluster on just
        # on the histogram of omega, which means just one "sample".
        omega_array = omega_array[~np.isnan(omega_array)]
        omega_array = self.transform_omega(omega_array, method=method)

        if kmeans_kwargs is None:
            random_state = 0
            kmeans_kwargs = {
                "n_init": "auto",
                "random_state": random_state,
            }
        kmeans = MiniBatchKMeans(n_clusters=n_components, **kmeans_kwargs)
        omega_classes = kmeans.fit_predict(np.atleast_2d(omega_array).T)
        cluster_centroids = kmeans.cluster_centers_.flatten()

        # Sort the clusters by the centroid magnitude.
        idx = np.argsort(cluster_centroids)
        lut = np.zeros_like(idx)
        lut[idx] = np.arange(n_components)
        omega_classes = lut[omega_classes]
        cluster_centroids = cluster_centroids[idx]

        return cluster_centroids, omega_classes, omega_array

    @staticmethod
    def transform_omega(omega_array, method=None):
        if method is None:
            omega_array = np.abs(omega_array.imag.astype("float"))
        elif method == "square_frequencies":
            omega_array = (np.conj(omega_array) * omega_array).astype("float")
        elif method == "period":
            omega_array = 1 / np.abs(omega_array.imag.astype("float"))
        elif method == "log10":
            omega_array = np.log10(np.abs(omega_array.imag.astype("float")))

        return omega_array

    def global_scale_reconstruction(
        self,
        n_components,
        omega_classes_array,
        suppress_growth=True,
        include_means=False,
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
            (
                len(self._costs_array),
                n_components,
                self._n_data_vars,
                self._n_time_steps,
            )
        )

        for n_mrd, mrd in enumerate(self._costs_array):
            # Track the total contribution from all windows to each time step
            xn = np.zeros(self._n_time_steps)

            # Convolve each windowed reconstruction with a gaussian filter.
            # Std dev of gaussian filter
            recon_filter_sd = mrd._window_length / 8
            recon_filter = np.exp(
                -(
                    (
                        np.arange(mrd._window_length)
                        - (mrd._window_length + 1) / 2
                    )
                    ** 2
                )
                / recon_filter_sd**2
            )
            omega_classes = omega_classes_array[n_mrd, :, :].squeeze()

            for k in range(mrd._n_slides):
                w = mrd._modes_array[k]
                b = mrd._amplitudes_array[k]
                omega = np.atleast_2d(mrd._omega_array[k]).T
                classification = omega_classes[k]

                if suppress_growth:
                    omega[omega.real > 0] = 1j * omega[omega.real > 0].imag

                # Compute each segment of xr starting at "t = 0"
                t = mrd._time_array[k]
                t_start = mrd._time_array[k, 0]
                t = t - t_start

                xr_sep_window = np.zeros(
                    (n_components, mrd._n_data_vars, mrd._window_length)
                )
                for j in np.arange(0, n_components):
                    xr_sep_window[j, :, :] = np.linalg.multi_dot(
                        [
                            w[:, classification == j],
                            np.diag(b[classification == j]),
                            np.exp(omega[classification == j] * t),
                        ]
                    )

                    xr_sep_window[j, :, :] = (
                        xr_sep_window[j, :, :] * recon_filter
                    )

                    if k == mrd._n_slides - 1 and mrd._non_integer_n_slide:
                        window_indices = slice(-mrd._window_length, None)
                    else:
                        window_indices = slice(
                            k * mrd._step_size,
                            k * mrd._step_size + mrd._window_length,
                        )
                    xr_sep[n_mrd, j, :, window_indices] = (
                        xr_sep[n_mrd, j, :, window_indices]
                        + xr_sep_window[j, :, :]
                    )

                xn[window_indices] += recon_filter

            xr_sep[n_mrd, :, :, :] = xr_sep[n_mrd, :, :, :] / xn

        return xr_sep
