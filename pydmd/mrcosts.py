"""
Module for the multi-resolution Coherent Spatio-Temporal Scale Separation with
DMD.

References:
- Dylewsky, D., Tao, M., & Kutz, J. N. (2019). Dynamic mode decomposition for
multiscale nonlinear physics. Physics Review E, 99(6),
10.1103/PhysRevE.99.063311. https://doi.org/10.1103/PhysRevE.99.063311
"""

import os
import copy
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import xarray as xr
from pydmd.costs import COSTS


class mrCOSTS:
    """Multi-resolution Coherent Spatio-Temporal Scale Separation (mrCOSTS)
    with DMD.

    :param window_length_array: Length of the analysis window in number of time
        steps.
    :type window_length_array: list of int
    :param step_size_array: Number of time steps to slide each CSM-DMD window.
    :type step_size_array: list of int
    :param n_components_array: Number of frequency bands to use for clustering
        each decomposition level. Only one of `cluster_sweep` or
        `n_components_array` should be provided.
    :type n_components_array: list of int
    :param svd_rank_array: The rank of the BOPDMD fit.
    :type svd_rank_array: list of int
    :param global_svd_array: Flag indicating whether to find the proj_basis and
        initial values using the entire dataset instead of individually for
        each window. Generally using the global_svd speeds up the fitting
        process by not finding a new initial value for each window. Default
        is True.
    :type cluster_sweep: bool
    :param cluster_sweep: Flag indicating whether to find the optimal value
        of `n_clusters` for the clustering of fitted eigenvalues. Only one of
        `cluster_sweep` or `n_components_array` should be provided.
    :type global_svd_array: list of bool
    :param pydmd_kwargs: Keyword arguments to pass onto the BOPDMD object.
    :type pydmd_kwargs: dict
    """

    def __init__(
        self,
        window_length_array=None,
        step_size_array=None,
        svd_rank_array=None,
        global_svd_array=None,
        pydmd_kwargs=None,
        costs_recon_kwargs=None,
        n_components_array=None,
        cluster_sweep=False,
        transform_method=None,
        kern_method=None,
        relative_filter_length=2,
    ):
        self._n_components_array = n_components_array
        self._step_size_array = step_size_array
        self._window_length_array = window_length_array
        self._svd_rank_array = svd_rank_array
        self._global_svd_array = global_svd_array
        self._cluster_sweep = cluster_sweep
        self._transform_method = transform_method
        self._kern_method = kern_method
        self._relative_filter_length = relative_filter_length

        if (self._n_components_array is not None) and (
            self._step_size_array is not None
        ):
            if not len(self._n_components_array) == len(self._step_size_array):
                raise ValueError(
                    (
                        "n_components_array and step_size_array must be the"
                        "same length."
                    )
                )

        # Initialize variables that are defined in fitting.
        self._n_decompositions = None
        self._n_data_vars = None
        self._n_time_steps = None
        self._omega_classes = None
        self._costs_array = None
        self._da_omega = None
        self._n_components_global = None
        self._cluster_centroids = None

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
            self._costs_recon_kwargs = {}
        else:
            self._costs_recon_kwargs = costs_recon_kwargs

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
        :return: the number of frequency bands used for each decomposition
            level.
        :rtype: list of int or float
        """
        return self._n_components_array

    @property
    def cluster_centroids(self):
        """
        :return: Cluster centroids from clustering of eigenvalues.
        :rtype: list of float
        """
        if self._cluster_centroids is None:
            raise ValueError("You need to call `cluster_omega()` first.")
        return self._cluster_centroids

    @property
    def n_components_global(self):
        """
        :return: Number of global frequency bands
        :rtype: list of float
        """
        if self._n_components_global is None:
            raise ValueError(
                "You need to call `global_cluster_hyperparameter_sweep()` "
                "first or assign a the value directly."
            )
        return self._n_components_global

    # @ToDo: Use the class variable instead of passing it around
    @property
    def omega_classes_interpolated(self):
        """Returns the multi-resolution interpolation of omega classes

        :return: Ints for each omega value indicating which cluster it
            belongs to.
        :rtype: list of numpy.ndarray
        """
        if self._omega_classes is None:
            raise ValueError("You need to call `cluster_omega()` first.")
        return self._omega_classes

    @property
    def ragged_omega_classes(self):
        """Omega classes for each decomposition level after global clustering.

        :return: list of classes for each omega value for each decomposition
            level.
        :rtype: list of numpy.ndarray
        """
        if self._omega_classes is None:
            raise ValueError("You need to call `cluster_omega()` first.")
        return self.multi_res_deterp()

    @property
    def ragged_omega_array(self):
        """Omega values for each decomposition level.

        :return: list of omega arrays for each decomposition level.
        :rtype: list of numpy.ndarray
        """
        if self._costs_array is None:
            raise ValueError(
                "You need to `fit` or load previous fit from file first."
            )
        return [c.omega_array for c in self._costs_array]

    @property
    def ragged_modes_array(self):
        """Modes for each decomposition level.

        :return: list of modes arrays for each decomposition level.
        :rtype: list of numpy.ndarray
        """
        if self._costs_array is None:
            raise ValueError(
                "You need to `fit` or load previous fit from file first."
            )
        return [c.modes_array for c in self._costs_array]

    @property
    def ragged_amplitudes_array(self):
        """Amplitudes for each decomposition level.

        :return: list of amplitudes arrays for each decomposition level.
        :rtype: list of numpy.ndarray
        """
        if self._costs_array is None:
            raise ValueError(
                "You need to `fit` or load previous fit from file first."
            )
        return [c.amplitudes_array for c in self._costs_array]

    @staticmethod
    def _data_shape(data):
        """Give the data shape.

        :return: Shape of the data for fitting.
        :rtype: Tuple of ints
        """
        n_time_steps = np.shape(data)[1]
        n_data_vars = np.shape(data)[0]
        return n_time_steps, n_data_vars

    def fit(self, data, time, verbose=True):
        """
        Compute mrCOSTS for the given data.

        :param data: the input snapshots.
        :type data: numpy.ndarray
        :param time: the input time vector.
        :type time: numpy.ndarray
        :param verbose: If mrCOSTS should report progress.
        :type verbose: bool
        """
        window_lengths = self._window_length_array
        step_sizes = self._step_size_array
        svd_ranks = self._svd_rank_array
        self._n_decompositions = len(self._window_length_array)
        n_decompositions = self._n_decompositions
        transform_method = self._transform_method

        # Check for the n_components array and cluster sweeping.
        if self._cluster_sweep:
            if self._n_components_array is not None:
                raise ValueError(
                    (
                        "Only one of `cluster_sweep` and `n_components_array` "
                        "can be provided."
                    )
                )
            self._n_components_array = np.zeros(self._n_decompositions) * np.nan

        # Set the global_svd flag if none was provided.
        if self._global_svd_array is None:
            self._global_svd_array = [True] * n_decompositions

        self._costs_array = []
        self._n_time_steps, self._n_data_vars = self._data_shape(data)

        x_iter = data

        for n_decomp, (window, step, rank) in enumerate(
            zip(window_lengths, step_sizes, svd_ranks)
        ):
            global_svd = self._global_svd_array[n_decomp]

            mrd = COSTS(
                svd_rank=rank,
                global_svd=global_svd,
                pydmd_kwargs=self._pydmd_kwargs,
                kern_method=self._kern_method,
                relative_filter_length=self._relative_filter_length,
            )

            if verbose:
                print("_________________________________________________")
                print(f"Fitting window length = {window:}")
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

            mrd.cluster_omega(
                n_components=n_components, transform_method=transform_method
            )

            # Global reconstruction
            if verbose:
                global_reconstruction = mrd.global_reconstruction(
                    scale_reconstruction_kwargs=self._costs_recon_kwargs,
                )
                re = mrd.relative_error(global_reconstruction.real, x_iter)
                print(f"Error in Global Reconstruction = {re:.2}")

            # Pass the low frequency component to the next level of
            # decomposition.
            if n_decomp < n_decompositions - 1:
                # Scale separation
                xr_low_frequency, _ = mrd.scale_separation(
                    scale_reconstruction_kwargs=self._costs_recon_kwargs
                )
                x_iter = xr_low_frequency

            # Save the fitted costs object.
            self._costs_array.append(copy.copy(mrd))

    @staticmethod
    def interp_fill(
        da_interp,
        da,
    ):
        """Extrapolate values to the full data window.

        Since multi_res_interp uses a nearest neighbors interpolation without
        exraploation and the window_time_means are in the middle of the
        decomposition window, the first and last windows are incompletely
        filled. This function fills the NaN entries of these windows.

        :param da_interp: Interpolated eigenvalues following multi_res_interp()
        :type da_interp: xarray.DataArray
        :param da: Uninterpolated eigenvalues
        :type da: xarray.DataArray
        :return: Interpolated eigenvalues with filled first and last windows.
        :rtype: xarray.DataArray
        """
        window_delta = da.window_time_means.diff(
            dim="window_time_means"
        ).values[0]

        # Backwards fill the last non-NaN to the beginning of the first window
        first_window = da_interp.dropna(
            dim="window_time_means", how="all"
        ).isel(window_time_means=0)

        da_interp = xr.where(
            (
                da_interp.window_time_means
                > da.window_time_means[0] - window_delta / 2
            )
            & (da_interp.window_time_means < da.window_time_means[0]),
            first_window,
            da_interp,
        )

        # Forward fill the last non-NaN to the end of the last window
        last_window = da_interp.dropna(dim="window_time_means", how="all").isel(
            window_time_means=-1
        )

        da_interp = xr.where(
            (
                da_interp.window_time_means
                < da.window_time_means[-1] + window_delta / 2
            )
            & (da_interp.window_time_means > da.window_time_means[-1]),
            last_window,
            da_interp,
        )

        return da_interp

    def multi_res_interp(
        self,
    ):
        """
        Interpolate the mrCOSTS eigenvalues to the smallest decomposition level.

        :return: Interpolated mrCOSTS eigenvalues
        :rtype: xarray DataArray
        """
        ds_list = [c.to_xarray() for c in self._costs_array]
        # Remove the low-frequency bands.
        da_to_concat = [
            ds_list[0].omega.where(ds_list[0].omega_classes > 0, drop=True)
        ]

        # Interpolate the larger decomposition levels to the timestep of the
        # smallest decomposition level.

        # Previously there was a mistake in the frequencies squared omega
        # transformation which made this step particularly likely to fail
        # in difficult to discover ways. The interpolation was over-engineered
        # prior to discovering this mistake, but is more explicit than the
        # previous version.
        for ds in ds_list[1:]:
            da = ds.omega.where(ds.omega_classes > 0, drop=True)
            da_real = da.real
            da_imag = da.imag

            da_real = da_real.interp(
                window_time_means=ds_list[0].window_time_means,
                method="nearest",
            )

            da_imag = da_imag.interp(
                window_time_means=ds_list[0].window_time_means,
                method="nearest",
            )

            # Complete the interpolation of the first and last window
            da_imag = self.interp_fill(da_imag, da)
            da_real = self.interp_fill(da_real, da)

            da = da_real + 1j * da_imag
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

        self._da_omega = da_omega

    def multi_res_deterp(self):
        """
        Un-interpolate the mrCOSTS eigenvalues to the original spacing.

        :return: Omega classes from global clustering.
        :rtype: list of numpy.ndarrays
        """

        # Get the indices for the 3-d omega structure
        index = np.nonzero(~np.isnan(self._da_omega.values))

        # Unravel the flattened omega_classes. The class value of `-1` refers
        # to the slowest mode in each decomposition window which shouldn't be
        # included in the global clustering.
        omega_classes_full = (
            np.zeros_like(self._da_omega.values, dtype="int") - 1
        )
        omega_classes_full[index] = self._omega_classes

        # Build the omega_classes array into a labeled xarray DataArray object.
        da_omega_classes = xr.zeros_like(self._da_omega, dtype="int")
        da_omega_classes.values = omega_classes_full
        da_omega_classes = da_omega_classes.swap_dims(
            {"window_length": "decomposition_level"}
        )

        # Get a list of the costs results in xarray format.
        ds_list = [c.to_xarray() for c in self._costs_array]

        # Interpolate from the high-resolution of the first decomposition level
        # to the coarser time resolution of the higher decomposition levels
        omega_classes_list = [
            da_omega_classes.sel(decomposition_level=d)
            .sel(
                window_time_means=ds_list[d].window_time_means, method="nearest"
            )
            .values
            for d in da_omega_classes.decomposition_level.values
        ]

        return omega_classes_list

    def from_netcdf(self, file_list):
        """
        Create an mrCoSTS object from saved netcdf files.

        :param file_list: Filenames to open.
        :type file_list: list of str
        """
        mrd_list = []
        for f in file_list:
            if os.path.isfile(f):
                mrd_list.append(xr.load_dataset(f, engine="h5netcdf"))
            else:
                raise ValueError(f"{f:} was not found.")
        # Sort by window length
        mrd_list = sorted(mrd_list, key=lambda mrd: mrd.window_length)

        # Populate information about the fitted data.
        n_data_vars = mrd_list[0].attrs["n_data_vars"]
        n_time_steps = mrd_list[0].attrs["n_time_steps"]

        # Convert to an array of costs objects.
        mrd_list = [COSTS().from_xarray(mrd) for mrd in mrd_list]

        window_length_array = [mrd.window_length for mrd in mrd_list]
        step_size_array = [mrd.step_size for mrd in mrd_list]
        svd_rank_array = [mrd.svd_rank for mrd in mrd_list]
        n_components_array = [mrd.n_components for mrd in mrd_list]
        global_svd_array = [mrd.global_svd for mrd in mrd_list]

        # mrCOSTS currently does not support variable pydmd_kwargs
        # for each level.
        pydmd_kwargs = mrd_list[0]._pydmd_kwargs

        # Initialize the mrcosts object.
        self.__init__(
            window_length_array=window_length_array,
            step_size_array=step_size_array,
            svd_rank_array=svd_rank_array,
            global_svd_array=global_svd_array,
            pydmd_kwargs=pydmd_kwargs,
            n_components_array=n_components_array,
            relative_filter_length=mrd_list[0]._relative_filter_length,
            kern_method=mrd_list[0]._kern_method,
        )

        # Initialize variables that are defined in fitting.
        self._n_decompositions = len(mrd_list)
        self.costs_array = mrd_list
        self._n_data_vars = n_data_vars
        self._n_time_steps = n_time_steps

    def to_netcdf(self, filename, filepath="."):
        """
        Save the mrCoSTS fit to file in netcdf format.

        Each decomposition level is saved as a separate file with a common
        name and an identifier for the decomposition level.

        :param filename: Common name shared by each file.
        :type filename: str
        :param filepath: Path to save the results. Default is the current
        directory.
        :type filename: str

        """
        for c in self._costs_array:
            fname = ".".join(
                (
                    filename,
                    f"window={c.window_length:}",
                    "nc",
                )
            )
            fpath = os.path.join(filepath, fname)
            c.to_xarray().to_netcdf(
                fpath,
                engine="h5netcdf",
                invalid_netcdf=True,
            )

    def _plot_helper_data_check(self, level, data=None):
        """Checks the input data for plotting.

        :param level: Decomposition level to plot (zero indexed).
        :type level: int
        :param data: Original data, only necessary for level=0.
        :type data: numpy.ndarray
        :return: Data for plotting
        :rtype: numpy.ndarray
        """
        if data is not None:
            x_iter = data
        else:
            if level == 0:
                raise ValueError(
                    (
                        "Data must be provided when plotting the first "
                        "decomposition level"
                    )
                )
            if level > 0:
                x_iter, _ = self.costs_array[level - 1].scale_separation()

        if not x_iter.shape == (self._n_data_vars, self._n_time_steps):
            raise ValueError("Input data has the wrong shape.")

        return x_iter

    def plot_local_reconstructions(
        self,
        level,
        data=None,
        kwargs=None,
        scale_reconstruction_kwargs=None,
    ):
        """Plot reconstruction of each frequency band of a decomposition level.

        Plots are space-time diagrams assuming a 1D spatial dimension. These
        are the local frequency band clusters, not the global clusters.

        Requires the input data for the decomposition as well as
        reconstructing the fit. Deriving the input data requires providing
        the actual input data for the first decomposition. Otherwise,
        the input data is recovered by reconstructing
        decomposition = level - 1.

        :param level: Decomposition level to plot (zero indexed).
        :type level: int
        :param data: Original data, only necessary for level=0.
        :type data: numpy.ndarray
        :param kwargs: Keyword arguments given to costs.plot_reconstruction()
        :type kwargs: dict
        :param scale_reconstruction_kwargs: Arguments for reconstructing the
            fit.
        :type scale_reconstruction_kwargs: dict
        :return: None
        """

        x_iter = self._plot_helper_data_check(level, data=data)

        if kwargs is None:
            kwargs = {}

        _ = self.costs_array[level].plot_reconstructions(
            x_iter,
            scale_reconstruction_kwargs=scale_reconstruction_kwargs,
            **kwargs,
        )

    def plot_local_error(
        self,
        level,
        data=None,
        scale_reconstruction_kwargs=None,
        plot_kwargs=None,
    ):
        """Plots the error for the local decomposition fit

        Plots are a space-time diagram assuming a 1D spatial dimension.

        Determining the error requires the input data for the decomposition
        as well as reconstructing the fit. Deriving the input data requires
        providing the actual input data for the first decomposition.
        Otherwise, the input data is recovered by reconstructing
        decomposition = level - 1.

        Error is expressed as a percent.

        :param level: Decomposition level for plotting
        :type level: int
        :param data: Original data, only necessary for level=0.
        :type data: numpy.ndarray
        :param scale_reconstruction_kwargs: Arguments for reconstructing the
            fit.
        :type scale_reconstruction_kwargs: dict
        :param plot_kwargs: Arguments passed to costs.plot_error().
        :type scale_reconstruction_kwargs: dict
        :return:
        """

        x_iter = self._plot_helper_data_check(level, data=data)

        if plot_kwargs is None:
            plot_kwargs = {}

        _ = self.costs_array[level].plot_error(
            x_iter,
            scale_reconstruction_kwargs=scale_reconstruction_kwargs,
            plot_kwargs=plot_kwargs,
        )

    def plot_local_scale_separation(
        self,
        level,
        data=None,
        plot_kwargs=None,
        scale_reconstruction_kwargs=None,
    ):
        """Plot local scale separation of the high and low frequency components.

        Requires the input data for the decomposition as well as
        reconstructing the fit. Deriving the input data requires providing
        the actual input data for the first decomposition. Otherwise,
        the input data is recovered by reconstructing decomposition = level - 1.

        :param level: Decomposition level for plotting
        :type level: int
        :param data: Original data, only necessary for level=0.
        :param scale_reconstruction_kwargs: Arguments for reconstructing the
            fit.
        :param plot_kwargs: Arguments passed to costs.plot_scale_separation()
        :return fig: figure handle for the plot
        :rtype fig: matplotlib.figure()
        :return axes: matplotlib subplot instances
        :rtype fig: matplotlib.Axes()
        """

        x_iter = self._plot_helper_data_check(level, data=data)

        if plot_kwargs is None:
            plot_kwargs = {}

        fig, axes = self.costs_array[level].plot_scale_separation(
            x_iter,
            plot_kwargs=plot_kwargs,
            scale_reconstruction_kwargs=scale_reconstruction_kwargs,
        )

        return fig, axes

    def plot_local_time_series(
        self,
        space_index,
        level,
        data=None,
        scale_reconstruction_kwargs=None,
    ):
        """Plots time series for an individual point.

        Includes the input data for decomposition, the low-frequency
        component for the next decomposition level, the residual of the high
        frequency component, and the reconstructions of the frequency bands
        for the point.

        :param space_index: Index of the point in space for the 1D snapshot.
        :type space_index: int
        :param level: Decomposition level for plotting
        :type level: int
        :param data: Original data, only necessary for level=0.
        :type data: numpy.ndarray
        :param scale_reconstruction_kwargs: Arguments for reconstructing the
            fit.
        :type scale_reconstruction_kwargs: dict
        :return:
        """

        x_iter = self._plot_helper_data_check(level, data=data)

        fig, axes = self.costs_array[level].plot_time_series(
            space_index,
            x_iter,
            scale_reconstruction_kwargs=scale_reconstruction_kwargs,
        )

        return fig, axes

    def _global_cluster(
        self,
        n_components=None,
        method=MiniBatchKMeans,
        transform_method=None,
        clustering_kwargs=None,
    ):
        """Helper function for clustering global frequency bands.

        :param method: Clustering method following the sklearn pattern (has
            `fit_predict` and `n_clusters` keywords). Default is
            MiniBatchKMeans.
        :type method: method
        :param n_components: The number of clusters to find.
        :type n_components: int
        :param transform_method: How to transform omega.
        :type transform_method: str or NoneType
        :param clustering_kwargs: Arguments for clustering method.
        :type clustering_kwargs: dict
        :return cluster_centroids: Centroid values for each cluster in
            transformed omega.
        :rtype cluster_centroids: numpy.ndarray
        :return omega_classes: Global frequency band cluster identifiers.
        :rtype omega_classes: numpy.ndarray
        :return omega_array: Transformed omega with NaNs removed
        :rtype omega_array: numpy.ndarray
        """

        if self._da_omega is None:
            self.multi_res_interp()

        if transform_method is None:
            transform_method = self._transform_method

        # This step flattens the array for clustering through the numpy
        # indexing to remove NaNs. NaNs, which must be exlcuded from
        # clustering, can appear as a result of the multi-resolution
        # interpolation.
        omega_array = self._da_omega.values
        omega_array = omega_array[~np.isnan(omega_array)]
        omega_array = self.transform_omega(
            omega_array, transform_method=transform_method
        )

        if clustering_kwargs is None:
            clustering_kwargs = {}
            random_state = 0
            clustering_kwargs["random_state"] = clustering_kwargs.get(
                "random_state", random_state
            )
        clustering = method(n_clusters=n_components, **clustering_kwargs)
        if not hasattr(clustering, "fit_predict") and callable(
            getattr(clustering, "fit_predict")
        ):
            raise ValueError(
                "Clustering method must have `fit_predict()` method."
            )

        omega_classes = clustering.fit_predict(np.atleast_2d(omega_array).T)
        cluster_centroids = clustering.cluster_centers_.flatten()

        # Sort the clusters by the centroid magnitude.
        idx = np.argsort(cluster_centroids)
        lut = np.zeros_like(idx)
        lut[idx] = np.arange(n_components)
        omega_classes = lut[omega_classes]
        cluster_centroids = cluster_centroids[idx]

        return omega_classes, cluster_centroids, omega_array

    def global_cluster_hyperparameter_sweep(
        self,
        n_components_range,
        transform_method=None,
        score_method=None,
        verbose=True,
        method=MiniBatchKMeans,
        clustering_kwargs=None,
    ):
        """
        Hyperparameter search for n_components for kmeans clustering.

        :param verbose: Flag for informing the user of parameter sweep progress.
        :type verbose: bool
        :param transform_method: How to transform omega for clustering. See
            `global_cluster_omega`
        :type transform_method: str
        :param n_components_range: Values of n_components for the
            hyperparameter sweep.
        :type n_components_range: numpy.ndarray of ints
        :param method: Clustering method following the sklearn pattern (has
            `fit_predict` and `n_clusters` keywords). Default is
            MiniBatchKMeans.
        :type method: method
        :param score_method: Valid scoring methods are 'silhouette' and
            'calinski-harabasz'. Default is the silhouette score, which can
            be slow for large numbers of samples.
        :type score_method: str or None
        :param clustering_kwargs: Keywords to pass to the clustering method.
        :return score: Scores for each n_components in n_components_range
        :rtype score: numpy.ndarray
        :return n_components: Optimal n_components for frequency band separation
        :rtype n_components: int
        """

        score = np.zeros_like(n_components_range, float)

        for nind, n in enumerate(n_components_range):
            if verbose:
                print(f"fitting n_components = {n:}")
            omega_classes, _, omega = self._global_cluster(
                n_components=n,
                transform_method=transform_method,
                method=method,
                clustering_kwargs=clustering_kwargs,
            )

            if score_method is None or score_method == "silhouette":
                score[nind] = silhouette_score(
                    omega.reshape(-1, 1),
                    omega_classes.reshape(-1, 1),
                )
            # Calinski-Harabasz is not a good counter to the silhouette score
            # since it just increases with increasing number of clusters. It is
            # only included as a reference and should be replaced if a serious
            # look is given to altering the clustering algorithm.
            elif score_method == "calinski-harabasz":
                score[nind] = calinski_harabasz_score(
                    omega.reshape(-1, 1),
                    omega_classes.reshape(-1, 1),
                )

        self._n_components_global = n_components_range[np.argmax(score)]

        return score, n_components_range[np.argmax(score)]

    def global_cluster_omega(
        self,
        n_components=None,
        transform_method=None,
        clustering_kwargs=None,
        method=MiniBatchKMeans,
    ):
        """Performs frequency band clustering on the global distribution of
        omega.

        Uses k-means clustering with the MiniBatchKMeans method. The
        hyperparameter for this unsupervised method is the number of
        clusters, given by `n_components`. Transforming omega may be
        necessary to get well-separated frequency bands. Options for
        transforming omega are:
            "period": :math:`\\frac{1}{\\omega}`
            "log10": :math:`log10(\\omega)`
            "square_frequencies": :math:`\\omega^2`
            "absolute": :math:`|\\omega|`
        Default value is "absolute". All transformations and clustering are
        performed on the imaginary portion of omega.

        :param method: Clustering method following the sklearn pattern (has
            `fit_predict` and `n_clusters` keywords). Default is
            MiniBatchKMeans.
        :type method: method
        :param n_components: The number of clusters to find.
        :type n_components: int
        :param transform_method: How to transform omega.
        :type transform_method: str or NoneType
        :param clustering_kwargs: Arguments for clustering method.
        :type clustering_kwargs: dict
        :return cluster_centroids: Centroid values for each cluster in
            transformed omega.
        :rtype cluster_centroids: numpy.ndarray
        :return omega_classes: Global frequency band cluster identifiers.
        :rtype omega_classes: numpy.ndarray
        :return omega_array: Transformed omega with NaNs removed
        :rtype omega_array: numpy.ndarray
        """

        if n_components is None and self._n_components_global is None:
            raise ValueError(
                (
                    "Either perform a cluster hyperparameter sweep or provide"
                    " `n_components`"
                )
            )
        if n_components is not None:
            self._n_components_global = n_components

        omega_classes, cluster_centroids, omega_array = self._global_cluster(
            n_components=self._n_components_global,
            method=method,
            transform_method=transform_method,
            clustering_kwargs=clustering_kwargs,
        )

        self._cluster_centroids = cluster_centroids
        self._omega_classes = omega_classes

        return cluster_centroids, omega_classes, omega_array

    @staticmethod
    def transform_omega(omega_array, transform_method=None):
        """Transform omega, primarily for clustering.
        Options for transforming omega are:
            "period": :math:`\\frac{1}{\\omega}`
            "log10": :math:`log10(\\omega)`
            "square_frequencies": :math:`\\omega^2`
            "absolute": :math:`|\\omega|`
        Default value is "absolute". All transformations and clustering are
        performed on the imaginary portion of omega.

        :param omega_array: Omega values
        :type omega_array: numpy.ndarray
        :param transform_method: How to transform the imaginary component of
            omega.
        :type transform_method: str
        :return: transformed omega array
        :rtype: numpy.ndarray
        """
        # @ToDo: Move to a set-based evaluation.
        if transform_method is None or transform_method == "absolute":
            omega_array = np.abs(omega_array.imag.astype("float"))
        elif transform_method == "square_frequencies":
            omega_array = (omega_array.imag**2).real.astype("float")
        elif transform_method == "period":
            omega_array = 1 / np.abs(omega_array.imag.astype("float"))
        elif transform_method == "log10":
            omega_array = np.log10(np.abs(omega_array.imag.astype("float")))
            # Impute log10(0) with the smallest non-zero values in log10(omega).
            zero_imputer = omega_array[np.isfinite(omega_array)].min()
            omega_array[~np.isfinite(omega_array)] = zero_imputer
        else:
            # @ToDo: Return accepted methods
            raise ValueError(
                f"Transform method {transform_method} not supported."
            )

        return omega_array

    def global_scale_reconstruction(
        self,
    ):
        """Reconstruct mrCOSTS into the constituent frequency bands.

        The reconstructed data are convolved with a guassian filter since
        points near the middle of the window are more reliable than points
        at the edge of the window. Note that this will leave the beginning
        and end of time series prone to larger errors. A best practice is
        to cut off `window_length` from each end before further analysis.

        :return: Reconstruction with dimensions of:
            n_decompositions x n_components x n_data_vars x n_time_steps
        :rtype: numpy.ndarray
        """

        # Each individual reconstructed window
        xr_sep = np.zeros(
            (
                self.n_decompositions,
                self._n_components_global,
                self._n_data_vars,
                self._n_time_steps,
            )
        )

        omega_classes_list = self.multi_res_deterp()

        for n_mrd, mrd in enumerate(self._costs_array):
            # Track the total contribution from all windows to each time step
            xn = np.zeros(self._n_time_steps)

            # Convolve each windowed reconstruction with a gaussian filter.
            # Std dev of gaussian filter
            recon_filter = mrd.build_kern(
                mrd.window_length, mrd._relative_filter_length
            )

            omega_classes = omega_classes_list[n_mrd]

            if mrd.svd_rank < np.max(self._svd_rank_array):
                truncate_slice = slice(None, mrd.svd_rank)
                omega_classes = omega_classes[:, truncate_slice]

            # Iterate over each window slide performed.
            for k in range(mrd.n_slides):
                w = mrd.modes_array[k]
                b = mrd.amplitudes_array[k]
                omega = np.atleast_2d(mrd.omega_array[k]).T
                classification = omega_classes[k]

                if not w.shape[1] == omega_classes.shape[1]:
                    print(w.shape)
                    print(omega_classes.shape)
                    print(n_mrd)
                    print(mrd.svd_rank)
                    print(truncate_slice)

                # Compute each segment of xr starting at "t = 0"
                t = mrd.time_array[k]
                t_start = mrd.time_array[k, 0]
                t = t - t_start

                # Reconstruct each frequency band separately.
                xr_sep_window = np.zeros(
                    (
                        self._n_components_global,
                        self._n_data_vars,
                        mrd.window_length,
                    )
                )

                # Get the indices for this window.
                if k == mrd.n_slides - 1 and mrd._non_integer_n_slide:
                    # Handle non-integer number of window slides by slightly
                    # shortening the last window's slide.
                    window_indices = slice(-mrd.window_length, None)
                else:
                    window_indices = slice(
                        k * mrd.step_size,
                        k * mrd.step_size + mrd.window_length,
                    )

                for j in np.arange(0, self._n_components_global):
                    class_ind = classification == j
                    xr_sep_window[j, :, :] = np.linalg.multi_dot(
                        [
                            w[:, class_ind],
                            np.diag(b[class_ind]),
                            np.exp(omega[class_ind] * t),
                        ]
                    )

                    # Multiply by the reconstruction filter which weights
                    # the reconstruction towards the middle of the window.
                    xr_sep_window[j, :, :] = (
                        xr_sep_window[j, :, :] * recon_filter
                    )

                    xr_sep[n_mrd, j, :, window_indices] = (
                        xr_sep[n_mrd, j, :, window_indices]
                        + xr_sep_window[j, :, :]
                    )

                # A normalization factor which weights the global reconstruction
                # by the number of window centers it contains. This accounts
                # for the convolution above.
                xn[window_indices] += recon_filter

            # Normalize by the reconstruction filter.
            xr_sep[n_mrd, :, :, :] = xr_sep[n_mrd, :, :, :] / xn

        return xr_sep

    def get_background(self):
        """Build the background values not included in the scale separation.

        :return: The low frequency components of the largest decomposition
            level which are not included in the scale separation.
        :rtype: numpy.ndarray
        """

        background = self.costs_array[-1].scale_reconstruction()[0, :, :]
        return background

    def global_reconstruction(self):
        """Reconstruction using all global frequency bands and decomposition
        levels.

        :return: Global reconstruction with background component.
        :rtype: numpy.ndarray
        """
        xr_sep = self.global_scale_reconstruction()
        xr_sep = xr_sep.sum(axis=(0, 1))
        xr_background = self.get_background()

        return xr_sep + xr_background
