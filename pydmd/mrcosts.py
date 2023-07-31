import numpy as np
from pydmd.bopdmd import BOPDMD
from .utils import compute_rank, compute_svd
import copy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import xarray as xr
from pydmd.costs import COSTS


class mrCOSTS:
    """Multi-resolution Coherent Spatio-Temporal Scale Separation (mrCOSTS) with DMD.

    :param window_length_array: Length of the analysis window in number of time steps.
    :type window_length_array: int
    :param step_size_array: Number of time steps to slide each CSM-DMD window.
    :type step_size_array: int
    :param n_components: Number of independent frequency bands for this window length.
    :type n_components: int
    :param svd_rank_array: The rank of the BOPDMD fit.
    :type svd_rank_array: int
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
        window_length_array=None,
        step_size_array=None,
        svd_rank_array=None,
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
        n_components_array=None,
    ):
        self._n_components_array = n_components_array
        self._step_size_array = step_size_array
        self._window_length_array = window_length_array
        self._svd_rank_array = svd_rank_array
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
        return self._svd_rank_array

    @property
    def window_length(self):
        """
        :return: the length of the windows used for this decomposition level.
        :rtype: int or float
        """
        return self._window_length

    @property
    def n_slides(self):
        """
        :return: number of window slides for this decomposition level.
        :rtype: int
        """
        return self._n_slides

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
    def n_components(self):
        if not hasattr(self, "_n_components"):
            raise ValueError("You need to call `cluster_omega()` first.")
        return self._n_components

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

    def fit(self, data):
        window_lengths = self._window_length_array
        step_sizes = self.step_size_array
        svd_ranks = self.svd_rank_array

        mrd_list = []
        suppress_growth = True
        transform_method = "squared"

        data_iter = np.zeros((num_decompositions, data.shape[0], data.shape[1]))
        data_iter[0, :, :] = data

        for n_decomp, (window, step, rank) in enumerate(
            zip(window_lengths, step_sizes, svd_ranks)
        ):
            print("Working on window length={}".format(window))

            x_iter = data_iter[n_decomp, :, :].squeeze()
            mrd = COSTS(
                svd_rank=rank,
                global_svd=True,
                pydmd_kwargs={"eig_constraints": {"conjugate_pairs", "stable"}},
            )

            print("Fitting")
            print("_________________________________________________")
            mrd.fit(x_iter, np.atleast_2d(time), window, step, verbose=True)
            print("_________________________________________________")

            # Cluster the frequency bands
            if self._cluster_sweep:
                n_components = mrd.cluster_hyperparameter_sweep(
                    transform_method=transform_method
                )
            else:
                n_components = self._n_components_array[n_decomp]

            _ = mrd.cluster_omega(
                n_components=n_components, transform_method=transform_method
            )

            # Global reconstruction
            global_reconstruction = mrd.global_reconstruction(
                {"suppress_growth": suppress_growth}
            )
            re = mrd.relative_error(global_reconstruction.real, x_iter)
            if verbose:
                print("Error in Global Reconstruction = {:.2}".format(re))

            # Scale separation
            xr_low_frequency, xr_high_frequency = mrd.scale_separation(
                scale_reconstruction_kwargs={"suppress_growth": suppress_growth}
            )

            # Pass the low frequency component to the next level of decomposition.
            if n_decomp < num_decompositions - 1:
                data_iter[n_decomp + 1, :, :] = xr_low_frequency

            # Save the object for later use.
            mrd_list.append(copy.copy(mrd))
