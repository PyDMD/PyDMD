"""
Base module for the DMD: `fit` method must be implemented in inherited classes
"""
from __future__ import division

import warnings
from builtins import object
from builtins import range
from os.path import splitext

import matplotlib as mpl
import numpy as np
from past.utils import old_div

mpl.rcParams["figure.max_open_warning"] = 0
import matplotlib.pyplot as plt

from .dmdoperator import DMDOperator

from functools import partial


class DMDBase(object):
    """
    Dynamic Mode Decomposition base class.

    :param svd_rank: the rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive interger, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation.
    :type svd_rank: int or float
    :param int tlsq_rank: rank truncation computing Total Least Square. Default
        is 0, that means no truncation.
    :param bool exact: flag to compute either exact DMD or projected DMD.
        Default is False.
    :param opt: If True, amplitudes are computed like in optimized DMD  (see
        :func:`~dmdbase.DMDBase._compute_amplitudes` for reference). If
        False, amplitudes are computed following the standard algorithm. If
        `opt` is an integer, it is used as the (temporal) index of the snapshot
        used to compute DMD modes amplitudes (following the standard algorithm).
        The reconstruction will generally be better in time instants near the
        chosen snapshot; however increasing `opt` may lead to wrong results when
        the system presents small eigenvalues. For this reason a manual
        selection of the number of eigenvalues considered for the analyisis may
        be needed (check `svd_rank`). Also setting `svd_rank` to a value between
        0 and 1 may give better results. Default is False.
    :type opt: bool or int
    :param rescale_mode: Scale Atilde as shown in
            10.1016/j.jneumeth.2015.10.010 (section 2.4) before computing its
            eigendecomposition. None means no rescaling, 'auto' means automatic
            rescaling using singular values, otherwise the scaling factors.
    :type rescale_mode: {'auto'} or None or numpy.ndarray
    :param bool forward_backward: If True, the low-rank operator is computed
        like in fbDMD (reference: https://arxiv.org/abs/1507.02264). Default is
        False.
    :param sorted_eigs: Sort eigenvalues (and modes/dynamics accordingly) by
        magnitude if `sorted_eigs='abs'`, by real part (and then by imaginary
        part to break ties) if `sorted_eigs='real'`. Default: False.
    :type sorted_eigs: {'real', 'abs'} or False

    :cvar dict original_time: dictionary that contains information about the
        time window where the system is sampled:

           - `t0` is the time of the first input snapshot;
           - `tend` is the time of the last input snapshot;
           - `dt` is the delta time between the snapshots.

    :cvar dict dmd_time: dictionary that contains information about the time
        window where the system is reconstructed:

            - `t0` is the time of the first approximated solution;
            - `tend` is the time of the last approximated solution;
            - `dt` is the delta time between the approximated solutions.

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
    ):
        self._Atilde = DMDOperator(
            svd_rank=svd_rank,
            exact=exact,
            rescale_mode=rescale_mode,
            forward_backward=forward_backward,
            sorted_eigs=sorted_eigs,
        )

        self._tlsq_rank = tlsq_rank
        self.original_time = None
        self.dmd_time = None
        self._opt = opt

        self._b = None  # amplitudes
        self._snapshots = None
        self._snapshots_shape = None

    @property
    def opt(self):
        return self._opt

    @property
    def tlsq_rank(self):
        return self._tlsq_rank

    @property
    def svd_rank(self):
        return self.operator._svd_rank

    @property
    def rescale_mode(self):
        return self.operator._rescale_mode

    @property
    def exact(self):
        return self.operator._exact

    @property
    def forward_backward(self):
        return self.operator._forward_backward

    @property
    def dmd_timesteps(self):
        """
        Get the timesteps of the reconstructed states.

        :return: the time intervals of the original snapshots.
        :rtype: numpy.ndarray
        """
        return np.arange(
            self.dmd_time["t0"],
            self.dmd_time["tend"] + self.dmd_time["dt"],
            self.dmd_time["dt"],
        )

    @property
    def original_timesteps(self):
        """
        Get the timesteps of the original snapshot.

        :return: the time intervals of the original snapshots.
        :rtype: numpy.ndarray
        """
        return np.arange(
            self.original_time["t0"],
            self.original_time["tend"] + self.original_time["dt"],
            self.original_time["dt"],
        )

    @property
    def modes(self):
        """
        Get the matrix containing the DMD modes, stored by column.

        :return: the matrix containing the DMD modes.
        :rtype: numpy.ndarray
        """
        return self.operator.modes

    @property
    def atilde(self):
        """
        Get the reduced Koopman operator A, called A tilde.

        :return: the reduced Koopman operator A.
        :rtype: numpy.ndarray
        """
        return self.operator.as_numpy_array

    @property
    def operator(self):
        """
        Get the instance of DMDOperator.

        :return: the instance of DMDOperator
        :rtype: DMDOperator
        """
        return self._Atilde

    @property
    def eigs(self):
        """
        Get the eigenvalues of A tilde.

        :return: the eigenvalues from the eigendecomposition of `atilde`.
        :rtype: numpy.ndarray
        """
        return self.operator.eigenvalues

    def _translate_eigs_exponent(self, tpow):
        """
        Transforms the exponent of the eigenvalues in the dynamics formula
        according to the selected value of `self.opt` (check the documentation
        for `opt` in :func:`__init__ <dmdbase.DMDBase.__init__>`).

        :param tpow: the exponent(s) of Sigma in the original DMD formula.
        :type tpow: int or np.ndarray
        :return: the exponent(s) adjusted according to `self.opt`
        :rtype: int or np.ndarray
        """

        if isinstance(self.opt, bool):
            amplitudes_snapshot_index = 0
        else:
            amplitudes_snapshot_index = self.opt

        if amplitudes_snapshot_index < 0:
            # we take care of negative indexes: -n becomes T - n
            return tpow - (self.snapshots.shape[1] + amplitudes_snapshot_index)
        else:
            return tpow - amplitudes_snapshot_index

    @property
    def dynamics(self):
        """
        Get the time evolution of each mode.

        .. math::

            \\mathbf{x}(t) \\approx
            \\sum_{k=1}^{r} \\boldsymbol{\\phi}_{k} \\exp \\left( \\omega_{k} t \\right) b_{k} =
            \\sum_{k=1}^{r} \\boldsymbol{\\phi}_{k} \\left( \\lambda_{k} \\right)^{\\left( t / \\Delta t \\right)} b_{k}

        :return: the matrix that contains all the time evolution, stored by
            row.
        :rtype: numpy.ndarray

        """
        temp = np.repeat(
            self.eigs[:, None], self.dmd_timesteps.shape[0], axis=1
        )
        tpow = old_div(
            self.dmd_timesteps - self.original_time["t0"],
            self.original_time["dt"],
        )

        # The new formula is x_(k+j) = \Phi \Lambda^k \Phi^(-1) x_j.
        # Since j is fixed, for a given snapshot "u" we have the following
        # formula:
        # x_u = \Phi \Lambda^{u-j} \Phi^(-1) x_j
        # Therefore tpow must be scaled appropriately.
        tpow = self._translate_eigs_exponent(tpow)

        return np.power(temp, tpow) * self.amplitudes[:, None]

    @property
    def reconstructed_data(self):
        """
        Get the reconstructed data.

        :return: the matrix that contains the reconstructed snapshots.
        :rtype: numpy.ndarray
        """
        return self.modes.dot(self.dynamics)

    @property
    def snapshots(self):
        """
        Get the original input data.

        :return: the matrix that contains the original snapshots.
        :rtype: numpy.ndarray
        """
        return self._snapshots

    @property
    def frequency(self):
        """
        Get the amplitude spectrum.

        :return: the array that contains the frequencies of the eigenvalues.
        :rtype: numpy.ndarray
        """
        return np.log(self.eigs).imag / (2 * np.pi * self.original_time["dt"])

    @property
    def growth_rate(self):  # To check
        """
        Get the growth rate values relative to the modes.

        :return: the Floquet values
        :rtype: numpy.ndarray
        """
        return self.eigs.real / self.original_time["dt"]

    @property
    def amplitudes(self):
        """
        Get the coefficients that minimize the error between the original
        system and the reconstructed one. For futher information, see
        `dmdbase._compute_amplitudes`.

        :return: the array that contains the amplitudes coefficient.
        :rtype: numpy.ndarray
        """
        return self._b

    def fit(self, X):
        """
        Abstract method to fit the snapshots matrices.

        Not implemented, it has to be implemented in subclasses.
        """
        raise NotImplementedError(
            "Subclass must implement abstract method {}.fit".format(
                self.__class__.__name__
            )
        )

    @staticmethod
    def _col_major_2darray(X):
        """
        Private method that takes as input the snapshots and stores them into a
        2D matrix, by column. If the input data is already formatted as 2D
        array, the method saves it, otherwise it also saves the original
        snapshots shape and reshapes the snapshots.

        :param X: the input snapshots.
        :type X: int or numpy.ndarray
        :return: the 2D matrix that contains the flatten snapshots, the shape
            of original snapshots.
        :rtype: numpy.ndarray, tuple
        """
        # If the data is already 2D ndarray
        if isinstance(X, np.ndarray) and X.ndim == 2:
            snapshots = X
            snapshots_shape = None
        else:
            input_shapes = [np.asarray(x).shape for x in X]

            if len(set(input_shapes)) != 1:
                raise ValueError("Snapshots have not the same dimension.")

            snapshots_shape = input_shapes[0]
            snapshots = np.transpose([np.asarray(x).flatten() for x in X])

        # check condition number of the data passed in
        cond_number = np.linalg.cond(snapshots)
        if cond_number > 10e4:
            warnings.warn(
                "Input data matrix X has condition number {}. "
                "Consider preprocessing data, passing in augmented data matrix, or regularization methods.".format(
                    cond_number
                )
            )

        return snapshots, snapshots_shape

    def _optimal_dmd_matrixes(self):
        # compute the vandermonde matrix
        vander = np.vander(self.eigs, len(self.dmd_timesteps), True)

        # perform svd on all the snapshots
        U, s, V = np.linalg.svd(self._snapshots, full_matrices=False)

        P = np.multiply(
            np.dot(self.modes.conj().T, self.modes),
            np.conj(np.dot(vander, vander.conj().T)),
        )

        tmp = np.linalg.multi_dot([U, np.diag(s), V]).conj().T
        q = np.conj(
            np.diag(np.linalg.multi_dot([vander, tmp, self.modes]))
        )

        return P,q


    def _compute_amplitudes(self):
        """
        Compute the amplitude coefficients. If `self.opt` is False the
        amplitudes are computed by minimizing the error between the modes and
        the first snapshot; if `self.opt` is True the amplitudes are computed by
        minimizing the error between the modes and all the snapshots, at the
        expense of bigger computational cost.

        This method uses the class variables self._snapshots (for the
        snapshots), self.modes and self.eigs.

        :return: the amplitudes array
        :rtype: numpy.ndarray

        References for optimal amplitudes:
        Jovanovic et al. 2014, Sparsity-promoting dynamic mode decomposition,
        https://hal-polytechnique.archives-ouvertes.fr/hal-00995141/document
        """
        if isinstance(self.opt, bool) and self.opt:
            # b optimal
            a = np.linalg.solve(*self._optimal_dmd_matrixes())
        else:
            if isinstance(self.opt, bool):
                amplitudes_snapshot_index = 0
            else:
                amplitudes_snapshot_index = self.opt

            a = np.linalg.lstsq(
                self.modes,
                self._snapshots.T[amplitudes_snapshot_index],
                rcond=None,
            )[0]

        return a

    def select_modes(self, func):
        """
        Select the DMD modes by using the given `func`.
        `func` has to be a callable function which takes as input the DMD
        object itself and return a numpy.ndarray of boolean where `False`
        indicates that the corresponding mode will be discarded.
        The class :class:`ModesSelectors` contains some pre-packed selector
        functions.

        :param callable func: the function to select the modes

        Example:

        >>> def stable_modes(dmd):
        >>>    toll = 1e-3
        >>>    return np.abs(np.abs(dmd.eigs) - 1) < toll
        >>> dmd = DMD(svd_rank=10)
        >>> dmd.fit(sample_data)
        >>> dmd.select_modes(stable_modes)
        """
        selected_indeces = func(self)

        self.operator._eigenvalues = self.operator._eigenvalues[
            selected_indeces
        ]
        self.operator._Lambda = self.operator._Lambda[selected_indeces]

        self.operator._eigenvectors = self.operator._eigenvectors[
            :, selected_indeces
        ]
        self.operator._modes = self.operator._modes[:, selected_indeces]

        self.operator._Atilde = np.linalg.multi_dot(
            [
                self.operator._eigenvectors,
                np.diag(self.operator._eigenvalues),
                np.linalg.pinv(self.operator._eigenvectors),
            ]
        )

        self._b = self._compute_amplitudes()

    class ModesSelectors:
        """
        A container class which defines some static methods for pre-packed
        modes selectors functions to be used in `select_modes`.

        `functools.partial` is used to provide both parametrization of the
        functions and immediate usability. For instance, to select the first
        x modes by integral contributions one would call:

        >>> from pydmd import DMDBase
        >>> dmd.select_modes(DMDBase.ModesSelectors.integral_contribution(x))
        """

        @staticmethod
        def _stable_modes(dmd, max_distance_from_unity):
            """
            Complete function of the modes selector `stable_modes`.

            :param float max_distance_from_unity: the maximum distance from the
                unit circle.
            :return np.ndarray: an array of bool, where each "True" index means
                that the corresponding DMD mode is selected.
            """
            return np.abs(np.abs(dmd.eigs) - 1) < max_distance_from_unity

        @staticmethod
        def stable_modes(max_distance_from_unity):
            """
            Select all the modes such that the magnitude of the corresponding
            eigenvalue is in `(1-max_distance_from_unity,1+max_distance_from_unity)`,
            non inclusive.

            :param float max_distance_from_unity: the maximum distance from the
                unit circle.
            :return callable: function which can be used as the parameter
                of `DMDBase.select_modes` to select DMD modes according to
                the criteria of stability.
            """
            return partial(
                DMDBase.ModesSelectors._stable_modes,
                max_distance_from_unity=max_distance_from_unity,
            )

        @staticmethod
        def _compute_integral_contribution(mode, dynamic):
            """
            Compute the integral contribution across time of the given DMD mode,
            given the mode and its dynamic, as shown in
            http://dx.doi.org/10.1016/j.euromechflu.2016.11.015

            :param numpy.ndarray mode: the DMD mode.
            :param numpy.ndarray dynamic: the dynamic of the given DMD mode, as
                returned by `dmd.dynamics[mode_index]`.
            :return float: the integral contribution of the given DMD mode.
            """
            return pow(np.linalg.norm(mode), 2) * sum(np.abs(dynamic))

        @staticmethod
        def _integral_contribution(dmd, n):
            """
            Complete function of the modes selector `integral_contribution`.

            :param int n: the number of DMD modes to be selected.
            :return np.ndarray: an array of bool, where each "True" index means
                that the corresponding DMD mode is selected.
            """

            # temporary reset dmd_time to original_time
            temp = dmd.dmd_time
            dmd.dmd_time = dmd.original_time

            dynamics = dmd.dynamics
            modes = dmd.modes

            # reset dmd_time
            dmd.dmd_time = temp

            n_of_modes = modes.shape[1]
            integral_contributions = [
                DMDBase.ModesSelectors._compute_integral_contribution(*tp)
                for tp in zip(modes.T, dynamics)
            ]

            indexes_first_n = np.array(integral_contributions).argsort()[-n:]

            truefalse_array = np.array([False for _ in range(n_of_modes)])
            truefalse_array[indexes_first_n] = True
            return truefalse_array

        @staticmethod
        def integral_contribution(n):
            """
            Select the
            Reference: http://dx.doi.org/10.1016/j.euromechflu.2016.11.015

            :param int n: the number of DMD modes to be selected.
            :return callable: function which can be used as the parameter
                of `DMDBase.select_modes` to select DMD modes according to
                the criteria of integral contribution.
            """
            return partial(DMDBase.ModesSelectors._integral_contribution, n=n)

    def _enforce_ratio(self, goal_ratio, supx, infx, supy, infy):
        """
        Computes the right value of `supx,infx,supy,infy` to obtain the desired
        ratio in :func:`plot_eigs`. Ratio is defined as
        ::
            dx = supx - infx
            dy = supy - infy
            max(dx,dy) / min(dx,dy)

        :param float goal_ratio: the desired ratio.
        :param float supx: the old value of `supx`, to be adjusted.
        :param float infx: the old value of `infx`, to be adjusted.
        :param float supy: the old value of `supy`, to be adjusted.
        :param float infy: the old value of `infy`, to be adjusted.
        :return tuple: a tuple which contains the updated values of
            `supx,infx,supy,infy` in this order.
        """

        dx = supx - infx
        if dx == 0:
            dx = 1.0e-16
        dy = supy - infy
        if dy == 0:
            dy = 1.0e-16
        ratio = max(dx, dy) / min(dx, dy)

        if ratio >= goal_ratio:
            if dx < dy:
                goal_size = dy / goal_ratio

                supx += (goal_size - dx) / 2
                infx -= (goal_size - dx) / 2
            elif dy < dx:
                goal_size = dx / goal_ratio

                supy += (goal_size - dy) / 2
                infy -= (goal_size - dy) / 2

        return (supx, infx, supy, infy)

    def _plot_limits(self, narrow_view):
        if narrow_view:
            supx = max(self.eigs.real) + 0.05
            infx = min(self.eigs.real) - 0.05

            supy = max(self.eigs.imag) + 0.05
            infy = min(self.eigs.imag) - 0.05

            return self._enforce_ratio(8, supx, infx, supy, infy)
        else:
            return np.max(np.ceil(np.absolute(self.eigs)))

    def plot_eigs(
        self,
        show_axes=True,
        show_unit_circle=True,
        figsize=(8, 8),
        title="",
        narrow_view=False,
        dpi=None,
        filename=None,
    ):
        """
        Plot the eigenvalues.
        :param bool show_axes: if True, the axes will be showed in the plot.
            Default is True.
        :param bool show_unit_circle: if True, the circle with unitary radius
            and center in the origin will be showed. Default is True.
        :param tuple(int,int) figsize: tuple in inches defining the figure
            size. Default is (8, 8).
        :param str title: title of the plot.
        :param narrow_view bool: if True, the plot will show only the smallest
            rectangular area which contains all the eigenvalues, with a padding
            of 0.05. Not compatible with `show_axes=True`. Default is False.
        :param dpi int: If not None, the given value is passed to ``plt.figure``.
        :param str filename: if specified, the plot is saved at `filename`.
        """
        if self.eigs is None:
            raise ValueError(
                "The eigenvalues have not been computed."
                "You have to perform the fit method."
            )

        if dpi is not None:
            plt.figure(figsize=figsize, dpi=dpi)
        else:
            plt.figure(figsize=figsize)

        plt.title(title)
        plt.gcf()
        ax = plt.gca()

        (points,) = ax.plot(
            self.eigs.real, self.eigs.imag, "bo", label="Eigenvalues"
        )

        if narrow_view:
            supx, infx, supy, infy = self._plot_limits(narrow_view)

            # set limits for axis
            ax.set_xlim((infx, supx))
            ax.set_ylim((infy, supy))

            # x and y axes
            if show_axes:
                endx = np.min([supx, 1.0])
                ax.annotate(
                    "",
                    xy=(endx, 0.0),
                    xytext=(np.max([infx, -1.0]), 0.0),
                    arrowprops=dict(arrowstyle=("->" if endx == 1.0 else "-")),
                )

                endy = np.min([supy, 1.0])
                ax.annotate(
                    "",
                    xy=(0.0, endy),
                    xytext=(0.0, np.max([infy, -1.0])),
                    arrowprops=dict(arrowstyle=("->" if endy == 1.0 else "-")),
                )
        else:
            # set limits for axis
            limit = self._plot_limits(narrow_view)

            ax.set_xlim((-limit, limit))
            ax.set_ylim((-limit, limit))

            # x and y axes
            if show_axes:
                ax.annotate(
                    "",
                    xy=(np.max([limit * 0.8, 1.0]), 0.0),
                    xytext=(np.min([-limit * 0.8, -1.0]), 0.0),
                    arrowprops=dict(arrowstyle="->"),
                )
                ax.annotate(
                    "",
                    xy=(0.0, np.max([limit * 0.8, 1.0])),
                    xytext=(0.0, np.min([-limit * 0.8, -1.0])),
                    arrowprops=dict(arrowstyle="->"),
                )

        plt.ylabel("Imaginary part")
        plt.xlabel("Real part")

        if show_unit_circle:
            unit_circle = plt.Circle(
                (0.0, 0.0),
                1.0,
                color="green",
                fill=False,
                label="Unit circle",
                linestyle="--",
            )
            ax.add_artist(unit_circle)

        # Dashed grid
        gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        for line in gridlines:
            line.set_linestyle("-.")
        ax.grid(True)

        # legend
        if show_unit_circle:
            ax.add_artist(
                plt.legend(
                    [points, unit_circle],
                    ["Eigenvalues", "Unit circle"],
                    loc="best",
                )
            )
        else:
            ax.add_artist(plt.legend([points], ["Eigenvalues"], loc="best"))

        ax.set_aspect("equal")

        if filename:
            plt.savefig(filename)
        else:
            plt.show()

    def plot_modes_2D(
        self,
        index_mode=None,
        filename=None,
        x=None,
        y=None,
        order="C",
        figsize=(8, 8),
    ):
        """
        Plot the DMD Modes.

        :param index_mode: the index of the modes to plot. By default, all
            the modes are plotted.
        :type index_mode: int or sequence(int)
        :param str filename: if specified, the plot is saved at `filename`.
        :param numpy.ndarray x: domain abscissa.
        :param numpy.ndarray y: domain ordinate
        :param order: read the elements of snapshots using this index order,
            and place the elements into the reshaped array using this index
            order.  It has to be the same used to store the snapshot. 'C' means
            to read/ write the elements using C-like index order, with the last
            axis index changing fastest, back to the first axis index changing
            slowest.  'F' means to read / write the elements using Fortran-like
            index order, with the first index changing fastest, and the last
            index changing slowest.  Note that the 'C' and 'F' options take no
            account of the memory layout of the underlying array, and only
            refer to the order of indexing.  'A' means to read / write the
            elements in Fortran-like index order if a is Fortran contiguous in
            memory, C-like order otherwise.
        :type order: {'C', 'F', 'A'}, default 'C'.
        :param tuple(int,int) figsize: tuple in inches defining the figure
            size. Default is (8, 8).
        """
        if self.modes is None:
            raise ValueError(
                "The modes have not been computed."
                "You have to perform the fit method."
            )

        if x is None and y is None:
            if self._snapshots_shape is None:
                raise ValueError(
                    "No information about the original shape of the snapshots."
                )

            if len(self._snapshots_shape) != 2:
                raise ValueError(
                    "The dimension of the input snapshots is not 2D."
                )

        # If domain dimensions have not been passed as argument,
        # use the snapshots dimensions
        if x is None and y is None:
            x = np.arange(self._snapshots_shape[0])
            y = np.arange(self._snapshots_shape[1])

        xgrid, ygrid = np.meshgrid(x, y)

        if index_mode is None:
            index_mode = list(range(self.modes.shape[1]))
        elif isinstance(index_mode, int):
            index_mode = [index_mode]

        if filename:
            basename, ext = splitext(filename)

        for idx in index_mode:
            fig = plt.figure(figsize=figsize)
            fig.suptitle("DMD Mode {}".format(idx))

            real_ax = fig.add_subplot(1, 2, 1)
            imag_ax = fig.add_subplot(1, 2, 2)

            mode = self.modes.T[idx].reshape(xgrid.shape, order=order)

            real = real_ax.pcolor(
                xgrid,
                ygrid,
                mode.real,
                cmap="jet",
                vmin=mode.real.min(),
                vmax=mode.real.max(),
            )
            imag = imag_ax.pcolor(
                xgrid,
                ygrid,
                mode.imag,
                vmin=mode.imag.min(),
                vmax=mode.imag.max(),
            )

            fig.colorbar(real, ax=real_ax)
            fig.colorbar(imag, ax=imag_ax)

            real_ax.set_aspect("auto")
            imag_ax.set_aspect("auto")

            real_ax.set_title("Real")
            imag_ax.set_title("Imag")

            # padding between elements
            plt.tight_layout(pad=2.0)

            if filename:
                plt.savefig("{0}.{1}{2}".format(basename, idx, ext))
                plt.close(fig)

        if not filename:
            plt.show()

    def plot_snapshots_2D(
        self,
        index_snap=None,
        filename=None,
        x=None,
        y=None,
        order="C",
        figsize=(8, 8),
    ):
        """
        Plot the snapshots.

        :param index_snap: the index of the snapshots to plot. By default, all
            the snapshots are plotted.
        :type index_snap: int or sequence(int)
        :param str filename: if specified, the plot is saved at `filename`.
        :param numpy.ndarray x: domain abscissa.
        :param numpy.ndarray y: domain ordinate
        :param order: read the elements of snapshots using this index order,
            and place the elements into the reshaped array using this index
            order.  It has to be the same used to store the snapshot. 'C' means
            to read/ write the elements using C-like index order, with the last
            axis index changing fastest, back to the first axis index changing
            slowest.  'F' means to read / write the elements using Fortran-like
            index order, with the first index changing fastest, and the last
            index changing slowest.  Note that the 'C' and 'F' options take no
            account of the memory layout of the underlying array, and only
            refer to the order of indexing.  'A' means to read / write the
            elements in Fortran-like index order if a is Fortran contiguous in
            memory, C-like order otherwise.
        :type order: {'C', 'F', 'A'}, default 'C'.
        :param tuple(int,int) figsize: tuple in inches defining the figure
            size. Default is (8, 8).
        """
        if self._snapshots is None:
            raise ValueError("Input snapshots not found.")

        if x is None and y is None:
            if self._snapshots_shape is None:
                raise ValueError(
                    "No information about the original shape of the snapshots."
                )

            if len(self._snapshots_shape) != 2:
                raise ValueError(
                    "The dimension of the input snapshots is not 2D."
                )

        # If domain dimensions have not been passed as argument,
        # use the snapshots dimensions
        if x is None and y is None:
            x = np.arange(self._snapshots_shape[0])
            y = np.arange(self._snapshots_shape[1])

        xgrid, ygrid = np.meshgrid(x, y)

        if index_snap is None:
            index_snap = list(range(self._snapshots.shape[1]))
        elif isinstance(index_snap, int):
            index_snap = [index_snap]

        if filename:
            basename, ext = splitext(filename)

        for idx in index_snap:
            fig = plt.figure(figsize=figsize)
            fig.suptitle("Snapshot {}".format(idx))

            snapshot = self._snapshots.T[idx].real.reshape(
                xgrid.shape, order=order
            )

            contour = plt.pcolor(
                xgrid,
                ygrid,
                snapshot,
                vmin=snapshot.min(),
                vmax=snapshot.max(),
            )

            fig.colorbar(contour)

            if filename:
                plt.savefig("{0}.{1}{2}".format(basename, idx, ext))
                plt.close(fig)

        if not filename:
            plt.show()


class DMDTimeDict(dict):
    def __setitem__(self, key, value):
        if key in ["t0", "tend", "dt"]:
            dict.__setitem__(self, key, value)
        else:
            raise KeyError(
                'DMDBase.dmd_time accepts only the following keys: "t0", "tend", "dt", {} is not allowed.'.format(
                    key
                )
            )
