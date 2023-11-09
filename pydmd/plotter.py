"""
Module for DMD plotting.
"""
import warnings
from os.path import splitext

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pydmd import MrDMD

from .bopdmd import BOPDMD
from .hankeldmd import HankelDMD
from .havok import HAVOK
from .preprocessing import PrePostProcessingDMD

mpl.rcParams["figure.max_open_warning"] = 0


def _enforce_ratio(goal_ratio, supx, infx, supy, infy):
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


def _plot_limits(dmd, narrow_view):
    if narrow_view:
        supx = max(dmd.eigs.real) + 0.05
        infx = min(dmd.eigs.real) - 0.05

        supy = max(dmd.eigs.imag) + 0.05
        infy = min(dmd.eigs.imag) - 0.05

        return _enforce_ratio(8, supx, infx, supy, infy)
    return np.max(np.ceil(np.absolute(dmd.eigs)))


def plot_eigs(
    dmd,
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

    :param dmd: DMD instance.
    :type dmd: pydmd.DMDBase
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
    :param dpi int: If not None, the given value is passed to
        ``plt.figure``.
    :param str filename: if specified, the plot is saved at `filename`.
    """
    if isinstance(dmd, MrDMD):
        raise ValueError("You should use plot_eigs_mrdmd instead")

    if dmd.eigs is None:
        raise ValueError(
            "The eigenvalues have not been computed."
            "You have to call the fit() method."
        )

    if dpi is not None:
        plt.figure(figsize=figsize, dpi=dpi)
    else:
        plt.figure(figsize=figsize)

    plt.title(title)
    plt.gcf()
    ax = plt.gca()

    points = ax.plot(dmd.eigs.real, dmd.eigs.imag, "bo", label="Eigenvalues")

    if narrow_view:
        supx, infx, supy, infy = _plot_limits(dmd, narrow_view)

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
        limit = _plot_limits(dmd, narrow_view)

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


def plot_eigs_mrdmd(
    dmd,
    show_axes=True,
    show_unit_circle=True,
    figsize=(8, 8),
    title="",
    level=None,
    node=None,
    filename=None,
):
    """
    Plot the eigenvalues.

    :param bool show_axes: if True, the axes will be showed in the plot.
            Default is True.
    :param bool show_unit_circle: if True, the circle with unitary radius
            and center in the origin will be showed. Default is True.
    :param tuple(int,int) figsize: tuple in inches of the figure.
    :param str title: title of the plot.
    :param int level: plot only the eigenvalues of specific level.
    :param int node: plot only the eigenvalues of specific node.
    :param str filename: if specified, the plot is saved at `filename`.
    """
    if not isinstance(dmd, MrDMD):
        raise ValueError(f"Expected MrDMD, found {type(dmd)}")

    if dmd.eigs is None:
        raise ValueError(
            "The eigenvalues have not been computed."
            "You have to perform the fit method."
        )

    if level:
        peigs = dmd.partial_eigs(level=level, node=node)
    else:
        peigs = dmd.eigs

    plt.figure(figsize=figsize)
    plt.title(title)
    plt.gcf()
    ax = plt.gca()

    if not level:
        cmap = plt.get_cmap("viridis")
        colors = [cmap(i) for i in np.linspace(0, 1, len(dmd.dmd_tree.levels))]

        points = []
        for l in dmd.dmd_tree.levels:
            eigs = dmd.partial_eigs(l)

            points.append(
                ax.plot(eigs.real, eigs.imag, ".", color=colors[l])[0]
            )
    else:
        points = []
        points.append(
            ax.plot(peigs.real, peigs.imag, "bo", label="Eigenvalues")[0]
        )

    # set limits for axis
    limit = np.max(np.ceil(np.absolute(peigs)))
    ax.set_xlim((-limit, limit))
    ax.set_ylim((-limit, limit))

    plt.ylabel("Imaginary part")
    plt.xlabel("Real part")

    if show_unit_circle:
        unit_circle = plt.Circle(
            (0.0, 0.0), 1.0, color="green", fill=False, linestyle="--"
        )
        ax.add_artist(unit_circle)

    # Dashed grid
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle("-.")
    ax.grid(True)

    ax.set_aspect("equal")

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

    # legend
    if level:
        labels = [f"Eigenvalues - level {level}"]
    else:
        labels = [f"Eigenvalues - level {i}" for i in range(len(points))]

    if show_unit_circle:
        points += [unit_circle]
        labels += ["Unit circle"]

    ax.add_artist(plt.legend(points, labels, loc="best"))
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def plot_modes_2D(
    dmd,
    snapshots_shape=None,
    index_mode=None,
    filename=None,
    x=None,
    y=None,
    order="C",
    figsize=(8, 8),
):
    """
    Plot the DMD Modes.

    :param dmd: DMD instance.
    :type dmd: pydmd.DMDBase
    :param snapshots_shape: Shape of the snapshots.
    :type tuple: A tuple of ints containing the shape of a single snapshot.
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
    if dmd.modes is None:
        raise ValueError(
            "The modes have not been computed."
            "You have to perform the fit method."
        )

    if snapshots_shape is None:
        snapshots_shape = dmd.snapshots_shape

    if x is None and y is None:
        if snapshots_shape is None:
            raise ValueError(
                "No information about the original shape of the snapshots."
            )

        if len(snapshots_shape) != 2:
            raise ValueError("The dimension of the input snapshots is not 2D.")

    # If domain dimensions have not been passed as argument,
    # use the snapshots dimensions
    if x is None and y is None:
        x = np.arange(snapshots_shape[0])
        y = np.arange(snapshots_shape[1])

    xgrid, ygrid = np.meshgrid(x, y)

    if index_mode is None:
        index_mode = list(range(dmd.modes.shape[1]))
    elif isinstance(index_mode, int):
        index_mode = [index_mode]

    if filename:
        basename, ext = splitext(filename)

    for idx in index_mode:
        fig = plt.figure(figsize=figsize)
        fig.suptitle(f"DMD Mode {idx}")

        real_ax = fig.add_subplot(1, 2, 1)
        imag_ax = fig.add_subplot(1, 2, 2)

        mode = dmd.modes.T[idx].reshape(xgrid.shape, order=order)

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
            plt.savefig(f"{basename}.{idx}{ext}")
            plt.close(fig)

    if not filename:
        plt.show()


def plot_snapshots_2D(
    dmd,
    snapshots_shape=None,
    index_snap=None,
    filename=None,
    x=None,
    y=None,
    order="C",
    figsize=(8, 8),
):
    """
    Plot the snapshots.

    :param dmd: DMD instance.
    :type dmd: pydmd.DMDBase
    :param snapshots_shape: Shape of the snapshots.
    :type tuple: A tuple of ints containing the shape of a single snapshot.
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
    if dmd.snapshots is None:
        raise ValueError("Input snapshots not found.")

    if snapshots_shape is None:
        snapshots_shape = dmd.snapshots_shape

    if x is None and y is None:
        if snapshots_shape is None:
            raise ValueError(
                "No information about the original shape of the snapshots."
            )

        if len(snapshots_shape) != 2:
            raise ValueError("The dimension of the snapshots is not 2D.")

    # If domain dimensions have not been passed as argument,
    # use the snapshots dimensions
    if x is None and y is None:
        x = np.arange(snapshots_shape[0])
        y = np.arange(snapshots_shape[1])

    xgrid, ygrid = np.meshgrid(x, y)

    if index_snap is None:
        index_snap = list(range(dmd.snapshots.shape[1]))
    elif isinstance(index_snap, int):
        index_snap = [index_snap]

    if filename:
        basename, ext = splitext(filename)

    for idx in index_snap:
        fig = plt.figure(figsize=figsize)
        fig.suptitle(f"Snapshot {idx}")

        snapshot = dmd.snapshots.T[idx].real.reshape(xgrid.shape, order=order)

        contour = plt.pcolormesh(
            xgrid,
            ygrid,
            snapshot,
            vmin=snapshot.min(),
            vmax=snapshot.max(),
        )

        fig.colorbar(contour)

        if filename:
            plt.savefig(f"{basename}.{idx}{ext}")
            plt.close(fig)

    if not filename:
        plt.show()


def plot_summary(
    dmd,
    *,
    continuous=False,
    snapshots_shape=None,
    index_modes=None,
    filename=None,
    order="C",
    figsize=(12, 8),
    dpi=200,
    tight_layout_kwargs=None,
    main_colors=("r", "b", "g", "gray"),
    mode_color="k",
    mode_cmap="bwr",
    dynamics_color="tab:blue",
    sval_ms=8,
    max_eig_ms=10,
    max_sval_plot=50,
    title_fontsize=14,
    label_fontsize=12,
    plot_semilogy=False,
    remove_cmap_ticks=False,
):
    """
    Generate a 3 x 3 summarizing plot that contains the following components:
    - the singular value spectrum of the data
    - the discrete-time and continuous-time DMD eigenvalues
    - the three DMD modes specified by the `index_modes` parameter
    - the dynamics corresponding with each plotted mode
    Eigenvalues, modes, and dynamics are ordered according to the magnitude of
    their corresponding amplitude value. Singular values and eigenvalues that
    are associated with plotted modes and dynamics are also highlighted.

    :param dmd: DMD instance.
    :type dmd: pydmd.DMDBase
    :param continuous: Whether or not the eigenvalues of the given DMD instance
        are continuous-time. If `False`, the eigenvalues are assumed to be the
        discrete-time eigenvalues. If `True`, the eigenvalues are taken to be
        the continuous-time eigenvalues. Note that `continuous` is
        automatically assumed to be `True` if a `BOPDMD` model is given.
    :type continuous: bool
    :param snapshots_shape: Shape of the snapshots. If not provided, the shape
        of the snapshots and modes is assumed to be the flattened space dim of
        the snapshot data.
    :type snapshots_shape: tuple(int, int)
    :param index_modes: A list of the indices of the modes to plot. By default,
        the first three leading modes are plotted.
    :type index_modes: list
    :param filename: If specified, the plot is saved at `filename`.
    :type filename: str
    :param order: Read the elements of snapshots using this index order,
        and place the elements into the reshaped array using this index order.
        It has to be the same used to store the snapshot. "C" means to
        read/write the elements using C-like index order, with the last axis
        index changing fastest, back to the first axis index changing slowest.
        "F" means to read/write the elements using Fortran-like index order,
        with the first index changing fastest, and the last index changing
        slowest. Note that the "C" and "F" options take no account of the
        memory layout of the underlying array, and only refer to the order of
        indexing. "A" means to read/write the elements in Fortran-like index
        order if a is Fortran contiguous in memory, C-like order otherwise.
        "C" is used by default.
    :type order: {"C", "F", "A"}
    :param figsize: Tuple in inches defining the figure size.
    :type figsize: tuple(int, int)
    :param dpi: Figure resolution.
    :type dpi: int
    :param tight_layout_kwargs: Optional dictionary of
        `matplotlib.pyplot.tight_layout()` parameters.
    :type tight_layout_kwargs: dict
    :param main_colors: Tuple of strings defining the colors used to denote
        eigenvalue, mode, dynamics associations. The first three colors are
        used to highlight the singular values and eigenvalues associated with
        the plotted modes and dynamics, while the fourth color is used to
        denote all other values.
    :type main_colors: tuple(str, str, str, str)
    :param mode_color: Color used to plot the modes, if modes are 1D.
    :type mode_color: str
    :param mode_cmap: Colormap used to plot the modes, if modes are 2D.
    :type mode_cmap: str
    :param dynamics_color: Color used to plot the dynamics.
    :type dynamics_color: str
    :param sval_ms: Marker size of all singular values.
    :type sval_ms: int
    :param max_eig_ms: Marker size of the most prominent eigenvalue.
    :type max_eig_ms: int
    :param max_sval_plot: Maximum number of singular values to plot.
    :type max_sval_plot: int
    :param title_fontsize: Fontsize used for subplot titles.
    :type title_fontsize: int
    :param label_fontsize: Fontsize used for axis labels.
    :type label_fontsize: int
    :param plot_semilogy: Whether or not to plot the singular values on a
        semilogy plot. If `True`, a semilogy plot is used.
    :type plot_semilogy: bool
    :param remove_cmap_ticks: Whether or not to include the ticks on 2D mode
        plots. If `True`, ticks are removed from all 2D mode plots.
    :type remove_cmap_ticks: bool
    """

    # This plotting method is inappropriate for plotting HAVOK results.
    if isinstance(dmd, HAVOK):
        raise ValueError("You should use HAVOK.plot_summary() instead.")

    # Check that the DMD instance has been fitted.
    if dmd.modes is None:
        raise ValueError(
            "The modes have not been computed."
            "You need to perform fit() first."
        )

    # By default, snapshots_shape is the flattened space dimension.
    if snapshots_shape is None:
        snapshots_shape = (len(dmd.snapshots),)
    # Only 2D tuples are admissible for snapshots_shape.
    elif not isinstance(snapshots_shape, tuple) or len(snapshots_shape) != 2:
        raise ValueError("snapshots_shape must be None or a 2D tuple.")

    # Override index_modes if there are less than 3 modes available.
    if len(dmd.eigs) < 3:
        warnings.warn(
            "Provided dmd model has less than 3 modes."
            "Plotting all available modes."
        )
        index_modes = list(range(len(dmd.eigs)))
    # By default, we plot the 3 leading modes and their dynamics.
    elif index_modes is None:
        index_modes = list(range(3))
    # index_modes was provided - check its type and its length.
    elif not isinstance(index_modes, list) or len(index_modes) > 3:
        raise ValueError("index_modes must be a list of length at most 3.")
    # Indices cannot go past the total number of available or plottable modes.
    elif np.any(np.array(index_modes) >= min(len(dmd.eigs), max_sval_plot)):
        raise ValueError(
            f"Cannot view past mode {min(len(dmd.eigs), max_sval_plot)}."
        )

    # Sort eigenvalues, modes, and dynamics according to amplitude magnitude.
    mode_order = np.argsort(-np.abs(dmd.amplitudes))
    lead_eigs = dmd.eigs[mode_order]
    lead_modes = dmd.modes[:, mode_order]
    lead_dynamics = dmd.dynamics[mode_order]

    # Get time information for eigenvalue conversions.
    if isinstance(dmd, BOPDMD) or (
        isinstance(dmd, PrePostProcessingDMD)
        and isinstance(dmd.pre_post_processed_dmd, BOPDMD)
    ):
        # BOPDMD models store time in the time attribute.
        # BOPDMD models also always compute continuous-time eigenvalues.
        cont_eigs = lead_eigs
        time = dmd.time
        dt = dmd.time[1] - dmd.time[0]
        if not np.allclose(dmd.time[1:] - dmd.time[:-1], dt):
            warnings.warn(
                "Time step is not uniform. "
                "No discrete-time eigenvalues to plot..."
            )
            disc_eigs = None
        else:
            disc_eigs = np.exp(lead_eigs * dt)
    else:
        # For all other dmd models, go to the TimeDict for time information.
        try:
            time = dmd.dmd_timesteps
            dt = dmd.original_time["dt"]
        except AttributeError:
            warnings.warn(
                "No time step information available. "
                "Using dt = 1 and t0 = 0."
            )
            time = np.arange(dmd.snapshots.shape[-1])
            dt = 1.0

        if continuous:
            cont_eigs = lead_eigs
            disc_eigs = np.exp(lead_eigs * dt)
        else:
            disc_eigs = lead_eigs
            cont_eigs = np.log(lead_eigs) / dt

    # Compute the singular values of the data matrix.
    if isinstance(dmd, HankelDMD):
        # Use time-delay data matrix to compute singular values.
        snp = dmd.ho_snapshots
    else:
        # Use input data matrix to compute singular values.
        snp = dmd.snapshots
    s = np.linalg.svd(snp, full_matrices=False, compute_uv=False)
    # Compute the percent of data variance captured by each singular value.
    s_var = s * (100 / np.sum(s))

    # Generate the summarizing plot.
    fig, (eig_axes, mode_axes, dynamics_axes) = plt.subplots(
        3, 3, figsize=figsize, dpi=dpi
    )

    # PLOT 1: Plot the singular value spectrum.
    s_var_plot = s_var[:max_sval_plot]
    eig_axes[0].set_title("Singular Values", fontsize=title_fontsize)
    eig_axes[0].set_ylabel("% variance", fontsize=label_fontsize)
    s_t = np.arange(len(s_var_plot)) + 1
    eig_axes[0].plot(
        s_t, s_var_plot, "o", c=main_colors[-1], ms=sval_ms, mec="k"
    )
    for i, idx in enumerate(index_modes):
        eig_axes[0].plot(
            s_t[idx],
            s_var_plot[idx],
            "o",
            c=main_colors[i],
            ms=sval_ms,
            mec="k",
        )
    if plot_semilogy:
        eig_axes[0].semilogy()

    # PLOTS 2-3: Plot the eigenvalues (discrete-time and continuous-time).
    # Scale marker sizes to reflect the amount of variance captured.
    ms_vals = max_eig_ms * np.sqrt(s_var / s_var[0])
    for i, (ax, eigs) in enumerate(zip(eig_axes[1:], [disc_eigs, cont_eigs])):
        # Plot the complex plane axes.
        ax.axvline(x=0, c="k", lw=1)
        ax.axhline(y=0, c="k", lw=1)
        ax.axis("equal")
        # PLOT 2: Plot the discrete-time eigenvalues on the unit circle.
        if i == 0:
            ax.set_title("Discrete-time Eigenvalues", fontsize=title_fontsize)
            t = np.linspace(0, 2 * np.pi, 100)
            ax.plot(np.cos(t), np.sin(t), c="tab:blue", ls="--")
            ax.set_xlabel(r"$Re(\lambda)$", fontsize=label_fontsize)
            ax.set_ylabel(r"$Im(\lambda)$", fontsize=label_fontsize)
        # PLOT 3: Plot the continuous-time eigenvalues.
        else:
            ax.set_title("Continuous-time Eigenvalues", fontsize=title_fontsize)
            ax.set_xlabel(r"$Im(\omega)$", fontsize=label_fontsize)
            ax.set_ylabel(r"$Re(\omega)$", fontsize=label_fontsize)
        # Plot the eigenvalues (discrete or continuous).
        if eigs is not None:
            for idx, eig in enumerate(eigs):
                if idx in index_modes:
                    color = main_colors[index_modes.index(idx)]
                else:
                    color = main_colors[-1]
                if i == 0:
                    ax.plot(eig.real, eig.imag, "o", c=color, ms=ms_vals[idx])
                else:
                    ax.plot(eig.imag, eig.real, "o", c=color, ms=ms_vals[idx])

    # PLOTS 4-6: Plot the DMD modes.
    for i, (ax, idx) in enumerate(zip(mode_axes, index_modes)):
        ax.set_title(
            f"Mode {idx + 1}", c=main_colors[i], fontsize=title_fontsize
        )
        # Plot modes in 1D.
        if len(snapshots_shape) == 1:
            ax.plot(lead_modes[:, idx].real, c=mode_color)
        # Plot modes in 2D.
        else:
            mode = lead_modes[:, idx].reshape(*snapshots_shape, order=order)
            # Multiply by factor of 0.9 to intensify the plotted image.
            vmax = 0.9 * np.abs(mode.real).max()
            im = ax.imshow(mode.real, vmax=vmax, vmin=-vmax, cmap=mode_cmap)
            # Align the colorbar with the plotted image.
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.05)
            fig.colorbar(im, cax=cax)
            if remove_cmap_ticks:
                ax.set_xticks([])
                ax.set_yticks([])

    # PLOTS 7-9: Plot the DMD mode dynamics.
    for i, (ax, idx) in enumerate(zip(dynamics_axes, index_modes)):
        dynamics_data = lead_dynamics[idx].real
        ax.set_title("Mode Dynamics", c=main_colors[i], fontsize=title_fontsize)
        ax.plot(time, dynamics_data, c=dynamics_color)
        ax.set_xlabel("Time", fontsize=label_fontsize)
        dynamics_range = dynamics_data.max() - dynamics_data.min()
        # Re-adjust ylim if dynamics oscillations are extremely small.
        if dynamics_range / np.abs(np.average(dynamics_data)) < 1e-4:
            ax.set_ylim([0.0, 2 * np.average(dynamics_data)])

    # Padding between elements.
    if tight_layout_kwargs is None:
        tight_layout_kwargs = {}
    plt.tight_layout(**tight_layout_kwargs)

    # Save plot if filename is provided.
    if filename:
        plt.savefig(filename)
        plt.close(fig)
    else:
        plt.show()
