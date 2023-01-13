"""
Module for DMD plotting.
"""
from os.path import splitext

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pydmd import MrDMD

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
        colors = [
            cmap(i) for i in np.linspace(0, 1, len(dmd.dmd_tree.levels))
        ]

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
        labels = [
            f"Eigenvalues - level {i}"
            for i in range(dmd.max_level)
        ]

    if show_unit_circle:
        points += [unit_circle]
        labels += ["Unit circle"]

    ax.add_artist(plt.legend(points, labels, loc="best"))
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
