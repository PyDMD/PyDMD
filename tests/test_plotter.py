import os

import matplotlib.pyplot as plt
import numpy as np
from pytest import raises

from pydmd import DMD, DMDBase, MrDMD
from pydmd.plotter import (
    _enforce_ratio,
    _plot_limits,
    plot_eigs,
    plot_eigs_mrdmd,
    plot_modes_2D,
    plot_snapshots_2D,
    plot_summary,
)

from .test_mrdmd import sample_data as mrdmd_sample_data

sample_data = np.load("tests/test_datasets/input_sample.npy")


def test_enforce_ratio_y():
    supx, infx, supy, infy = _enforce_ratio(10, 20, 10, 0, 0)

    dx = supx - infx
    dy = supy - infy
    np.testing.assert_almost_equal(max(dx, dy) / min(dx, dy), 10, decimal=6)


def test_enforce_ratio_x():
    supx, infx, supy, infy = _enforce_ratio(10, 0, 0, 20, 10)

    dx = supx - infx
    dy = supy - infy
    np.testing.assert_almost_equal(max(dx, dy) / min(dx, dy), 10, decimal=6)


def test_plot_limits_narrow():
    dmd = DMDBase()
    dmd.operator._eigenvalues = np.array([complex(1, 2), complex(-1, -2)])
    dmd.operator._modes = np.array(np.ones((10, 2)))

    tp = _plot_limits(dmd, True)

    assert len(tp) == 4

    supx, infx, supy, infy = tp
    assert supx == 1.05
    assert infx == -1.05
    assert supy == 2.05
    assert infy == -2.05


def test_plot_limits():
    dmd = DMDBase()
    dmd.operator._eigenvalues = np.array([complex(-2, 2), complex(3, -3)])
    dmd.operator._modes = np.array(np.ones((10, 2)))

    limit = _plot_limits(dmd, False)
    assert limit == 5


def test_plot_eigs():
    dmd = DMDBase()
    with raises(ValueError):
        plot_eigs(dmd, show_axes=True, show_unit_circle=True)


def test_plot_eigs_narrowview_empty():
    dmd = DMDBase()
    # max/min throws an error if the array is empty (max used on empty
    # array)
    dmd.operator._eigenvalues = np.array([], dtype=complex)
    with raises(ValueError):
        plot_eigs(dmd, show_axes=False, narrow_view=True, dpi=200)


def test_plot_modes_2D():
    dmd = DMDBase()
    with raises(ValueError):
        plot_modes_2D(dmd)


def test_plot_snaps_2D():
    dmd = DMDBase()
    with raises(ValueError):
        plot_snapshots_2D(dmd)


def test_plot_eigs_1():
    dmd = DMD()
    dmd.fit(X=sample_data)
    plot_eigs(dmd, show_axes=True, show_unit_circle=True)
    plt.close()


def test_plot_eigs_2():
    dmd = DMD()
    dmd.fit(X=sample_data)
    plot_eigs(dmd, show_axes=False, show_unit_circle=False)
    plt.close()


def test_plot_eigs_3():
    dmd = DMD()
    dmd.fit(X=sample_data)
    plot_eigs(dmd, show_axes=False, show_unit_circle=True, filename="eigs.png")
    os.remove("eigs.png")


def test_plot_modes_1():
    dmd = DMD()
    dmd.fit(X=sample_data)
    with raises(ValueError):
        plot_modes_2D(dmd)


def test_plot_modes_2():
    dmd = DMD(svd_rank=-1)
    dmd.fit(X=sample_data)
    plot_modes_2D(dmd, index_mode=(1, 2, 5), x=np.arange(20), y=np.arange(20))
    plt.close()


def test_plot_modes_3():
    dmd = DMD()
    snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
    dmd.fit(X=snapshots)
    plot_modes_2D(dmd, snapshots_shape=(20, 20))
    plt.close()


def test_plot_modes_4():
    dmd = DMD()
    snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
    dmd.fit(X=snapshots)
    plot_modes_2D(dmd, snapshots_shape=(20, 20), index_mode=1)
    plt.close()


def test_plot_modes_5():
    dmd = DMD()
    snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
    dmd.fit(X=snapshots)
    plot_modes_2D(
        dmd, snapshots_shape=(20, 20), index_mode=1, filename="tmp.png"
    )
    os.remove("tmp.1.png")


def test_plot_snapshots_1():
    dmd = DMD()
    dmd.fit(X=sample_data)
    with raises(ValueError):
        plot_snapshots_2D(dmd)


def test_plot_snapshots_2():
    dmd = DMD(svd_rank=-1)
    dmd.fit(X=sample_data)
    plot_snapshots_2D(dmd, (1, 2, 5), x=np.arange(20), y=np.arange(20))
    plt.close()


def test_plot_snapshots_3():
    dmd = DMD()
    snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
    dmd.fit(X=snapshots)
    plot_snapshots_2D(dmd, snapshots_shape=(20, 20))
    plt.close()


def test_plot_snapshots_4():
    dmd = DMD()
    snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
    dmd.fit(X=snapshots)
    plot_snapshots_2D(dmd, snapshots_shape=(20, 20), index_snap=2)
    plt.close()


def test_plot_snapshots_5():
    dmd = DMD()
    snapshots = [snap.reshape(20, 20) for snap in sample_data.T]
    dmd.fit(X=snapshots)
    plot_snapshots_2D(
        dmd, snapshots_shape=(20, 20), index_snap=2, filename="tmp.png"
    )
    os.remove("tmp.2.png")


def test_tdmd_plot():
    dmd = DMD(tlsq_rank=3)
    dmd.fit(X=sample_data)
    plot_eigs(dmd, show_axes=False, show_unit_circle=False)
    plt.close()


def test_mrdmd_wrong_plot_eig1():
    rank = 2
    dmd = DMD(svd_rank=rank)
    mrdmd = MrDMD(dmd, max_level=6, max_cycles=2)
    mrdmd.fit(X=mrdmd_sample_data)
    with raises(ValueError):
        plot_eigs_mrdmd(
            mrdmd,
            show_axes=True,
            show_unit_circle=True,
            figsize=(8, 8),
            level=7,
        )


def test_mrdmd_plot_eig1():
    dmd = DMD()
    mrdmd = MrDMD(dmd, max_level=6, max_cycles=2)
    mrdmd.fit(X=mrdmd_sample_data)
    plot_eigs_mrdmd(
        mrdmd, show_axes=True, show_unit_circle=True, figsize=(8, 8)
    )
    plt.close()


def test_mrdmd_plot_eig2():
    dmd = DMD()
    mrdmd = MrDMD(dmd, max_level=6, max_cycles=2)
    mrdmd.fit(X=mrdmd_sample_data)
    plot_eigs_mrdmd(
        mrdmd, show_axes=True, show_unit_circle=False, title="Title"
    )
    plt.close()


def test_mrdmd_plot_eig3():
    dmd = DMD()
    mrdmd = MrDMD(dmd, max_level=6, max_cycles=2)
    mrdmd.fit(X=mrdmd_sample_data)
    plot_eigs_mrdmd(
        mrdmd, show_axes=False, show_unit_circle=False, level=1, node=0
    )
    plt.close()


def test_plot_summary_1():
    # Test 1: Throw error if more than 3 modes are requested.
    dmd = DMD()
    dmd.fit(X=sample_data)
    with raises(ValueError):
        plot_summary(dmd, index_modes=[0, 1, 2, 3])


def test_plot_summary_2():
    # Test 2: Everything is fine with all default settings.
    dmd = DMD()
    dmd.fit(X=sample_data)
    plot_summary(dmd)
    plt.close()


def test_plot_summary_3():
    # Test 3: Everything is fine when plotting in 2D.
    dmd = DMD()
    dmd.fit(X=sample_data)
    plot_summary(dmd, snapshots_shape=(20, 20))
    plt.close()


def test_plot_summary_4():
    # Test 4: Everything is fine when plotting fewer than 3 modes.
    dmd = DMD(svd_rank=2)
    dmd.fit(X=sample_data)
    plot_summary(dmd, snapshots_shape=(20, 20))
    plt.close()


def test_plot_summary_5():
    # Test 5: Everything is fine when saving the plot.
    dmd = DMD()
    dmd.fit(X=sample_data)
    plot_summary(dmd, snapshots_shape=(20, 20), filename="tmp.png")
    os.remove("tmp.png")


# A non-square factorization of the 400-point spatial dimension of
# sample_data. Square snapshots hide a transposed grid, which is why the
# axis swap of issue #565 went unnoticed.
ROWS, COLS = 25, 16

# An asymmetric marker, so that a grid which is reversed or transposed cannot
# render the same picture as a correct one.
MARKER = np.zeros((ROWS, COLS))
MARKER[1:4, 1:3] = 1.0


def marker_snapshots():
    """Snapshots carrying MARKER, scaled so the leading mode reproduces it."""
    return [MARKER * (1.0 + 0.01 * k) for k in range(sample_data.shape[1])]


def rendered_grids(axes):
    """
    The C array each axis handed to pcolor, in the shape it was drawn.
    """
    grids = []
    for ax in axes:
        for collection in ax.collections:
            array = collection.get_array()
            if array is not None:
                coords = np.asarray(collection.get_coordinates())
                # Corner coordinates must run left to right and bottom to top,
                # otherwise the values below are drawn mirrored.
                assert np.all(np.diff(coords[..., 0], axis=1) > 0)
                assert np.all(np.diff(coords[..., 1], axis=0) > 0)
                grids.append(np.asarray(array))
    assert grids, "no pcolor mesh was drawn, so there is nothing to check"
    return grids


def axes_extents(axes):
    """
    The x and y limits of every given axis.
    """
    extents = [(ax.get_xlim(), ax.get_ylim()) for ax in axes]
    assert extents, "no axes were left open, so there is nothing to compare"
    return extents


def test_plot_modes_2D_default_grid():
    # Omitting x and y must be equivalent to passing the grid implied by the
    # snapshot shape: x spans the columns, y spans the rows. See issue #565.
    dmd = DMD(svd_rank=1)
    dmd.fit(X=marker_snapshots())

    plot_modes_2D(dmd, index_mode=0)
    default_extents = axes_extents(plt.gcf().axes)
    default_grids = rendered_grids(plt.gcf().axes)
    plt.close()

    plot_modes_2D(dmd, index_mode=0, x=np.arange(COLS), y=np.arange(ROWS))
    assert default_extents == axes_extents(plt.gcf().axes)
    plt.close()

    # The drawn mode must carry MARKER in its original orientation, which a
    # transposed or reversed grid cannot produce.
    drawn = np.abs(default_grids[0])
    assert drawn.shape == (ROWS, COLS)
    hot = drawn > 0.5 * drawn.max()
    assert list(np.flatnonzero(hot.any(axis=1))) == [1, 2, 3]
    assert list(np.flatnonzero(hot.any(axis=0))) == [1, 2]


def test_plot_snapshots_2D_default_grid():
    dmd = DMD(svd_rank=1)
    dmd.fit(X=marker_snapshots())

    plot_snapshots_2D(dmd, index_snap=0)
    default_extents = axes_extents(plt.gcf().axes)
    default_grids = rendered_grids(plt.gcf().axes)
    plt.close()

    plot_snapshots_2D(dmd, index_snap=0, x=np.arange(COLS), y=np.arange(ROWS))
    assert default_extents == axes_extents(plt.gcf().axes)
    plt.close()

    drawn = np.abs(default_grids[0])
    assert drawn.shape == (ROWS, COLS)
    hot = drawn > 0.5 * drawn.max()
    assert list(np.flatnonzero(hot.any(axis=1))) == [1, 2, 3]
    assert list(np.flatnonzero(hot.any(axis=0))) == [1, 2]
