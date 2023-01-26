import numpy as np
import pytest

from pydmd.snapshots import Snapshots


def test_1d():
    with pytest.raises(ValueError):
        Snapshots(np.ones(10))


def test_2d():
    X = np.random.rand(10, 5)
    snapshots = Snapshots(X)
    assert np.all(snapshots.snapshots == X)
    assert snapshots.snapshots_shape == (10,)


def test_3d():
    X = np.random.rand(10, 3, 5)
    snapshots = Snapshots(X)
    assert np.all(snapshots.snapshots == X.reshape((-1, 5)))
    assert snapshots.snapshots_shape == (10, 3)


def test_4d():
    X = np.random.rand(10, 3, 2, 5)
    snapshots = Snapshots(X)
    assert np.all(snapshots.snapshots == X.reshape((-1, 5)))
    assert snapshots.snapshots_shape == (10, 3, 2)


def test_list_1d():
    X = [1, 2, 3, 4]
    with pytest.raises(ValueError):
        Snapshots(X)


def test_list_2d():
    X = [[1, 2], [2, 3], [3, 4], [4, 5]]
    snapshots = Snapshots(X)
    assert snapshots.snapshots_shape == (2,)
    assert np.all(snapshots.snapshots == [[1, 2, 3, 4], [2, 3, 4, 5]])


def test_list_3d():
    X = [
        [[1, 2, 0], [2, 3, 0]],
        [[3, 4, 0], [4, 5, 0]],
        [[5, 6, 0], [6, 7, 0]],
        [[7, 8, 0], [8, 9, 0]],
    ]
    snapshots = Snapshots(X)
    assert snapshots.snapshots_shape == (2, 3)
    assert np.all(
        snapshots.snapshots
        == [
            [1, 3, 5, 7],
            [2, 4, 6, 8],
            [0, 0, 0, 0],
            [2, 4, 6, 8],
            [3, 5, 7, 9],
            [0, 0, 0, 0],
        ]
    )
