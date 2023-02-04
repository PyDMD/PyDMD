"""
Module for snapshots normalization.
"""
import warnings
import logging
from functools import reduce
from operator import mul

import numpy as np

from pydmd.linalg import build_linalg_module, cast_as_array

def _prod(iter):
    """
    Equivalent to math.prod, compatible with Python 3.7
    """
    return reduce(mul, iter, 1)


class Snapshots:
    """
    Utility class to preprocess snapshots shape for DMD.

    This class expects the time to be the last dimensions of the array.
    If a Python list is passed to the constructor, each element in the
    list is assumed to be a snapshot in time.

    Space dimensions are flattened (C-order) such that the
    matrix becomes 2D (time changes along the last axis).

    :param numpy.array | list(numpy.array) X: Training snapshots.
    :param batch: If `True`, the first dimension is dedicated to batching.
    :type batch: bool
    """

    def __init__(self, X, batch=False):
        (
            self._snapshots,
            self._snapshots_shape,
        ) = Snapshots._unroll_space_dimensions(X, batch)

        if self._snapshots.shape[-1] == 1:
            raise ValueError("Received only one time snapshot.")

        Snapshots._check_condition_number(self._snapshots)

        logging.info(
            "Snapshots: %s, snapshot shape: %s",
            self._snapshots.shape,
            self._snapshots_shape,
        )

    @staticmethod
    def _unroll_space_dimensions(X, batch):
        if hasattr(X, "ndim"):
            if X.ndim == 1:
                raise ValueError(
                    "Expected at least a 2D matrix (space x time)."
                )

            if batch and X.ndim < 3:
                raise ValueError(
                    "Expected at least a 3D matrix for batched DMD."
                )

            n_batches, *space, time = X.shape
            if not batch:
                space = [n_batches] + space
                n_batches = 1

            linalg_module = build_linalg_module(X)
            snapshots = linalg_module.reshape(X, (n_batches, _prod(space), time))
            if not batch:
                snapshots = snapshots[0]

            return snapshots, tuple(space)
        else:
            if batch:
                raise ValueError(
                    "Batched DMD requires the input data to be "
                    "passed as a 3D PyTorch tensor."
                )

            snapshots = cast_as_array(X)
            if snapshots.ndim == 1:
                raise ValueError(
                    "Expected at least a 2D matrix (space x time)."
                )

            linalg_module = build_linalg_module(snapshots)
            snapshots_flattened = linalg_module.reshape(
                snapshots, (len(snapshots), -1)
            )

            return snapshots_flattened.swapaxes(-1, -2), snapshots.shape[1:]

    @staticmethod
    def _check_condition_number(X):
        cond_number = np.linalg.cond(X)
        if cond_number > 10e4:
            warnings.warn(
                f"Input data condition number {cond_number}. "
                """Consider preprocessing data, passing in augmented data
matrix, or regularization methods."""
            )

    @property
    def snapshots(self):
        """
        Snapshots of the system (space flattened).
        """
        return self._snapshots

    @property
    def snapshots_shape(self):
        """
        Original (i.e. non-flattened) snapshot shape (time is ignored).
        """
        return self._snapshots_shape
