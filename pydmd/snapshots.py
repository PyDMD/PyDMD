"""
Module for snapshots normalization.
"""
import warnings
import logging

import numpy as np

from pydmd.linalg import build_linalg_module, cast_as_array


class Snapshots:
    """
    Utility class to preprocess snapshots shape for DMD.

    This class expects the time to be the last dimensions of the array.
    If a Python list is passed to the constructor, each element in the
    list is assumed to be a snapshot in time.

    Space dimensions are flattened (C-order) such that the
    matrix becomes 2D (time changes along the last axis).

    :param numpy.array | list(numpy.array) X: Training snapshots.
    """

    def __init__(self, X):
        (
            self._snapshots,
            self._snapshots_shape,
        ) = Snapshots._unroll_space_dimensions(X)

        if self._snapshots.shape[-1] == 1:
            raise ValueError("Received only one time snapshot.")

        logging.info(
            "Snapshots: %s, snapshot shape: %s",
            self._snapshots.shape,
            self._snapshots_shape,
        )

    @staticmethod
    def _unroll_space_dimensions(X):
        if hasattr(X, "ndim"):
            if X.ndim == 1:
                raise ValueError(
                    "Expected at least a 2D matrix (space x time)."
                )

            linalg_module = build_linalg_module(X)
            snapshots = linalg_module.reshape(X, (len(X), -1))

            return snapshots, X.shape[1:]
        else:
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

    def transpose(self):
        """
        Compute a new set of snapshots by permuting space and time.
        """
        return Snapshots(self.snapshots.T)

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
