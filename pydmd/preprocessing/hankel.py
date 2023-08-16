"""
Hankel pre-processing.
"""

from typing import Dict, Union, List, Tuple
from functools import partial
import numpy as np
from pydmd.dmdbase import DMDBase
from pydmd.preprocessing.pre_post_processing import PrePostProcessingDMD
from pydmd.utils import pseudo_hankel_matrix

_reconstruction_method_type = Union[str, np.ndarray, List, Tuple]


def hankel_preprocessing(
    dmd: DMDBase,
    d: int,
    reconstruction_method: _reconstruction_method_type = "first",
):
    """
    Hankel pre-processing.

    :param dmd: DMD instance to be wrapped.
    :param d: Hankel matrix rank
    :param reconstruction_method: Reconstruction method.
    """
    return PrePostProcessingDMD(
        dmd,
        partial(_preprocessing, d=d),
        partial(
            _hankel_post_processing,
            d=d,
            reconstruction_method=reconstruction_method,
        ),
    )


def _preprocessing(_: Dict, X: np.ndarray, d: int, **kwargs):
    return (pseudo_hankel_matrix(X, d),) + tuple(kwargs.values())


def _reconstructions(rec: np.ndarray, d: int):
    """
    Build a collection of all the available versions of the given
    `timeindex`. The indexing of time instants is the same used for
    :func:`reconstructed_data`. For each time instant there are at least
    one and at most `d` versions. If `timeindex` is `None` the function
    returns the whole collection, for all the time instants.

    :param rec: reconstructed data.
    :param d: Hankel matrix rank.
    :param timeindex: The index of the time snapshot.
    :return: a collection of all the available versions for the given
        time snapshot, or for all the time snapshots if `timeindex` is
        `None` (in the second case, time varies along the first dimension
        of the array returned).
    """
    space_dim = rec.shape[-2] // d
    time_instants = rec.shape[-1] + d - 1

    rec_snapshots_shape = (time_instants, d, space_dim)
    reconstructed_snapshots = np.full(
        rec_snapshots_shape, np.nan, dtype=rec.dtype
    )
    time_idxes = np.arange(d)[None].repeat(rec.shape[-1], axis=0)
    time_idxes += np.arange(rec.shape[-1])[:, None]
    d_idxes = np.arange(d)[None].repeat(rec.shape[-1], axis=0)
    splitted = np.array(np.split(rec, d, axis=-2))
    splitted = np.moveaxis(splitted, -1, -3)

    reconstructed_snapshots[..., time_idxes, d_idxes, :] = splitted
    return reconstructed_snapshots


def _first_reconstructions(reconstructions: np.ndarray, d: int) -> np.ndarray:
    """Return the first occurrence of each snapshot available in the given
    matrix (which must be the result of `self._sub_dmd.reconstructed_data`,
    or have the same shape).

    :param reconstructions: A matrix of (higher-order) snapshots having
        shape `(space*self.d, time_instants)`.
    :param d: Hankel matrix rank.
    :return: The first snapshot that occurs in `reconstructions` for each
        available time instant.
    """
    n_time = reconstructions.shape[-3]
    time_idxes = np.arange(n_time)
    d_idxes = np.arange(n_time)
    d_idxes[d - 1 :] = d - 1
    return reconstructions[..., time_idxes, d_idxes, :].swapaxes(-1, -2)


def _hankel_post_processing(
    _: Dict,  # No state
    X: np.ndarray,
    d: int,
    reconstruction_method: _reconstruction_method_type,
) -> np.ndarray:
    rec = _reconstructions(X, d=d)
    rec = np.ma.array(rec, mask=np.isnan(rec))

    if reconstruction_method == "first":
        result = _first_reconstructions(rec, d=d)
    elif reconstruction_method == "mean":
        result = np.nanmean(rec, axis=1).T
    elif isinstance(reconstruction_method, (np.ndarray, list, tuple)):
        result = np.ma.average(rec, axis=1, weights=reconstruction_method).T
    else:
        raise ValueError(
            f"The reconstruction method wasn't recognized: {reconstruction_method}"
        )

    return result.filled(fill_value=0)
