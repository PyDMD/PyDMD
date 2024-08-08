"""
Randomized pre-processing.
"""

from functools import partial
from numbers import Number
from typing import Dict

import numpy as np

from pydmd.dmdbase import DMDBase
from pydmd.preprocessing import PrePostProcessingDMD
from pydmd.utils import compute_rqb


def randomized_preprocessing(
    dmd: DMDBase,
    svd_rank: Number,
    oversampling: int = 10,
    power_iters: int = 2,
    test_matrix: np.ndarray = None,
    seed: int = None,
):
    """
    Randomized QB pre-processing.

    :param dmd: DMD instance to be wrapped.
    :param svd_rank: target rank of the input data.
    :param oversampling: amount to oversample beyond the target rank.
    :param power_iters: number of power iterations to perform.
    :param test_matrix: optional custom random test matrix.
    :param seed: optional random generator seed.
    """
    return PrePostProcessingDMD(
        dmd,
        partial(
            _pre,
            svd_rank=svd_rank,
            oversampling=oversampling,
            power_iters=power_iters,
            test_matrix=test_matrix,
            seed=seed,
        ),
        _post,
    )


def _pre(
    state: Dict,
    X: np.ndarray,
    svd_rank: Number,
    oversampling: int,
    power_iters: int,
    test_matrix: np.ndarray,
    seed: int,
    **kwargs,
):
    Q = compute_rqb(
        X,
        svd_rank,
        oversampling,
        power_iters,
        test_matrix,
        seed,
    )[0]
    state["compression_matrix"] = Q.conj().T

    return (state["compression_matrix"].dot(X),) + tuple(kwargs.values())


def _post(state: Dict, X: np.ndarray) -> np.ndarray:
    return state["compression_matrix"].conj().T.dot(X)
