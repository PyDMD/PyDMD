"""
Randomized pre-processing.
"""

from functools import partial
from typing import Dict, Union

import numpy as np

from pydmd.dmdbase import DMDBase
from pydmd.preprocessing import PrePostProcessingDMD
from pydmd.utils import compute_rqb

svd_rank_type = Union[int, float]
seed_type = Union[None, int]

def randomized_preprocessing(
    dmd: DMDBase,
    svd_rank: svd_rank_type,
    oversampling: int,
    power_iters: int,
    test_matrix: np.ndarray = None,
    seed: seed_type = None,
):
    """
    Randomized QB pre-processing.

    :param dmd: DMD instance to be wrapped.
    """
    return PrePostProcessingDMD(
        dmd,
        partial(
            _rand_preprocessing,
            svd_rank=svd_rank,
            oversampling=oversampling,
            power_iters=power_iters,
            test_matrix=test_matrix,
            seed=seed,
        ),
        _rand_postprocessing,
    )

def _rand_preprocessing(
    state: Dict,
    X: np.ndarray,
    svd_rank: svd_rank_type,
    oversampling: int,
    power_iters: int,
    test_matrix: np.ndarray,
    seed: seed_type,
    **kwargs
):
    Q = compute_rqb(
        X, svd_rank, oversampling, power_iters, test_matrix, seed
    )[0]
    state["compression_matrix"] = Q.conj().T

    return (state["compression_matrix"].dot(X),) + tuple(kwargs.values())


def _rand_postprocessing(state: Dict, X: np.ndarray) -> np.ndarray:
    return state["compression_matrix"].conj().T.dot(X)
