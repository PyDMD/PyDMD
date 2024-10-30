"""
Randomized pre-processing.
"""

from __future__ import annotations

import sys
from numbers import Number

import numpy as np

from pydmd.dmdbase import DMDBase
from pydmd.preprocessing.pre_post_processing import (
    PrePostProcessing,
    PrePostProcessingDMD,
)
from pydmd.utils import compute_rqb

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


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
    pre_post_processing = _RandomizedPrePostProcessing(
        svd_rank=svd_rank,
        oversampling=oversampling,
        power_iters=power_iters,
        test_matrix=test_matrix,
        seed=seed,
    )
    return PrePostProcessingDMD(dmd, pre_post_processing)


class _RandomizedPrePostProcessing(PrePostProcessing):
    def __init__(self, **kwargs):
        self._compute_rqb_kwargs = kwargs

    @override
    def pre_processing(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Q = compute_rqb(X, **self._compute_rqb_kwargs).Q
        compression_matrix = Q.conj().T
        return compression_matrix, compression_matrix.dot(X)

    @override
    def post_processing(
        self, pre_processing_output: np.ndarray, Y: np.ndarray
    ) -> np.ndarray:
        return pre_processing_output.conj().T.dot(Y)
