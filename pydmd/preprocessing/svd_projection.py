"""
SVD projection pre-processing.
"""

import sys
import warnings
from numbers import Number
from typing import Tuple

import numpy as np

from pydmd.dmdbase import DMDBase
from pydmd.preprocessing.pre_post_processing import (
    PrePostProcessing,
    PrePostProcessingDMD,
)
from pydmd.utils import compute_svd

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


def svd_projection_preprocessing(dmd: DMDBase, svd_rank: Number):
    """
    SVD projection pre-processing.

    :param dmd: DMD instance to be wrapped.
    :param svd_rank: SVD rank argument passed to :func:`pydmd.utils.compute_svd`
        to compute the projection matrix.
    """
    return PrePostProcessingDMD(dmd, _SvdProjectionPrePostProcessing(svd_rank))


class _SvdProjectionPrePostProcessing(PrePostProcessing):
    def __init__(self, svd_rank: Number):
        self._svd_rank = svd_rank

    @override
    def pre_processing(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        space_dim = X.shape[0]
        if space_dim == 1:
            svd_rank = -1
            warnings.warn(
                (
                    f"The parameter {svd_rank=} has "
                    "been ignored because the given system is a scalar function"
                )
            )
        else:
            svd_rank = self._svd_rank
        projection_matrix, _, _ = compute_svd(X, svd_rank)

        return projection_matrix, projection_matrix.T.dot(X)

    @override
    def post_processing(
        self, pre_processing_output: np.ndarray, Y: np.ndarray
    ) -> np.ndarray:
        return pre_processing_output.dot(Y)
