from typing import Dict, Union
from functools import partial
import warnings
import numpy as np
from pydmd.dmdbase import DMDBase
from pydmd.preprocessing import PrePostProcessingDMD
from pydmd.utils import compute_svd

svd_rank_type = Union[int, float]


def svd_projection_preprocessing(dmd: DMDBase, svd_rank: svd_rank_type):
    """
    SVD projection pre-processing.

    :param dmd: DMD instance to be wrapped.
    :param svd_rank: SVD rank argument passed to :func:`pydmd.utils.compute_svd`
        to compute the projection matrix.
    """
    return PrePostProcessingDMD(dmd, partial(_pre, svd_rank=svd_rank), _post)


def _pre(state: Dict, X: np.ndarray, svd_rank: svd_rank_type, **kwargs):
    space_dim = X.shape[0]
    if space_dim == 1:
        svd_rank = -1
        warnings.warn(
            (
                f"The parameter 'svd_rank_extra={svd_rank}' has "
                "been ignored because the given system is a scalar function"
            )
        )
    state["projection_matrix"], _, _ = compute_svd(X, svd_rank)

    return (state["projection_matrix"].T.dot(X),) + tuple(kwargs.values())


def _post(state: Dict, X: np.ndarray) -> np.ndarray:
    return state["projection_matrix"].dot(X)
