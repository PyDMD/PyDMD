"""
SVD projection pre-processing.
"""

from typing import Dict, Union
from functools import partial
import warnings
import numpy as np
from pydmd.dmdbase import DMDBase
from pydmd.preprocessing import PrePostProcessingDMD
from pydmd.utils import compute_svd
from pydmd.linalg import build_linalg_module

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
    space_dim = X.shape[-2]
    if space_dim == 1:
        svd_rank = -1
        warnings.warn(
            (
                f"The parameter 'svd_rank_extra={svd_rank}' has "
                "been ignored because the given system is a scalar function"
            )
        )

    proj, _, _ = compute_svd(X, svd_rank)
    linalg_module = build_linalg_module(X)
    projected = linalg_module.dot(proj.swapaxes(-1, -2), X)

    state["projection_matrix"] = proj
    return (projected,) + tuple(kwargs.values())


def _post(state: Dict, X: np.ndarray) -> np.ndarray:
    proj = state["projection_matrix"]
    if proj.ndim == 3:
        proj = proj[:, None]

    linalg_module = build_linalg_module(X)
    return linalg_module.dot(proj, X).swapaxes(-1, -2)
