"""
Zero-mean pre-processing.
"""

from functools import partial
from typing import Dict, Iterable

import numpy as np

from pydmd.dmdbase import DMDBase
from pydmd.preprocessing import PrePostProcessingDMD


def zero_mean_preprocessing(dmd: DMDBase, axis=1, weights=None):
    """
    Zero-mean pre-processing.

    :param dmd: DMD instance to be wrapped.
    """
    return PrePostProcessingDMD(
        dmd, partial(_pre, axis=axis, weights=weights), _post
    )


def _pre(
    state: Dict, X: np.ndarray, axis: int, weights: Iterable[float], **kwargs
):
    state["mean"] = np.average(X, axis=axis, weights=weights)
    if axis:
        state["mean"] = np.expand_dims(state["mean"], axis=axis)
    return (X - state["mean"],) + tuple(kwargs.values())


def _post(state: Dict, X: np.ndarray) -> np.ndarray:
    return X + state["mean"]
