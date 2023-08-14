from typing import Dict
import numpy as np
from pydmd.dmdbase import DMDBase
from pydmd.preprocessing import PrePostProcessingDMD


def zero_mean_preprocessing(dmd: DMDBase):
    """
    Zero-mean pre-processing.

    :param dmd: DMD instance to be wrapped.
    """
    return PrePostProcessingDMD(dmd, _pre, _post)


def _pre(state: Dict, X: np.ndarray, **kwargs):
    state["mean"] = np.mean(X)
    return (X - state["mean"],) + tuple(kwargs.values())


def _post(state: Dict, X: np.ndarray) -> np.ndarray:
    return X + state["mean"]
