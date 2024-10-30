"""
Zero-mean pre-processing.
"""

import sys
from typing import Tuple

import numpy as np

from pydmd.dmdbase import DMDBase
from pydmd.preprocessing.pre_post_processing import (
    PrePostProcessing,
    PrePostProcessingDMD,
)

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


def zero_mean_preprocessing(dmd: DMDBase, *args, **kwargs):
    """
    Zero-mean pre-processing. All exceeding arguments are passed to
    `np.average`.

    :param dmd: DMD instance to be wrapped.
    """

    return PrePostProcessingDMD(
        dmd, _ZeroMeanPrePostProcessing(*args, **kwargs)
    )


class _ZeroMeanPrePostProcessing(PrePostProcessing):
    def __init__(self, *args, **kwargs):
        self._average_args = args
        self._average_kwargs = kwargs

    @override
    def pre_processing(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if "keepdims" not in self._average_kwargs:
            self._average_kwargs["keepdims"] = True
        mean = np.average(X, *self._average_args, **self._average_kwargs)
        return mean, X - mean

    @override
    def post_processing(
        self, pre_processing_output: np.ndarray, Y: np.ndarray
    ) -> np.ndarray:
        return Y + pre_processing_output
