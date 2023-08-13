"""
Pre/post-processing capability for DMD instances.
"""

from typing import Callable, Dict
import numpy as np
from .dmdbase import DMDBase


def _shallow_preprocessing(state: Dict, *args, **kwargs):
    return *args, *kwargs.values()


def _shallow_postprocessing(state: Dict, *args):
    # The first item of args is always the output of dmd.reconstructed_data
    return args[0]


def tuplify(value):
    if isinstance(value, tuple):
        return value
    return (value,)


class PrePostProcessingDMD:
    """
    Pre/post-processing decorator. This class is not thread-safe in case of
    stateful transformations.

    :param dmd: DMD instance to be decorated.
    :type dmd: DMDBase
    :param pre_processing: Pre-processing function, receives a state holder `dict`
        for stateful preprocessing, and positional/keyword arguments passed to
        `DMDBase.fit()`. The returned values from this function are passed exactly
        in the same order to the wrapped DMD instance.
    :type pre_processing: Callable
    :param post_processing: Post-processing function, receives the state holder
        created during the pre-processing and the value of the reconstructed data
        from the wrapped DMD instance.
    :type post_processing: Callable
    """

    def __init__(
        self,
        dmd: DMDBase,
        pre_processing: Callable = _shallow_preprocessing,
        post_processing: Callable = _shallow_postprocessing,
    ):
        if dmd is None:
            raise ValueError("DMD instance cannot be None")
        if pre_processing is None:
            pre_processing = _shallow_preprocessing
        if post_processing is None:
            post_processing = _shallow_postprocessing

        self._dmd = dmd
        self._pre_processing = pre_processing
        self._post_processing = post_processing
        self._state_holder = None

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except:
            pass

        if "fit" == name:
            return self._pre_processing_fit

        if "reconstructed_data" == name:
            output = self._post_processing(
                self._state_holder, self._dmd.reconstructed_data
            )
            self._state_holder = None
            return output

        return self._dmd.__getattribute__(name)

    def _pre_processing_fit(self, *args, **kwargs):
        self._state_holder = dict()
        pre_processing_output = tuplify(
            self._pre_processing(self._state_holder, *args, **kwargs)
        )
        return self._dmd.fit(*pre_processing_output)


def zero_mean_preprocessing(dmd: DMDBase):
    """
    Zero-mean pre-processing.

    :param dmd: DMD instance to be wrapped.
    :type dmd: DMDBase
    """

    def pre(state: Dict, X: np.ndarray, **kwargs):
        state["mean"] = np.mean(X)
        return X - state["mean"], *kwargs.values()

    def post(state: Dict, X: np.ndarray):
        return X + state["mean"]

    return PrePostProcessingDMD(dmd, pre, post)
