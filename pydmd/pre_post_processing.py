"""
Pre/post-processing capability for DMD instances.
"""

from typing import Callable
import numpy as np
from .dmdbase import DMDBase


def _shallow_preprocessing(*args, **kwargs):
    return None


def _identity(*args):
    # The first item of args is always the output of dmd.reconstructed_data
    return args[0]


class PrePostProcessingDMD:
    """
    Pre/post-processing decorator. This class is not thread-safe in case of
    stateful transformations.

    :param dmd: DMD instance to be decorated.
    :type dmd: DMDBase
    :param pre_processing: Pre-processing function, receives positional and
        keyword arguments passed to `DMDBase.fit()`. The pre-processing should
        happen in-place (e.g. by modifying directly the array instead of
        assigning a new reference). The returned value of this function is kept
        in memory and passed to `post_processing` for stateful pre-processings.
        By default this is an empty transformation.
    :type pre_processing: Callable
    :param post_processing: Post-processing function, receives the result of
        `DMDBase.reconstructed_data` and the output value of `pre_processing`
        if not `None`. The returned value is forwarded to the caller of
        `PrePostProcessingDMD.reconstructed_data`. By default this is the
        identity function.
    :type post_processing: Callable
    """

    def __init__(
        self,
        dmd: DMDBase,
        pre_processing: Callable = _shallow_preprocessing,
        post_processing: Callable = _identity,
    ):
        if dmd is None:
            raise ValueError("DMD instance cannot be None")
        if pre_processing is None:
            pre_processing = _shallow_preprocessing
        if post_processing is None:
            post_processing = _identity

        self._dmd = dmd
        self._pre_processing = pre_processing
        self._post_processing = post_processing
        self._pre_processing_output = None

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except:
            pass

        if "fit" == name:
            return self._pre_processing_fit

        if "reconstructed_data" == name:
            post_processing_args = (self._dmd.reconstructed_data,)
            if self._pre_processing_output is not None:
                post_processing_args += (self._pre_processing_output,)
            return self._post_processing(*post_processing_args)

        return self._dmd.__getattribute__(name)

    def _pre_processing_fit(self, *args, **kwargs):
        self._pre_processing_output = self._pre_processing(*args, **kwargs)
        return self._dmd.fit(*args, **kwargs)


def zero_mean_preprocessing(dmd: DMDBase):
    """
    Zero-mean pre-processing.

    :param dmd: DMD instance to be wrapped.
    :type dmd: DMDBase
    """

    def pre(X):
        mean = np.mean(X)
        X[:] -= mean
        return mean

    def post(X, mean):
        return X + mean

    return PrePostProcessingDMD(dmd, pre, post)
