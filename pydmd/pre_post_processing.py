from typing import Callable
import numpy as np
from .dmdbase import DMDBase


class PrePostProcessingDMD:
    def __init__(
        self, dmd: DMDBase, pre_processing: Callable, post_processing: Callable
    ):
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
    def pre(X):
        mean = np.mean(X)
        X[:] -= mean
        return mean

    def post(X, mean):
        return X + mean

    return PrePostProcessingDMD(dmd, pre, post)
