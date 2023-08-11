from .dmdbase import DMDBase
from typing import Callable


class PrePostProcessingDMD(object):
    def __init__(
        self, dmd: DMDBase, pre_processing: Callable, post_processing: Callable
    ):
        self._dmd = dmd
        self._pre_processing = pre_processing
        self._post_processing = post_processing

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except:
            pass

        if "fit" == name:
            return self._pre_processing_fit

        if "reconstructed_data" == name:
            if self._pre_processing_output is None:
                return self._post_processing(self._dmd.reconstructed_data)
            else:
                return self._post_processing(
                    self._dmd.reconstructed_data, self._pre_processing_output
                )

        return self._dmd.__getattribute__(name)

    def _pre_processing_fit(self, *args, **kwargs):
        self._pre_processing_output = self._pre_processing(*args, **kwargs)
        return self._dmd.fit(*args, **kwargs)
