"""
Pre/post-processing capability for DMD instances.
"""

from typing import Callable, Dict
from pydmd.dmdbase import DMDBase


def _shallow_preprocessing(_: Dict, *args, **kwargs):
    return args + tuple(kwargs.values())


def _shallow_postprocessing(_: Dict, *args):
    # The first item of args is always the output of dmd.reconstructed_data
    return args[0]


def _tuplify(value):
    if isinstance(value, tuple):
        return value
    return (value,)


class PrePostProcessingDMD:
    """
    Pre/post-processing decorator. This class is not thread-safe in case of
    stateful transformations.

    :param dmd: DMD instance to be decorated.
    :param pre_processing: Pre-processing function, receives a state holder
        `dict` for stateful preprocessing, and positional/keyword arguments
        passed to`DMDBase.fit()`. The returned values from this function are
        passed exactly in the same order to the wrapped DMD instance.
    :param post_processing: Post-processing function, receives the state
        holder created during the pre-processing and the value of the
        reconstructed data from the wrapped DMD instance.
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

        self._pre_post_processed_dmd = dmd
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
                self._state_holder,
                self._pre_post_processed_dmd.reconstructed_data,
            )
            return output

        # This check is needed to allow copy/deepcopy
        if name != "_pre_post_processed_dmd":
            sub_dmd = self._pre_post_processed_dmd
            if isinstance(sub_dmd, PrePostProcessingDMD):
                return PrePostProcessingDMD.__getattribute__(sub_dmd, name)
            return object.__getattribute__(self._pre_post_processed_dmd, name)
        return None

    @property
    def modes_activation_bitmask(self):
        return self._pre_post_processed_dmd.modes_activation_bitmask

    @modes_activation_bitmask.setter
    def modes_activation_bitmask(self, value):
        self._pre_post_processed_dmd.modes_activation_bitmask = value

    def _pre_processing_fit(self, *args, **kwargs):
        self._state_holder = dict()
        pre_processing_output = _tuplify(
            self._pre_processing(self._state_holder, *args, **kwargs)
        )
        return self._pre_post_processed_dmd.fit(*pre_processing_output)
