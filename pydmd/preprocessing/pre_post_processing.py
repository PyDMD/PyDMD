"""
Pre/post-processing capability for DMD instances.
"""

from __future__ import annotations

from inspect import isroutine
from typing import Any, Generic, TypeVar

import numpy as np

from pydmd.dmdbase import DMDBase

# Pre-processing output type
S = TypeVar("S")


class PrePostProcessing(Generic[S]):
    def pre_processing(self, X: np.ndarray) -> tuple[S, np.ndarray]:
        return None, X

    def post_processing(
        self, pre_processing_output: S, Y: np.ndarray
    ) -> np.ndarray:
        return Y


class PrePostProcessingDMD(Generic[S]):
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
        pre_post_processing: PrePostProcessing[S] = PrePostProcessing(),
    ):
        if dmd is None:
            raise ValueError("DMD instance cannot be None")
        self._pre_post_processing = pre_post_processing

        self._dmd = dmd
        self._pre_processing_output: S | None = None

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            pass

        if "fit" == name:
            return self._pre_processing_fit

        if "reconstructed_data" == name:
            return self._reconstructed_data_with_post_processing()

        # This check is needed to allow copy/deepcopy
        if name != "_dmd":
            sub_dmd = self._dmd
            if isinstance(sub_dmd, PrePostProcessingDMD):
                return PrePostProcessingDMD.__getattribute__(sub_dmd, name)
            return object.__getattribute__(self._dmd, name)
        return None

    @property
    def pre_post_processed_dmd(self):
        """
        Return the pre/post-processed DMD instance.

        :return: decorated DMD instance.
        :rtype: pydmd.DMDBase
        """
        return self._dmd

    @property
    def modes_activation_bitmask(self):
        return self._dmd.modes_activation_bitmask

    @modes_activation_bitmask.setter
    def modes_activation_bitmask(self, value):
        self._dmd.modes_activation_bitmask = value

    def _pre_processing_fit(self, *args, **kwargs):
        X = PrePostProcessingDMD._extract_training_data(*args, **kwargs)
        self._pre_processing_output, pre_processed_training_data = (
            self._pre_post_processing.pre_processing(X)
        )
        new_args, new_kwargs = PrePostProcessingDMD._replace_training_data(
            pre_processed_training_data, *args, **kwargs
        )
        return self._dmd.fit(*new_args, **new_kwargs)

    def _reconstructed_data_with_post_processing(self) -> np.ndarray:
        data = self._dmd.reconstructed_data

        if not isroutine(data):
            return self._pre_post_processing.post_processing(
                self._pre_processing_output,
                data,
            )

        # e.g. DMDc
        def output(*args, **kwargs) -> np.ndarray:
            return self._pre_post_processing.post_processing(
                self._pre_processing_output,
                data(*args, **kwargs),
            )

        return output

    @staticmethod
    def _extract_training_data(*args, **kwargs):
        if len(args) >= 1:
            return args[0]
        elif "X" in kwargs:
            return kwargs["X"]
        raise ValueError(
            f"Could not extract training data from {args=}, {kwargs=}"
        )

    @staticmethod
    def _replace_training_data(
        new_training_data: Any, *args, **kwargs
    ) -> [tuple[Any, ...], dict[str, Any]]:
        if len(args) >= 1:
            return (new_training_data,) + args[1:], kwargs
        elif "X" in kwargs:
            new_kwargs = dict(kwargs)
            new_kwargs["X"] = new_training_data
            return args, new_kwargs
