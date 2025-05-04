"""
Decorator for pre- and post-processing a DMD model.

The purpose is to generate a pipeline that can look like:
data → pre-processing → fit (→ reconstruct) → post-processing

Where the reconstruction and processing hooks may or may not
be performed depending on the intent of the analysis.
"""

from __future__ import annotations

from inspect import isroutine
from typing import Any, Dict, Generic, Tuple, TypeVar

import numpy as np

from pydmd.dmdbase import DMDBase

# Generic type representing pre-processing state
S = TypeVar("S")


class PrePostProcessing(Generic[S]):
    """
    Interface for defining pre- and post-processing behavior.
    Override this class to add custom data transformations.
    """

    def pre_processing(self, X: np.ndarray) -> Tuple[S, np.ndarray]:
        """
        Apply transformations to input training data.

        :param X: Original input data.
        :return: A tuple of (state, transformed input).
        """
        return None, X

    def post_processing(
        self, pre_processing_output: S, Y: np.ndarray
    ) -> np.ndarray:
        """
        Apply transformations to reconstructed output.

        :param pre_processing_output: The state generated during pre-processing.
        :param Y: Output from wrapped DMD.
        :return: Transformed output.
        """
        return Y


class PrePostProcessingDMD(Generic[S]):
    """
    Decorator for a DMD instance that supports user-defined pre- and
    post-processing.

    Intended use:
      - Pre-process input data before `.fit()` is called
      - `undo` the processing step to return DMD components consistent
      with the original data.

    NOTE: This class is **not thread-safe** if the processing hooks are stateful.

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
        processing_hooks: PrePostProcessing[S] = PrePostProcessing(),
    ):
        if dmd is None:
            raise ValueError("DMD instance cannot be None")
        self._processing_hooks = processing_hooks

        self._wrapped_dmd = dmd
        self._processing_state: S | None = None

    def __getattribute__(self, name):
        """
        Transparent passthrough for almost all attribute access.

        Currently intercepts:
          - `.fit` by running pre-processing first
          - `.reconstructed_data` by wrapping with post-processing
        """
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            pass

        if "fit" == name:
            return self._fit_with_preprocessing

        if "reconstructed_data" == name:
            return self._reconstructed_data_with_post_processing()

        # This check is needed to allow copy/deepcopy
        if name != "_wrapped_dmd":
            wrapped = self._wrapped_dmd
            if isinstance(wrapped, PrePostProcessingDMD):
                return PrePostProcessingDMD.__getattribute__(wrapped, name)
            return object.__getattribute__(wrapped, name)
        return None

    @property
    def wrapped_dmd(self):
        """
        Return the warpped DMD instance.

        :return: decorated DMD instance.
        :rtype: pydmd.DMDBase
        """
        return self._wrapped_dmd

    # I am proposing to remove the activation bitmask as it heavily obscures
    # a basic function. Instead, I would expose this functionality in a way
    # that is consistent with the phsaor notation framework.
    # @property
    # def modes_activation_bitmask(self):
    #     return self._wrapped_dmd.modes_activation_bitmask

    # @modes_activation_bitmask.setter
    # def modes_activation_bitmask(self, value):
    #     self._wrapped_dmd.modes_activation_bitmask = value

    def _fit_with_preprocessing(self, *args, **kwargs):
        """
        Runs pre-processing before calling the wrapped DMD's fit method.
        Stores the pre-processing state for use during post-processing.
        """
        original_X = PrePostProcessingDMD._extract_training_data(
            *args, **kwargs
        )
        self._processing_state, transformed_X = (
            self._processing_hooks.pre_processing(original_X)
        )
        new_args, new_kwargs = PrePostProcessingDMD._replace_training_data(
            transformed_X, *args, **kwargs
        )
        return self._wrapped_dmd.fit(*new_args, **new_kwargs)

    def _reconstructed_data_with_post_processing(self) -> np.ndarray:
        """
        Returns the post-processed result of reconstructed_data.
        Handles both property and callable types.
        """
        data = self._wrapped_dmd.reconstructed_data

        if not isroutine(data):
            return self._processing_hooks.post_processing(
                self._processing_state,
                data,
            )

        # For DMDs where `reconstructed_data` is a callable, e.g. DMDc
        def wrapped_callable(*args, **kwargs) -> np.ndarray:
            return self._processing_hooks.post_processing(
                self._processing_state,
                data(*args, **kwargs),
            )

        return wrapped_callable

    @staticmethod
    def _extract_training_data(*args, **kwargs):
        """
        Locate the training data (X) from positional or keyword args.
        """
        if len(args) >= 1:
            return args[0]
        elif "X" in kwargs:
            return kwargs["X"]
        raise ValueError(
            f"Could not extract training data from {args=}, {kwargs=}"
        )

    @staticmethod
    def _replace_training_data(
        new_X: Any, *args, **kwargs
    ) -> [Tuple[Any, ...], Dict[str, Any]]:
        """
        Replace original training data with transformed version in the args/kwargs.
        """
        if len(args) >= 1:
            return (new_X,) + args[1:], kwargs
        elif "X" in kwargs:
            new_kwargs = dict(kwargs)
            new_kwargs["X"] = new_X
            return args, new_kwargs
