"""
Decorator for pre- and post-processing a DMD model.

The purpose is to generate a pipeline that can look like:
data → pre-processing → fit (→ reconstruct) → post-processing

Where the reconstruction and post-processing may or may not
be performed depending on the intent of the analysis.
"""

from __future__ import annotations

from inspect import isroutine
from typing import Any, Dict, Generic, Tuple, TypeVar

import numpy as np

from pydmd.dmdbase import DMDBase

# Generic type representing the pre-processing output
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
        :return: A tuple of (pre-processing output, transformed input).
        """
        return None, X

    def post_processing(
        self, pre_processing_output: S, Y: np.ndarray
    ) -> np.ndarray:
        """
        Apply transformations to reconstructed output.

        :param pre_processing_output: The output generated during
            pre-processing.
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

    Example, centering the data before fitting::

        class Center(PrePostProcessing):
            def pre_processing(self, X):
                mean = X.mean(axis=1, keepdims=True)
                return mean, X - mean  # (kept for later, transformed data)

            def post_processing(self, mean, Y):
                return Y + mean  # undo the transformation

        dmd = PrePostProcessingDMD(DMD(svd_rank=5), Center())
        dmd.fit(X)  # Center.pre_processing runs first
        dmd.reconstructed_data  # Center.post_processing is applied

    NOTE: This class is **not thread-safe** if the `PrePostProcessing`
    instance is stateful.

    :param dmd: DMD instance to be decorated.
    :param pre_post_processing: `PrePostProcessing` instance defining the
        pre- and post-processing.
    """

    def __init__(
        self,
        dmd: DMDBase,
        pre_post_processing: PrePostProcessing[S] = PrePostProcessing(),
    ):
        if dmd is None:
            raise ValueError("DMD instance cannot be None")
        self._pre_post_processing = pre_post_processing

        self._wrapped_dmd = dmd
        self._pre_processing_output: S | None = None

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
    def pre_post_processed_dmd(self):
        """
        Return the wrapped DMD instance.

        :return: decorated DMD instance.
        :rtype: pydmd.DMDBase
        """
        return self._wrapped_dmd

    @property
    def modes_activation_bitmask(self):
        return self._wrapped_dmd.modes_activation_bitmask

    @modes_activation_bitmask.setter
    def modes_activation_bitmask(self, value):
        self._wrapped_dmd.modes_activation_bitmask = value

    def _fit_with_preprocessing(self, *args, **kwargs):
        """
        Runs pre-processing before calling the wrapped DMD's fit method.

        Stores the pre-processing output for use during post-processing.
        """
        original_X = PrePostProcessingDMD._extract_training_data(
            *args, **kwargs
        )
        transformed_X = self._pre_process(original_X)
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
            return self._post_process(data)

        # For DMDs where `reconstructed_data` is a callable, e.g. DMDc
        def wrapped_callable(*args, **kwargs) -> np.ndarray:
            return self._post_process(data(*args, **kwargs))

        return wrapped_callable

    def _pre_process(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the user-defined pre-processing to the training data.

        Calls `pre_processing(X)` on the `PrePostProcessing` instance, which
        returns two values:

          1. The pre-processing output: whatever `post_processing` needs
             later to undo the transformation (e.g. the mean that was
             removed when centering). It is stored here, in
             `self._pre_processing_output`, and overwritten on every `fit`.
          2. The transformed training data. It is returned, so that it can
             be passed to the wrapped DMD's `fit` in place of `X`.

        :param X: Original training data.
        :return: Transformed training data.
        """
        self._pre_processing_output, transformed_X = (
            self._pre_post_processing.pre_processing(X)
        )
        return transformed_X

    def _post_process(self, Y: np.ndarray) -> np.ndarray:
        """
        Apply the user-defined post-processing to data from the wrapped DMD.

        Calls `post_processing(pre_processing_output, Y)` on the
        `PrePostProcessing` instance. `pre_processing_output` is the value
        stored by `_pre_process` during the most recent `fit`, which lets
        `post_processing` undo the transformation applied to the training
        data (e.g. add back the mean removed when centering).

        :param Y: Data computed by the wrapped DMD, e.g. its reconstructed
            data.
        :return: `Y` mapped back to the space of the original training data.
        """
        return self._pre_post_processing.post_processing(
            self._pre_processing_output, Y
        )

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
        Replace original training data with transformed version in the
        args/kwargs.
        """
        if len(args) >= 1:
            return (new_X,) + args[1:], kwargs
        elif "X" in kwargs:
            new_kwargs = dict(kwargs)
            new_kwargs["X"] = new_X
            return args, new_kwargs
