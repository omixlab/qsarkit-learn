"""Shared array-validation helpers for :mod:`qsarkit.metrics`.

These helpers are intentionally private: they exist only so every public
metric performs identical, predictable input coercion (1-D float arrays,
matched lengths, no NaN/inf) and therefore raises identical, informative
errors.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["as_float_1d", "check_pair"]


def as_float_1d(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Coerce ``values`` to a finite, 1-D ``float64`` array.

    Parameters
    ----------
    values : array-like
        Input to coerce.
    name : str
        Name used in error messages.

    Returns
    -------
    numpy.ndarray
        A 1-D ``float64`` array.

    Raises
    ------
    ValueError
        If the input is not 1-D (after squeezing a trailing singleton
        axis), is empty, or contains non-finite values.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def check_pair(
    y_true: ArrayLike, y_pred: ArrayLike
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Coerce and length-check an ``(y_true, y_pred)`` pair.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Observed and predicted values.

    Returns
    -------
    tuple of numpy.ndarray
        The validated ``(y_true, y_pred)`` arrays.

    Raises
    ------
    ValueError
        If the two arrays have different lengths.
    """
    yt = as_float_1d(y_true, "y_true")
    yp = as_float_1d(y_pred, "y_pred")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(
            f"y_true and y_pred must have the same length, "
            f"got {yt.shape[0]} and {yp.shape[0]}."
        )
    return yt, yp
