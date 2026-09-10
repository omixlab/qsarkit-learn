"""Tests for qsarkit.metrics._common array-validation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from qsarkit.metrics._common import as_float_1d, check_pair


def test_as_float_1d_basic() -> None:
    out = as_float_1d([1, 2, 3], "x")
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])


def test_as_float_1d_squeezes_column_vector() -> None:
    out = as_float_1d(np.array([[1.0], [2.0]]), "x")
    assert out.shape == (2,)


def test_as_float_1d_rejects_2d() -> None:
    with pytest.raises(ValueError, match="must be 1-D"):
        as_float_1d(np.zeros((2, 2)), "x")


def test_as_float_1d_rejects_empty() -> None:
    with pytest.raises(ValueError, match="is empty"):
        as_float_1d([], "x")


def test_as_float_1d_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        as_float_1d([1.0, np.nan], "x")
    with pytest.raises(ValueError, match="NaN or infinite"):
        as_float_1d([1.0, np.inf], "x")


def test_check_pair_matches_lengths() -> None:
    yt, yp = check_pair([1.0, 2.0], [3.0, 4.0])
    np.testing.assert_array_equal(yt, [1.0, 2.0])
    np.testing.assert_array_equal(yp, [3.0, 4.0])


def test_check_pair_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        check_pair([1.0, 2.0], [1.0, 2.0, 3.0])
