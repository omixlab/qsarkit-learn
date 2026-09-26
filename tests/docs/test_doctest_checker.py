"""The tolerance used when checking documentation examples.

``_FloatTolerantChecker`` relaxes every numeric assertion in the
documentation, so its limits are worth pinning down: too tight and the docs
fail on a platform other than the author's, too loose and a real regression
slips through looking like rounding. The cases below are the ones that
actually came up.
"""

from __future__ import annotations

import doctest

import pytest

# A sibling module, imported without a package prefix: tests/ has no
# __init__.py by design, so pytest puts this directory on sys.path and
# `tests.docs.…` would only resolve when pytest happens to run from the
# repository root.
from test_documentation_examples import OPTIONFLAGS, _FloatTolerantChecker


@pytest.fixture
def check():
    checker = _FloatTolerantChecker()

    def _check(want: str, got: str) -> bool:
        return checker.check_output(want, got, OPTIONFLAGS)

    return _check


class TestAcceptsPlatformDrift:
    """Differences of this size are BLAS and library versions, not code."""

    @pytest.mark.parametrize(
        "want, got",
        [
            ("0.235\n", "0.237\n"),                    # rmse, sklearn 1.8 -> 1.9
            ("(0.686, 0.602)\n", "(0.685, 0.603)\n"),  # CV scores, arm64 -> x86_64
            ("2.94\n", "2.89\n"),                      # ENCE
            ("1.11\n", "1.12\n"),                      # rmse_ratio
            ("4.2\n", "4.21\n"),                       # interval width
            ("-1.0\n", "-0.99\n"),                     # q2_f1 near -1
            ("array([17.3,  0.6, 12.8])\n", "array([17.4,  0.6, 12.8])\n"),
        ],
    )
    def test_small_numeric_drift_passes(self, check, want, got):
        assert check(want, got)


class TestRejectsRealChanges:
    """Anything that is not last-digit noise must still fail."""

    @pytest.mark.parametrize(
        "want, got, why",
        [
            ("6.1\n", "5.28\n", "a different neighbour was chosen, not rounding"),
            ("(1117, 2048)\n", "(1118, 2048)\n", "a count changed"),
            ("[[0, 3, 2]]\n", "[[0, 3, 1]]\n", "an index changed"),
            ("(24,)\n", "(23,)\n", "a shape changed"),
            ("0.0\n", "0.002\n", "a value moved away from zero"),
            ("0.95\n", "0.88\n", "a score moved by 7%"),
            ("(24, 2048)\n", "(24, 2048, 1)\n", "the structure changed"),
            ("['a', 'b']\n", "['a', 'c']\n", "a key changed"),
            ("True\n", "False\n", "a boolean flipped"),
        ],
    )
    def test_real_differences_fail(self, check, want, got, why):
        assert not check(want, got), why


class TestStillExact:
    """Non-numeric output is compared exactly, as doctest normally does."""

    def test_identical_output_passes(self, check):
        assert check("'Figure'\n", "'Figure'\n")

    def test_integers_are_never_tolerated(self, check):
        """A 1% difference in a count is a change; in a decimal it is noise."""
        assert not check("100\n", "101\n")
        assert check("100.0\n", "100.5\n")

    def test_a_missing_number_fails(self, check):
        assert not check("0.5 0.6\n", "0.5\n")
