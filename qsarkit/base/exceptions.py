"""Shared exception hierarchy for qsarkit.

Keeping a single hierarchy lets callers catch ``QsarkitError`` at a package
boundary (e.g. a curation pipeline that must not crash on one bad record)
while still being able to distinguish failure modes when needed.
"""

from __future__ import annotations


class QsarkitError(Exception):
    """Base class for all qsarkit exceptions."""


class InvalidMoleculeError(QsarkitError):
    """Raised when an input cannot be parsed or sanitized into an RDKit Mol."""


class ModelNotFittedError(QsarkitError):
    """Raised when ``.transform``/``.predict`` is called before ``.fit``."""


class OptionalDependencyError(QsarkitError):
    """Raised when an optional dependency required by a feature is missing."""

    def __init__(self, package: str, extra: str | None = None):
        msg = f"This feature requires the optional dependency '{package}'."
        if extra:
            msg += f" Install it with: pip install qsarkit[{extra}]"
        else:
            msg += f" Install it with: pip install {package}"
        super().__init__(msg)
        self.package = package
        self.extra = extra
