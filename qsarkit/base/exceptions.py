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
            msg += f" Install it with: pip install qsarkit-learn[{extra}]"
        else:
            msg += f" Install it with: pip install {package}"
        super().__init__(msg)
        self.package = package
        self.extra = extra


#: Exceptions RDKit raises when it cannot process one particular molecule.
#:
#: Catch this rather than bare ``Exception`` in a per-molecule loop. The
#: three members cover every way RDKit signals "this structure is not
#: usable":
#:
#: * ``ValueError`` -- the base of RDKit's whole sanitization hierarchy
#:   (``MolSanitizeException``, ``AtomValenceException``,
#:   ``KekulizeException``), and of InChI generation failures.
#: * ``RuntimeError`` -- lower-level C++ errors surfaced by the wrapper,
#:   such as a conformer or substructure-match failure.
#: * ``TypeError`` -- a non-``Mol`` argument reaching a wrapped function.
#:   ``Boost.Python.ArgumentError`` subclasses this.
#:
#: What it deliberately does *not* catch is ``MemoryError`` (which is an
#: ``Exception`` subclass, so a bare ``except Exception`` swallows it) or
#: ``KeyboardInterrupt``. Those are real failures of the run, not of one
#: molecule, and must propagate.
#:
#: Examples
#: --------
#: >>> from rdkit import Chem
#: >>> from qsarkit.base import RDKIT_MOLECULE_ERRORS
#: >>> def safe_inchikey(mol):
#: ...     try:
#: ...         return Chem.MolToInchiKey(mol)
#: ...     except RDKIT_MOLECULE_ERRORS:
#: ...         return None
#: >>> safe_inchikey("not a molecule") is None
#: True
RDKIT_MOLECULE_ERRORS = (ValueError, RuntimeError, TypeError)
