"""Shared base classes, exceptions and helpers used across all of qsarkit."""

from qsarkit.base.exceptions import (
    RDKIT_MOLECULE_ERRORS,
    InvalidMoleculeError,
    ModelNotFittedError,
    OptionalDependencyError,
    QsarkitError,
)
from qsarkit.base.optional_deps import require
from qsarkit.base.transformer import (
    FittableMoleculeTransformer,
    MoleculeToMoleculeTransformer,
    MoleculeTransformer,
    ensure_mol_list,
)

__all__ = [
    "QsarkitError",
    "RDKIT_MOLECULE_ERRORS",
    "InvalidMoleculeError",
    "ModelNotFittedError",
    "OptionalDependencyError",
    "require",
    "MoleculeTransformer",
    "MoleculeToMoleculeTransformer",
    "FittableMoleculeTransformer",
    "ensure_mol_list",
]
