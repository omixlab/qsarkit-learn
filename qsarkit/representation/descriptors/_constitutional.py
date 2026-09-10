"""Constitutional descriptors: atom/bond/ring counts (Todeschini & Consonni)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Tuple

from qsarkit.representation.descriptors._base import BaseDescriptorTransformer

if TYPE_CHECKING:  # pragma: no cover
    from rdkit.Chem import Mol

__all__ = ["ConstitutionalDescriptors"]


def _mol_wt(mol: "Mol") -> float:
    from rdkit.Chem import Descriptors

    return float(Descriptors.MolWt(mol))


def _average_mol_wt(mol: "Mol") -> float:
    from rdkit.Chem import Descriptors

    n_heavy = max(mol.GetNumAtoms(), 1)
    return float(Descriptors.MolWt(mol) / n_heavy)


def _num_atoms(mol: "Mol") -> float:
    from rdkit import Chem

    return float(Chem.AddHs(mol).GetNumAtoms())


def _num_heavy_atoms(mol: "Mol") -> float:
    return float(mol.GetNumAtoms())


def _num_heteroatoms(mol: "Mol") -> float:
    from rdkit.Chem import rdMolDescriptors

    return float(rdMolDescriptors.CalcNumHeteroatoms(mol))


def _num_bonds(mol: "Mol") -> float:
    return float(mol.GetNumBonds())


def _num_rings(mol: "Mol") -> float:
    from rdkit.Chem import rdMolDescriptors

    return float(rdMolDescriptors.CalcNumRings(mol))


def _num_aromatic_rings(mol: "Mol") -> float:
    from rdkit.Chem import rdMolDescriptors

    return float(rdMolDescriptors.CalcNumAromaticRings(mol))


def _num_saturated_rings(mol: "Mol") -> float:
    from rdkit.Chem import rdMolDescriptors

    return float(rdMolDescriptors.CalcNumSaturatedRings(mol))


def _num_aliphatic_rings(mol: "Mol") -> float:
    from rdkit.Chem import rdMolDescriptors

    return float(rdMolDescriptors.CalcNumAliphaticRings(mol))


def _num_aromatic_carbocycles(mol: "Mol") -> float:
    from rdkit.Chem import rdMolDescriptors

    return float(rdMolDescriptors.CalcNumAromaticCarbocycles(mol))


def _num_aromatic_heterocycles(mol: "Mol") -> float:
    from rdkit.Chem import rdMolDescriptors

    return float(rdMolDescriptors.CalcNumAromaticHeterocycles(mol))


def _num_saturated_carbocycles(mol: "Mol") -> float:
    from rdkit.Chem import rdMolDescriptors

    return float(rdMolDescriptors.CalcNumSaturatedCarbocycles(mol))


def _num_saturated_heterocycles(mol: "Mol") -> float:
    from rdkit.Chem import rdMolDescriptors

    return float(rdMolDescriptors.CalcNumSaturatedHeterocycles(mol))


def _num_aliphatic_carbocycles(mol: "Mol") -> float:
    from rdkit.Chem import rdMolDescriptors

    return float(rdMolDescriptors.CalcNumAliphaticCarbocycles(mol))


def _num_aliphatic_heterocycles(mol: "Mol") -> float:
    from rdkit.Chem import rdMolDescriptors

    return float(rdMolDescriptors.CalcNumAliphaticHeterocycles(mol))


def _num_spiro_atoms(mol: "Mol") -> float:
    from rdkit.Chem import rdMolDescriptors

    return float(rdMolDescriptors.CalcNumSpiroAtoms(mol))


def _num_bridgehead_atoms(mol: "Mol") -> float:
    from rdkit.Chem import rdMolDescriptors

    return float(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))


def _fraction_csp3(mol: "Mol") -> float:
    from rdkit.Chem import rdMolDescriptors

    return float(rdMolDescriptors.CalcFractionCSP3(mol))


_DESCRIPTORS: Tuple[Tuple[str, Callable[["Mol"], float]], ...] = (
    ("MolWt", _mol_wt),
    ("AMW", _average_mol_wt),
    ("NumAtoms", _num_atoms),
    ("NumHeavyAtoms", _num_heavy_atoms),
    ("NumHeteroatoms", _num_heteroatoms),
    ("NumBonds", _num_bonds),
    ("NumRings", _num_rings),
    ("NumAromaticRings", _num_aromatic_rings),
    ("NumSaturatedRings", _num_saturated_rings),
    ("NumAliphaticRings", _num_aliphatic_rings),
    ("NumAromaticCarbocycles", _num_aromatic_carbocycles),
    ("NumAromaticHeterocycles", _num_aromatic_heterocycles),
    ("NumSaturatedCarbocycles", _num_saturated_carbocycles),
    ("NumSaturatedHeterocycles", _num_saturated_heterocycles),
    ("NumAliphaticCarbocycles", _num_aliphatic_carbocycles),
    ("NumAliphaticHeterocycles", _num_aliphatic_heterocycles),
    ("NumSpiroAtoms", _num_spiro_atoms),
    ("NumBridgeheadAtoms", _num_bridgehead_atoms),
    ("FractionCSP3", _fraction_csp3),
)


class ConstitutionalDescriptors(BaseDescriptorTransformer):
    """Constitutional descriptors: simple atom/bond/ring counts.

    Constitutional descriptors are the simplest, 0-dimensional family in
    the Todeschini & Consonni taxonomy: they depend only on molecular
    composition and connectivity (how many of each atom/bond/ring type),
    not on any graph-theoretic weighting or 3-D geometry. They are cheap,
    always defined, and form the backbone of most QSAR descriptor sets.

    Parameters
    ----------
    missing_value : float, default nan
        Value substituted when a descriptor raises or returns a
        non-finite value for a given molecule.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.representation.descriptors import ConstitutionalDescriptors
    >>> cd = ConstitutionalDescriptors()
    >>> X = cd.fit_transform([Chem.MolFromSmiles("c1ccccc1")])
    >>> X.shape[1] == len(cd.get_feature_names_out())
    True

    References
    ----------
    - Todeschini, R. & Consonni, V. (2009). "Molecular Descriptors for
      Chemoinformatics." Wiley-VCH. https://doi.org/10.1002/9783527628766
    - RDKit ``rdkit.Chem.rdMolDescriptors`` documentation:
      https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html
    """

    def __init__(self, missing_value: float = float("nan")) -> None:
        self.missing_value = missing_value

    def _descriptor_functions(self) -> List[Tuple[str, Callable[["Mol"], float]]]:
        return list(_DESCRIPTORS)
