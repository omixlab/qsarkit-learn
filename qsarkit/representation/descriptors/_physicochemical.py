"""Core medicinal-chemistry physicochemical descriptors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Tuple

from qsarkit.representation.descriptors._base import BaseDescriptorTransformer

if TYPE_CHECKING:  # pragma: no cover
    from rdkit.Chem import Mol

__all__ = ["PhysicochemicalDescriptors"]


def _mol_wt(mol: "Mol") -> float:
    from rdkit.Chem import Descriptors

    return float(Descriptors.MolWt(mol))


def _mol_logp(mol: "Mol") -> float:
    from rdkit.Chem import Descriptors

    return float(Descriptors.MolLogP(mol))


def _tpsa(mol: "Mol") -> float:
    from rdkit.Chem import Descriptors

    return float(Descriptors.TPSA(mol))


def _num_hbd(mol: "Mol") -> float:
    from rdkit.Chem import Descriptors

    return float(Descriptors.NumHDonors(mol))


def _num_hba(mol: "Mol") -> float:
    from rdkit.Chem import Descriptors

    return float(Descriptors.NumHAcceptors(mol))


def _num_rotatable_bonds(mol: "Mol") -> float:
    from rdkit.Chem import Descriptors

    return float(Descriptors.NumRotatableBonds(mol))


def _fraction_csp3(mol: "Mol") -> float:
    from rdkit.Chem import rdMolDescriptors

    return float(rdMolDescriptors.CalcFractionCSP3(mol))


def _qed(mol: "Mol") -> float:
    from rdkit.Chem import QED

    return float(QED.qed(mol))


def _molar_refractivity(mol: "Mol") -> float:
    from rdkit.Chem import Crippen

    return float(Crippen.MolMR(mol))


_DESCRIPTORS: Tuple[Tuple[str, Callable[["Mol"], float]], ...] = (
    ("MolWt", _mol_wt),
    ("MolLogP", _mol_logp),
    ("TPSA", _tpsa),
    ("NumHDonors", _num_hbd),
    ("NumHAcceptors", _num_hba),
    ("NumRotatableBonds", _num_rotatable_bonds),
    ("FractionCSP3", _fraction_csp3),
    ("MolMR", _molar_refractivity),
    ("QED", _qed),
)


class PhysicochemicalDescriptors(BaseDescriptorTransformer):
    """Core medicinal-chemistry physicochemical property block.

    Bundles the handful of whole-molecule properties most commonly used to
    reason about drug-likeness and ADMET behaviour: molecular weight,
    octanol-water partition coefficient (Wildman-Crippen ``MolLogP``),
    topological polar surface area, hydrogen-bond donor/acceptor counts,
    rotatable-bond count, the fraction of sp3-hybridized carbons, molar
    refractivity, and the QED drug-likeness score.

    Parameters
    ----------
    missing_value : float, default nan
        Value substituted when a descriptor raises or returns a
        non-finite value for a given molecule.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.representation.descriptors import PhysicochemicalDescriptors
    >>> pc = PhysicochemicalDescriptors()
    >>> X = pc.fit_transform([Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")])
    >>> X.shape
    (1, 9)

    References
    ----------
    - Wildman, S. A. & Crippen, G. M. (1999). "Prediction of Physicochemical
      Parameters by Atomic Contributions." J. Chem. Inf. Comput. Sci.,
      39(5), 868-873. https://doi.org/10.1021/ci990307l
    - Ertl, P., Rohde, B. & Selzer, P. (2000). "Fast Calculation of
      Molecular Polar Surface Area as a Sum of Fragment-Based
      Contributions and Its Application to the Prediction of Drug
      Transport Properties." J. Med. Chem., 43(20), 3714-3717.
      https://doi.org/10.1021/jm000942e
    - Bickerton, G. R. et al. (2012). "Quantifying the Chemical Beauty of
      Drugs." Nat. Chem., 4(2), 90-98. https://doi.org/10.1038/nchem.1243
    - RDKit ``rdkit.Chem.QED`` documentation:
      https://www.rdkit.org/docs/source/rdkit.Chem.QED.html
    """

    def __init__(self, missing_value: float = float("nan")) -> None:
        self.missing_value = missing_value

    def _descriptor_functions(self) -> List[Tuple[str, Callable[["Mol"], float]]]:
        return list(_DESCRIPTORS)
