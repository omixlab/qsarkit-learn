"""Lipinski Rule-of-Five and Veber oral-bioavailability descriptors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Tuple

from qsarkit.representation.descriptors._base import BaseDescriptorTransformer

if TYPE_CHECKING:  # pragma: no cover
    from rdkit.Chem import Mol

__all__ = ["LipinskiDescriptors"]


def _mol_wt(mol: "Mol") -> float:
    from rdkit.Chem import Descriptors

    return float(Descriptors.MolWt(mol))


def _mol_logp(mol: "Mol") -> float:
    from rdkit.Chem import Descriptors

    return float(Descriptors.MolLogP(mol))


def _num_hbd(mol: "Mol") -> float:
    from rdkit.Chem import Descriptors

    return float(Descriptors.NumHDonors(mol))


def _num_hba(mol: "Mol") -> float:
    from rdkit.Chem import Descriptors

    return float(Descriptors.NumHAcceptors(mol))


def _num_rotatable_bonds(mol: "Mol") -> float:
    from rdkit.Chem import Descriptors

    return float(Descriptors.NumRotatableBonds(mol))


def _tpsa(mol: "Mol") -> float:
    from rdkit.Chem import Descriptors

    return float(Descriptors.TPSA(mol))


def _lipinski_violations(mol: "Mol") -> float:
    from rdkit.Chem import Descriptors

    violations = 0
    if Descriptors.MolWt(mol) > 500:
        violations += 1
    if Descriptors.MolLogP(mol) > 5:
        violations += 1
    if Descriptors.NumHDonors(mol) > 5:
        violations += 1
    if Descriptors.NumHAcceptors(mol) > 10:
        violations += 1
    return float(violations)


def _passes_lipinski(mol: "Mol") -> float:
    # The conventional threshold: at most one Rule-of-Five violation.
    return 1.0 if _lipinski_violations(mol) <= 1 else 0.0


def _passes_veber(mol: "Mol") -> float:
    from rdkit.Chem import Descriptors

    ok = Descriptors.NumRotatableBonds(mol) <= 10 and Descriptors.TPSA(mol) <= 140
    return 1.0 if ok else 0.0


_DESCRIPTORS: Tuple[Tuple[str, Callable[["Mol"], float]], ...] = (
    ("MolWt", _mol_wt),
    ("MolLogP", _mol_logp),
    ("NumHDonors", _num_hbd),
    ("NumHAcceptors", _num_hba),
    ("NumRotatableBonds", _num_rotatable_bonds),
    ("TPSA", _tpsa),
    ("LipinskiViolations", _lipinski_violations),
    ("PassesLipinski", _passes_lipinski),
    ("PassesVeber", _passes_veber),
)


class LipinskiDescriptors(BaseDescriptorTransformer):
    """Lipinski Rule-of-Five and Veber oral-bioavailability descriptors.

    Computes the four Rule-of-Five properties (molecular weight, LogP,
    hydrogen-bond donor/acceptor counts), the count of Ro5 violations, a
    ``PassesLipinski`` flag (violations <= 1, the conventional tolerance),
    and Veber's two additional oral-bioavailability criteria (rotatable
    bonds <= 10 and TPSA <= 140 A^2) as a ``PassesVeber`` flag.

    Parameters
    ----------
    missing_value : float, default nan
        Value substituted when a descriptor raises or returns a
        non-finite value for a given molecule.

    Notes
    -----
    Boolean outcomes are encoded as ``1.0``/``0.0`` rather than
    ``bool`` so the block stays a uniform ``float64`` matrix, consistent
    with every other transformer in :mod:`qsarkit.representation`.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.representation.descriptors import LipinskiDescriptors
    >>> ld = LipinskiDescriptors()
    >>> X = ld.fit_transform([Chem.MolFromSmiles("CCO")])
    >>> bool(X[0, list(ld.get_feature_names_out()).index("PassesLipinski")])
    True

    References
    ----------
    - Lipinski, C. A., Lombardo, F., Dominy, B. W. & Feeney, P. J. (2001).
      "Experimental and Computational Approaches to Estimate Solubility and
      Permeability in Drug Discovery and Development Settings." Adv. Drug
      Deliv. Rev., 46(1-3), 3-25.
      https://doi.org/10.1016/S0169-409X(96)00423-1
    - Veber, D. F. et al. (2002). "Molecular Properties That Influence the
      Oral Bioavailability of Drug Candidates." J. Med. Chem., 45(12),
      2615-2623. https://doi.org/10.1021/jm020017n
    - RDKit ``rdkit.Chem.Lipinski`` documentation:
      https://www.rdkit.org/docs/source/rdkit.Chem.Lipinski.html
    """

    def __init__(self, missing_value: float = float("nan")) -> None:
        self.missing_value = missing_value

    def _descriptor_functions(self) -> List[Tuple[str, Callable[["Mol"], float]]]:
        return list(_DESCRIPTORS)
