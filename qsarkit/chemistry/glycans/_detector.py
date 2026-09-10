"""Carbohydrate (glycan) moiety detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Set

# Generic pyranose / furanose ring skeletons: one ring oxygen, the rest
# sp3 ring carbons, any substitution pattern. These match the core ring of
# aldo/keto hexopyranoses, pentopyranoses and furanoses regardless of
# stereochemistry or substituent identity.
_PYRANOSE_RING = "[OX2;R1;r6]1[#6;R1;r6][#6;R1;r6][#6;R1;r6][#6;R1;r6][#6;R1;r6]1"
_FURANOSE_RING = "[OX2;R1;r5]1[#6;R1;r5][#6;R1;r5][#6;R1;r5][#6;R1;r5]1"

# A ring is only flagged as a sugar (as opposed to e.g. cyclohexane or THF)
# if at least this many ring carbons carry an exocyclic oxygen substituent
# (hydroxyl, glycosidic ether, or exocyclic CH2OH), matching the
# hydroxyl-dense profile of aldoses/ketoses.
_MIN_EXOCYCLIC_OXYGENS = 2


@dataclass
class GlycanMatch:
    """One detected sugar ring plus its directly attached exocyclic atoms."""

    ring_atoms: Set[int]
    substituent_atoms: Set[int]
    ring_size: int

    @property
    def all_atoms(self) -> Set[int]:
        return self.ring_atoms | self.substituent_atoms


class GlycanDetector:
    """Detect carbohydrate (sugar) ring systems in a molecule.

    Uses the circular-sugar detection strategy of the Sugar Removal
    Utility (SRU): a candidate ring is a 5- or 6-membered ring with
    exactly one ring oxygen and only sp3 carbons otherwise, and is
    accepted as a sugar only if it is decorated with enough exocyclic
    oxygens (hydroxyls / glycosidic ethers / an exocyclic CH2OH) to match
    the hydroxylation pattern of a real aldose/ketose ring - which
    excludes plain carbocycles (cyclohexane) and simple ethers
    (tetrahydropyran, tetrahydrofuran).

    Parameters
    ----------
    min_exocyclic_oxygens : int, default 2
        Exocyclic oxygens a candidate ring must carry to be called a
        sugar. Lowering it admits deoxy sugars along with false
        positives; raising it rejects rhamnose and other deoxy sugars.

    Examples
    --------
    Quercetin 3-O-glucoside carries one pyranose ring, contributing 39%
    of the molecular weight:

    >>> from rdkit import Chem
    >>> from qsarkit.chemistry import GlycanDetector
    >>> q3g = Chem.MolFromSmiles(
    ...     "OC[C@H]1O[C@@H](Oc2c(-c3ccc(O)c(O)c3)oc3cc(O)cc(O)c3c2=O)"
    ...     "[C@H](O)[C@@H](O)[C@@H]1O")
    >>> result = GlycanDetector().detect(q3g)
    >>> result["num_sugar_residues"]
    1
    >>> round(result["glycan_mw_fraction"], 3)
    0.386

    The exocyclic-oxygen requirement is what separates a sugar from a
    look-alike ring. Tetrahydropyran has the right ring but no
    hydroxylation, and is correctly rejected:

    >>> GlycanDetector().detect(Chem.MolFromSmiles("C1CCOCC1"))["num_sugar_residues"]
    0
    >>> GlycanDetector().detect(Chem.MolFromSmiles("CCO"))["num_sugar_residues"]
    0

    References
    ----------
    - Fischer et al. (2020). "The Sugar Removal Utility (SRU): An
      Open-Source Peptide- and Sugar-Stripping Tool for Chemical
      Structure Databases." Molecules, 25(8), 1988.
      https://doi.org/10.3390/molecules25081988
    - RDKit ring perception documentation:
      https://www.rdkit.org/docs/RDKit_Book.html#ring-perception
    """

    def __init__(self, min_exocyclic_oxygens: int = _MIN_EXOCYCLIC_OXYGENS):
        self.min_exocyclic_oxygens = min_exocyclic_oxygens

    def _candidate_rings(self, mol: Any) -> List[Any]:
        from rdkit import Chem

        patterns = [Chem.MolFromSmarts(_PYRANOSE_RING), Chem.MolFromSmarts(_FURANOSE_RING)]
        matches = []
        for patt in patterns:
            for match in mol.GetSubstructMatches(patt, uniquify=True):
                matches.append(set(match))
        # de-duplicate identical atom sets found by both patterns
        unique = []
        for m in matches:
            if m not in unique:
                unique.append(m)
        return unique

    def _is_sugar_ring(self, mol: Any, ring_atoms: Set[int]) -> bool:
        exocyclic_o = 0
        for idx in ring_atoms:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() != "C":
                continue
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() in ring_atoms:
                    continue
                if nbr.GetSymbol() == "O":
                    exocyclic_o += 1
                elif nbr.GetSymbol() == "C" and any(
                    n2.GetSymbol() == "O" for n2 in nbr.GetNeighbors()
                ):
                    # exocyclic CH2OH (the C6 of a hexopyranose)
                    exocyclic_o += 1
        return exocyclic_o >= self.min_exocyclic_oxygens

    def _substituents_of(self, mol: Any, ring_atoms: Set[int]) -> Set[int]:
        subs = set()
        for idx in ring_atoms:
            atom = mol.GetAtomWithIdx(idx)
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() in ring_atoms:
                    continue
                if nbr.GetSymbol() == "O" and nbr.GetDegree() <= 2:
                    subs.add(nbr.GetIdx())
                elif nbr.GetSymbol() == "C" and any(
                    n2.GetSymbol() == "O" and n2.GetDegree() <= 2 for n2 in nbr.GetNeighbors()
                ):
                    subs.add(nbr.GetIdx())
                    for n2 in nbr.GetNeighbors():
                        if n2.GetSymbol() == "O" and n2.GetDegree() <= 2:
                            subs.add(n2.GetIdx())
        return subs

    def find_glycans(self, mol: Any) -> List[GlycanMatch]:
        """Return one :class:`GlycanMatch` per detected sugar ring."""
        matches = []
        for ring_atoms in self._candidate_rings(mol):
            if self._is_sugar_ring(mol, ring_atoms):
                subs = self._substituents_of(mol, ring_atoms)
                matches.append(
                    GlycanMatch(
                        ring_atoms=ring_atoms,
                        substituent_atoms=subs - ring_atoms,
                        ring_size=len(ring_atoms),
                    )
                )
        return matches

    def detect(self, mol: Any) -> dict:
        """Summarize glycan content of a molecule.

        Returns
        -------
        dict
            ``num_sugar_residues``: number of detected sugar rings.
            ``glycan_atoms``: sorted list of atom indices belonging to
            glycan rings/substituents.
            ``glycan_mw_fraction``: fraction of the molecule's molecular
            weight contributed by glycan atoms (including attached H's).
        """
        from rdkit.Chem import Descriptors

        matches = self.find_glycans(mol)
        glycan_atoms: Set[int] = set()
        for m in matches:
            glycan_atoms |= m.all_atoms

        total_mw = Descriptors.MolWt(mol)
        glycan_mw = 0.0
        if glycan_atoms:
            from rdkit import Chem

            for idx in glycan_atoms:
                atom = mol.GetAtomWithIdx(idx)
                glycan_mw += atom.GetMass() + atom.GetTotalNumHs() * 1.008

        return {
            "num_sugar_residues": len(matches),
            "glycan_atoms": sorted(glycan_atoms),
            "glycan_mw_fraction": (glycan_mw / total_mw) if total_mw > 0 else 0.0,
        }

    def transform(self, mols: Any) -> List[Optional[dict]]:
        """Run :meth:`detect` over an ``Iterable[Mol]``."""
        return [self.detect(m) if m is not None else None for m in mols]
