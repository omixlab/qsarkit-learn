"""Removal of protecting groups, linkers, tags and handles."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from qsarkit.base import MoleculeToMoleculeTransformer
from qsarkit.base.exceptions import RDKIT_MOLECULE_ERRORS

# name -> SMARTS matching the group as attached to the rest of the molecule
# via exactly one single bond (the atom marked with a leading '*' bond is
# implicit: matches are trimmed and the attachment point capped with H).
DEFAULT_GROUPS: Dict[str, str] = {
    # protecting groups
    "Boc": "[#6](=O)OC(C)(C)C",
    "Cbz": "[#6](=O)OCc1ccccc1",
    "Fmoc": "[#6](=O)OCC1c2ccccc2-c2ccccc21",
    "Acetyl": "[#6](=O)[CH3]",
    "TBS": "[Si](C)(C)C(C)(C)C",
    "Trityl": "C(c1ccccc1)(c1ccccc1)c1ccccc1",
    "Benzyl_ether": "[OX2]Cc1ccccc1",
    # linkers / tags / handles
    "PEG_linker": "OCCOCCOCC",
    "Biotin_tag": "C1CSC2CC(=O)NC12",
    "Azide_handle": "[N-]=[N+]=[N-]",
    "Alkyne_handle": "C#C",
}


class FragmentRemover(MoleculeToMoleculeTransformer):
    """Strip protecting groups, synthesis linkers, tags and click-handles.

    Matches each pattern in ``groups`` as a substituent attached to the
    core structure via a single bond, deletes the matched atoms, and caps
    the resulting open valence with an implicit hydrogen - conceptually
    the reverse of a protection reaction.

    Parameters
    ----------
    groups : dict[str, str], optional
        Mapping of group name -> SMARTS pattern. Defaults to
        :data:`DEFAULT_GROUPS` (Boc, Cbz, Fmoc, acetyl, TBS, trityl,
        benzyl ether, PEG linkers, biotin tag, azide/alkyne click handles).
    max_iterations : int, default 5
        Repeat removal up to this many times per molecule, since removing
        one group can expose another (e.g. a doubly-Boc-protected amine).

    Examples
    --------
    Boc-protected benzylamine loses its carbamate:

    >>> from rdkit import Chem
    >>> from qsarkit.chemistry import FragmentRemover
    >>> boc = Chem.MolFromSmiles("CC(C)(C)OC(=O)NCc1ccccc1")
    >>> Chem.MolToSmiles(FragmentRemover().transform([boc])[0])
    'NCc1ccccc1'

    This belongs in curation because a protecting group is a synthesis
    artefact, not a pharmacophore: leaving it on makes every intermediate
    in a series look like a distinct chemotype and lets a model key on
    the tag instead of the chemistry.

    Restricting ``groups`` narrows what is stripped -- here Boc is not in
    the set, so the molecule is returned unchanged:

    >>> only_acetyl = FragmentRemover(groups={"acetyl": "[CX3](=O)[CH3]"})
    >>> Chem.MolToSmiles(only_acetyl.transform([boc])[0])
    'CC(C)(C)OC(=O)NCc1ccccc1'

    References
    ----------
    - Wuts, P. G. M. & Greene, T. W. (2014). "Greene's Protective Groups in
      Organic Synthesis," 5th ed. Wiley. https://doi.org/10.1002/9781118978075
    - RDKit reaction/substructure editing documentation:
      https://www.rdkit.org/docs/GettingStartedInPython.html#chemical-reactions
    """

    def __init__(
        self,
        groups: Optional[Dict[str, str]] = None,
        max_iterations: int = 5,
    ):
        # Stored as given, per the scikit-learn convention; the default is
        # resolved in _resolved_groups() so get_params() reports what was
        # actually passed and clone() round-trips.
        self.groups = groups
        self.max_iterations = max_iterations

    def _remove_once(self, mol: Any, patterns: List[Any]) -> bool:
        from rdkit import Chem

        for patt in patterns:
            match = mol.GetSubstructMatch(patt)
            if not match:
                continue
            match_set = set(match)
            rw = Chem.RWMol(mol)
            # find the single bond connecting the matched fragment to the rest
            boundary_bonds = []
            for idx in match_set:
                atom = rw.GetAtomWithIdx(idx)
                for nbr in atom.GetNeighbors():
                    if nbr.GetIdx() not in match_set:
                        boundary_bonds.append((idx, nbr.GetIdx()))
            for a1, a2 in boundary_bonds:
                if rw.GetBondBetweenAtoms(a1, a2) is not None:
                    rw.RemoveBond(a1, a2)
            for idx in sorted(match_set, reverse=True):
                rw.RemoveAtom(idx)
            new_mol = rw.GetMol()
            try:
                Chem.SanitizeMol(new_mol)
            except RDKIT_MOLECULE_ERRORS:
                # This cut left an invalid molecule; try the next match.
                continue
            mol.__init__(new_mol)
            return True
        return False

    def _clean_one(self, mol: Any) -> Any:
        from rdkit import Chem

        groups = self.groups if self.groups is not None else DEFAULT_GROUPS
        patterns = [Chem.MolFromSmarts(s) for s in groups.values()]
        current = Chem.Mol(mol)
        for _ in range(self.max_iterations):
            changed = self._remove_once(current, patterns)
            if not changed:
                break
        return current

    def _transform(self, mols: List[Any]) -> List[Any]:
        return [None if m is None else self._clean_one(m) for m in mols]
