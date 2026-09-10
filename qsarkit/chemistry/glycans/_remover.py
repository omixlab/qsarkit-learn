"""Carbohydrate (glycan) removal, preserving the aglycone."""

from __future__ import annotations

from typing import Any, List, Optional

from qsarkit.chemistry.glycans._detector import GlycanDetector


class GlycanRemover:
    """Remove carbohydrate (sugar) moieties, keeping the aglycone core.

    Detects sugar rings with :class:`GlycanDetector`, cleaves the
    glycosidic (single) bonds linking sugar atoms to the rest of the
    molecule, and returns the largest non-sugar fragment as the
    "aglycone" - e.g. quercetin-3-glucoside -> quercetin.

    Parameters
    ----------
    detector : GlycanDetector, optional
        Detector used to locate sugar rings. A default instance is
        created if not supplied.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.chemistry import GlycanRemover
    >>> q3g = Chem.MolFromSmiles(
    ...     "OC[C@H]1O[C@@H](Oc2c(-c3ccc(O)c(O)c3)oc3cc(O)cc(O)c3c2=O)"
    ...     "[C@H](O)[C@@H](O)[C@@H]1O")
    >>> result = GlycanRemover().remove(q3g)
    >>> Chem.MolToSmiles(result["aglycone"])
    'O=c1cc(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12'

    The result keeps the input alongside what was cut away, so a curation
    step can record why a structure changed:

    >>> sorted(result)
    ['aglycone', 'original', 'removed_fragments']

    Deglycosylation matters for QSAR because the sugar usually carries no
    activity of its own but dominates the fingerprint, so two glycosides
    of unrelated aglycones look more similar to each other than either
    does to its own aglycone. A molecule with no sugar passes through
    unchanged:

    >>> aspirin = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    >>> Chem.MolToSmiles(GlycanRemover().remove(aspirin)["aglycone"])
    'CC(=O)Oc1ccccc1C(=O)O'

    References
    ----------
    - Fischer et al. (2020). "The Sugar Removal Utility (SRU)." Molecules,
      25(8), 1988. https://doi.org/10.3390/molecules25081988
    - RDKit fragmentation documentation (``Chem.GetMolFrags``,
      ``Chem.RWMol``): https://www.rdkit.org/docs/GettingStartedInPython.html
    """

    def __init__(self, detector: Optional[GlycanDetector] = None):
        self.detector = detector or GlycanDetector()

    def remove(self, mol: Any) -> dict:
        """Remove glycan moieties from a single molecule.

        Returns
        -------
        dict
            ``original``: the input Mol.
            ``aglycone``: the largest non-sugar fragment, or ``None`` if
            the whole molecule was consumed by sugar rings.
            ``removed_fragments``: list of removed sugar (and any other
            minor) fragments as Mol objects.
        """
        from rdkit import Chem

        matches = self.detector.find_glycans(mol)
        if not matches:
            return {"original": mol, "aglycone": mol, "removed_fragments": []}

        glycan_atoms = set()
        for m in matches:
            glycan_atoms |= m.all_atoms

        rw = Chem.RWMol(mol)
        bonds_to_break = []
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            in1, in2 = a1 in glycan_atoms, a2 in glycan_atoms
            if in1 != in2 and bond.GetBondType() == Chem.BondType.SINGLE:
                bonds_to_break.append((a1, a2))

        for a1, a2 in bonds_to_break:
            rw.RemoveBond(a1, a2)

        frag_mol = rw.GetMol()
        for atom in frag_mol.GetAtoms():
            atom.SetNoImplicit(False)
        try:
            Chem.SanitizeMol(frag_mol)
        except Exception:
            pass

        frag_atom_indices = Chem.GetMolFrags(frag_mol, asMols=False, sanitizeFrags=False)
        frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False)

        aglycone = None
        aglycone_size = -1
        removed: List[Any] = []
        non_glycan_frags = []
        for frag, atom_idx in zip(frags, frag_atom_indices):
            try:
                Chem.SanitizeMol(frag)
            except Exception:
                pass
            if all(idx in glycan_atoms for idx in atom_idx):
                removed.append(frag)
            else:
                non_glycan_frags.append(frag)

        for frag in non_glycan_frags:
            if frag.GetNumAtoms() > aglycone_size:
                if aglycone is not None:
                    removed.append(aglycone)
                aglycone = frag
                aglycone_size = frag.GetNumAtoms()
            else:
                removed.append(frag)

        return {"original": mol, "aglycone": aglycone, "removed_fragments": removed}

    def transform(self, mols: Any) -> List[Optional[dict]]:
        """Run :meth:`remove` over an ``Iterable[Mol]``."""
        return [self.remove(m) if m is not None else None for m in mols]
