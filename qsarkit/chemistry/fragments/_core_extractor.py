"""Scaffold / core-structure extraction."""

from __future__ import annotations

from qsarkit.base.exceptions import RDKIT_MOLECULE_ERRORS
from typing import Any, List, Optional, Sequence


class CoreExtractor:
    """Extract a representative structural core from one or more molecules.

    Three definitions of "core", answering different questions:

    - :meth:`bemis_murcko` gives ring systems plus linkers (the classic
      scaffold definition), or the fully generic skeleton with element
      and bond-order information stripped.
    - :meth:`mcs` gives the maximum common substructure across a set of
      molecules, which is the right notion for a congeneric series.
    - :meth:`medchem_core` gives a medicinal-chemistry-oriented core: the
      Bemis-Murcko scaffold with terminal exocyclic double bonds trimmed
      back to rings, which tends to match how chemists describe a
      series' "core" more closely than the strict Murcko definition.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.chemistry import CoreExtractor
    >>> extractor = CoreExtractor()
    >>> paracetamol_like = Chem.MolFromSmiles("CC(=O)Nc1ccc(Cl)cc1")
    >>> Chem.MolToSmiles(extractor.bemis_murcko(paracetamol_like))
    'c1ccccc1'

    ``generic=True`` discards element identity and bond order, so
    pyridine and benzene analogues collapse onto one skeleton -- the
    right granularity for asking "how many ring systems are in this
    library", the wrong one for asking "which chemotype is this":

    >>> Chem.MolToSmiles(extractor.bemis_murcko(paracetamol_like, generic=True))
    'C1CCCCC1'

    :meth:`mcs` works across a series rather than on one molecule, and
    returns the shared substructure as a query mol:

    >>> pair = [Chem.MolFromSmiles(s) for s in
    ...         ("CC(=O)Nc1ccc(Cl)cc1", "CC(=O)Nc1ccc(Br)cc1")]
    >>> Chem.MolToSmarts(extractor.mcs(pair))
    '[#6]-[#6](=[#8])-[#7]-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1'

    References
    ----------
    - Bemis, G. W. & Murcko, M. A. (1996). "The Properties of Known
      Drugs. 1. Molecular Frameworks." J. Med. Chem., 39(15), 2887-2893.
      https://doi.org/10.1021/jm9602928
    - Rogers, D. & Hahn, M. (2010) discuss MCS-based series analysis;
      canonical algorithm: Cao, Y. et al. (2008). "A Maximum Common
      Substructure-Based Algorithm for Searching and Predicting Drug-like
      Compounds." Bioinformatics, 24(13), i366-i374.
      https://doi.org/10.1093/bioinformatics/btn186
    - RDKit ``rdFMCS`` and ``Chem.Scaffolds.MurckoScaffold`` documentation:
      https://www.rdkit.org/docs/source/rdkit.Chem.Scaffolds.MurckoScaffold.html
      https://www.rdkit.org/docs/source/rdkit.Chem.rdFMCS.html
    """

    def bemis_murcko(self, mol: Any, generic: bool = False) -> Any:
        from rdkit.Chem.Scaffolds import MurckoScaffold

        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        return scaffold

    def mcs(self, mols: Sequence[Any], **kwargs: Any) -> Optional[Any]:
        from rdkit import Chem
        from rdkit.Chem import rdFMCS

        result = rdFMCS.FindMCS(list(mols), **kwargs)
        if result.canceled or result.numAtoms == 0:
            return None
        return Chem.MolFromSmarts(result.smartsString)

    def medchem_core(self, mol: Any) -> Any:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        rw = Chem.RWMol(scaffold)
        # drop terminal atoms that are not part of any ring (residual
        # linker stubs left after Murcko decomposition, e.g. an exocyclic
        # =O or =N stub with no ring neighbor beyond it)
        to_remove = [
            atom.GetIdx()
            for atom in rw.GetAtoms()
            if atom.GetDegree() == 1 and not atom.IsInRing()
        ]
        for idx in sorted(to_remove, reverse=True):
            rw.RemoveAtom(idx)
        core = rw.GetMol()
        try:
            Chem.SanitizeMol(core)
        except RDKIT_MOLECULE_ERRORS:
            # Trimming can leave an unsanitizable fragment; the untrimmed
            # Murcko scaffold is the correct fallback.
            return scaffold
        return core

    def transform(
        self, mols: List[Any], method: str = "bemis_murcko", **kwargs: Any
    ) -> List[Optional[Any]]:
        """Apply a per-molecule extraction method over an ``Iterable[Mol]``."""
        fn = {"bemis_murcko": self.bemis_murcko, "medchem_core": self.medchem_core}[method]
        return [fn(m, **kwargs) if m is not None else None for m in mols]
