"""Glycan-content descriptors."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, List, Optional, Set

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

from qsarkit.base import MoleculeTransformer
from qsarkit.chemistry.glycans._detector import GlycanDetector, GlycanMatch


def _linkage_type(mol: Any, match: GlycanMatch, glycan_atoms: Set[int]) -> str:
    """Classify a sugar ring's linkage to the rest of the molecule as O/C/none."""
    for idx in match.ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in glycan_atoms:
                continue
            return "O-glycoside" if atom.GetSymbol() == "O" or nbr.GetSymbol() == "O" else "C-glycoside"
    return "terminal"


class GlycanDescriptors(MoleculeTransformer):
    """Compute glycan-content descriptors for a batch of molecules.

    The three columns produced are:

    ``sugar_count``
        Number of detected sugar rings (see :class:`GlycanDetector`).
    ``glycan_fraction``
        Fraction of molecular weight contributed by glycan atoms.
    ``glycosylation_pattern``
        Comma-separated summary such as ``"6-ring:O-glycoside x2"``
        describing ring size and linkage type of each detected sugar.

    Parameters
    ----------
    detector : GlycanDetector, optional

    Notes
    -----
    Unlike the transformers in :mod:`qsarkit.representation`, this one
    returns a :class:`pandas.DataFrame` rather than a NumPy array,
    because ``glycosylation_pattern`` is a string. Forcing it into an
    array would give the whole block ``object`` dtype and lose the
    numeric columns' types.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.chemistry import GlycanDescriptors
    >>> q3g = Chem.MolFromSmiles(
    ...     "OC[C@H]1O[C@@H](Oc2c(-c3ccc(O)c(O)c3)oc3cc(O)cc(O)c3c2=O)"
    ...     "[C@H](O)[C@@H](O)[C@@H]1O")
    >>> df = GlycanDescriptors().transform([q3g, Chem.MolFromSmiles("CCO")])
    >>> list(df.columns)
    ['sugar_count', 'glycan_fraction', 'glycosylation_pattern']
    >>> df["sugar_count"].tolist()
    [1, 0]
    >>> df["glycosylation_pattern"][0]
    '6-ring:terminal'

    References
    ----------
    - Fischer et al. (2020). "The Sugar Removal Utility (SRU)." Molecules,
      25(8), 1988. https://doi.org/10.3390/molecules25081988
    """

    def __init__(self, detector: Optional[GlycanDetector] = None):
        self.detector = detector or GlycanDetector()

    def _describe_one(self, mol: Any) -> dict:
        matches = self.detector.find_glycans(mol)
        glycan_atoms = set()
        for m in matches:
            glycan_atoms |= m.all_atoms

        det = self.detector.detect(mol)
        patterns = Counter(
            f"{m.ring_size}-ring:{_linkage_type(mol, m, glycan_atoms)}" for m in matches
        )
        pattern_str = ", ".join(
            f"{p} x{n}" if n > 1 else p for p, n in sorted(patterns.items())
        )
        return {
            "sugar_count": det["num_sugar_residues"],
            "glycan_fraction": det["glycan_mw_fraction"],
            "glycosylation_pattern": pattern_str,
        }

    def _transform(self, mols: List[Any]) -> "pd.DataFrame":
        import pandas as pd

        rows = [self._describe_one(m) if m is not None else None for m in mols]
        return pd.DataFrame(
            rows, columns=["sugar_count", "glycan_fraction", "glycosylation_pattern"]
        )
