"""Matched molecular pair (MMP) identification via the fragment-index algorithm."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from rdkit.Chem import Mol

__all__ = ["MatchedPair", "MatchedMolecularPairs", "MMPAnalyzer"]

# Any acyclic single bond between two heavy atoms. Terminal atoms are
# deliberately NOT excluded: the single most useful class of matched pair
# is a one-atom substituent swap (R=Cl -> R=Br, R=H -> R=F), which a
# "!D1" constraint would make undiscoverable. Runaway fragmentation is
# controlled by ``max_fragment_heavy_atoms`` instead.
_CUT_BOND_SMARTS = "[!#1]-!@[!#1]"


@dataclass(frozen=True)
class MatchedPair:
    """One matched molecular pair: two molecules sharing a common core.

    Attributes
    ----------
    mol_a, mol_b : Mol
        The paired molecules.
    core : str
        Canonical SMILES of the shared context (with attachment points).
    transformation : str
        The change, written ``"<frag_a>>>frag_b>"`` in SMIRKS-like form.
    delta_activity : float or None
        ``activity_b - activity_a`` when activities were supplied.
    index_a, index_b : int
        Positions of the two molecules in the input sequence.
    """

    mol_a: "Mol"
    mol_b: "Mol"
    core: str
    transformation: str
    delta_activity: Optional[float] = None
    index_a: int = -1
    index_b: int = -1

    def __repr__(self) -> str:  # pragma: no cover - display only
        delta = "" if self.delta_activity is None else f", d={self.delta_activity:+.2f}"
        return f"MatchedPair({self.transformation}{delta})"


def _fragment_molecule(mol: "Mol", max_cuts: int = 1) -> List[Tuple[str, str]]:
    """Enumerate (context, fragment) SMILES pairs from single/double/triple cuts.

    Returns
    -------
    list of (context_smiles, fragment_smiles)
        Both carry ``[*:n]`` attachment points so they can be re-joined.
    """
    from itertools import combinations

    from rdkit import Chem

    patt = Chem.MolFromSmarts(_CUT_BOND_SMARTS)
    matches = mol.GetSubstructMatches(patt)
    bond_indices = [
        mol.GetBondBetweenAtoms(a, b).GetIdx() for a, b in matches
    ]
    results: List[Tuple[str, str]] = []

    for n_cuts in range(1, max_cuts + 1):
        for bonds in combinations(bond_indices, n_cuts):
            fragmented = Chem.FragmentOnBonds(
                mol, list(bonds), addDummies=True,
                dummyLabels=[(i + 1, i + 1) for i in range(n_cuts)],
            )
            pieces = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=False)
            if len(pieces) != n_cuts + 1:
                # A ring cut can leave fewer pieces than bonds broken.
                continue
            smis = []
            ok = True
            for piece in pieces:
                try:
                    Chem.SanitizeMol(piece)
                    smis.append(Chem.MolToSmiles(piece))
                except Exception:
                    ok = False
                    break
            if not ok:
                continue
            # The largest piece is the context; the rest is the variable part.
            largest = max(range(len(smis)), key=lambda i: pieces[i].GetNumHeavyAtoms())
            context = smis[largest]
            fragment = ".".join(smis[i] for i in range(len(smis)) if i != largest)
            results.append((context, fragment))
    return results


class MatchedMolecularPairs:
    """Identify matched molecular pairs by the fragment-index algorithm.

    Implements the Hussain & Rea approach: every molecule is fragmented
    at each acyclic single bond (and optionally at pairs or triples of
    such bonds); each ``(context, fragment)`` split is indexed by its
    context; and any two molecules sharing a context — but differing in
    the attached fragment — constitute a matched pair whose
    transformation is ``fragment_a >> fragment_b``.

    This is what turns a flat activity table into interpretable SAR:
    the effect of a substituent change, measured across every pair in
    which it occurs.

    Parameters
    ----------
    max_cuts : int, default 1
        Number of bonds cut simultaneously. 1 finds single-substituent
        changes (the vast majority of useful MMPs); 2-3 also finds
        linker and multi-point changes at rapidly growing cost.
    max_fragment_heavy_atoms : int, optional
        Discard splits whose variable fragment is larger than this, which
        keeps pairs interpretable (a "pair" differing by half the
        molecule is not a useful MMP). ``None`` disables the filter.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mols = [Chem.MolFromSmiles(s) for s in ("c1ccccc1Cl", "c1ccccc1Br")]
    >>> pairs = MatchedMolecularPairs().find_pairs(mols)
    >>> len(pairs) >= 1
    True

    References
    ----------
    - Hussain, J. & Rea, C. (2010). "Computationally Efficient Algorithm
      to Identify Matched Molecular Pairs (MMPs) in Large Data Sets."
      J. Chem. Inf. Model., 50(3), 339-348.
      https://doi.org/10.1021/ci900450m
    - Griffen, E. et al. (2011). "Matched Molecular Pairs as a Medicinal
      Chemistry Tool." J. Med. Chem., 54(22), 7739-7750.
      https://doi.org/10.1021/jm200452d
    - Dossetter, A. G., Griffen, E. J. & Leach, A. G. (2013). "Matched
      Molecular Pair Analysis in Drug Discovery." Drug Discov. Today,
      18(15-16), 724-731. https://doi.org/10.1016/j.drudis.2013.03.003
    - RDKit ``FragmentOnBonds`` documentation:
      https://www.rdkit.org/docs/source/rdkit.Chem.rdmolops.html
    """

    def __init__(
        self,
        max_cuts: int = 1,
        max_fragment_heavy_atoms: Optional[int] = 13,
    ) -> None:
        self.max_cuts = max_cuts
        self.max_fragment_heavy_atoms = max_fragment_heavy_atoms

    def _fragment_size_ok(self, fragment_smiles: str) -> bool:
        if self.max_fragment_heavy_atoms is None:
            return True
        from rdkit import Chem

        frag = Chem.MolFromSmiles(fragment_smiles, sanitize=False)
        if frag is None:
            return False
        heavy = sum(
            1 for a in frag.GetAtoms() if a.GetAtomicNum() > 1 and a.GetAtomicNum() != 0
        )
        return heavy <= self.max_fragment_heavy_atoms

    def find_pairs(
        self,
        mols: Sequence["Mol"],
        activities: Optional[Sequence[float]] = None,
    ) -> List[MatchedPair]:
        """Find all matched molecular pairs in a set of molecules.

        Parameters
        ----------
        mols : sequence of Mol
            Molecules to pair up.
        activities : sequence of float, optional
            Parallel activity values (e.g. pIC50). When given, each pair
            carries ``delta_activity = activity_b - activity_a``.

        Returns
        -------
        list of MatchedPair
            All pairs found, de-duplicated by (index_a, index_b, core).
        """
        if activities is not None and len(activities) != len(mols):
            raise ValueError(
                f"activities has length {len(activities)} but mols has {len(mols)}."
            )

        index: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        for i, mol in enumerate(mols):
            if mol is None:
                continue
            seen_here: set[Tuple[str, str]] = set()
            for context, fragment in _fragment_molecule(mol, self.max_cuts):
                if not self._fragment_size_ok(fragment):
                    continue
                if (context, fragment) in seen_here:
                    continue
                seen_here.add((context, fragment))
                index[context].append((i, fragment))

        pairs: List[MatchedPair] = []
        emitted: set[Tuple[int, int, str]] = set()
        for context, entries in index.items():
            if len(entries) < 2:
                continue
            for pos_a in range(len(entries)):
                for pos_b in range(pos_a + 1, len(entries)):
                    i, frag_a = entries[pos_a]
                    j, frag_b = entries[pos_b]
                    if i == j or frag_a == frag_b:
                        continue
                    key = (min(i, j), max(i, j), context)
                    if key in emitted:
                        continue
                    emitted.add(key)
                    delta = None
                    if activities is not None:
                        delta = float(activities[j]) - float(activities[i])
                    pairs.append(
                        MatchedPair(
                            mol_a=mols[i],
                            mol_b=mols[j],
                            core=context,
                            transformation=f"{frag_a}>>{frag_b}",
                            delta_activity=delta,
                            index_a=i,
                            index_b=j,
                        )
                    )
        return pairs


class MMPAnalyzer:
    """Summarize the SAR encoded by a set of matched molecular pairs.

    Aggregates pairs by transformation to answer the medicinal-chemistry
    question "what does this substituent change usually do to potency?",
    which is the basis of MMP-derived design rules.

    Parameters
    ----------
    max_cuts : int, default 1
        Passed to :class:`MatchedMolecularPairs`.
    max_fragment_heavy_atoms : int, optional, default 13
        Passed to :class:`MatchedMolecularPairs`.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mols = [Chem.MolFromSmiles(s) for s in ("c1ccccc1Cl", "c1ccccc1Br")]
    >>> analyzer = MMPAnalyzer()
    >>> pairs = analyzer.find_pairs(mols, [5.0, 6.0])
    >>> len(pairs) >= 1
    True

    References
    ----------
    - Hussain, J. & Rea, C. (2010). J. Chem. Inf. Model., 50(3), 339-348.
      https://doi.org/10.1021/ci900450m
    - Leach, A. G. et al. (2006). "Matched Molecular Pairs as a Guide in
      the Optimization of Pharmaceutical Properties." J. Med. Chem.,
      49(23), 6672-6682. https://doi.org/10.1021/jm0605233
    """

    def __init__(
        self,
        max_cuts: int = 1,
        max_fragment_heavy_atoms: Optional[int] = 13,
    ) -> None:
        self.max_cuts = max_cuts
        self.max_fragment_heavy_atoms = max_fragment_heavy_atoms
        self._finder = MatchedMolecularPairs(
            max_cuts=max_cuts, max_fragment_heavy_atoms=max_fragment_heavy_atoms
        )

    def find_pairs(
        self,
        mols: Sequence["Mol"],
        activities: Optional[Sequence[float]] = None,
    ) -> List[MatchedPair]:
        """Delegate to :meth:`MatchedMolecularPairs.find_pairs`."""
        return self._finder.find_pairs(mols, activities)

    def transformation_summary(self, pairs: Sequence[MatchedPair]) -> "pd.DataFrame":
        """Aggregate activity change per transformation.

        Parameters
        ----------
        pairs : sequence of MatchedPair
            Pairs carrying ``delta_activity``.

        Returns
        -------
        pandas.DataFrame
            Columns ``transformation``, ``count``, ``mean_delta``,
            ``median_delta``, ``std_delta``, sorted by descending
            ``count`` then descending ``mean_delta``. Transformations
            whose pairs have no activity data are omitted.
        """
        import numpy as np
        import pandas as pd

        grouped: Dict[str, List[float]] = defaultdict(list)
        for pair in pairs:
            if pair.delta_activity is not None:
                grouped[pair.transformation].append(pair.delta_activity)

        rows = [
            {
                "transformation": transformation,
                "count": len(deltas),
                "mean_delta": float(np.mean(deltas)),
                "median_delta": float(np.median(deltas)),
                "std_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
            }
            for transformation, deltas in grouped.items()
        ]
        df = pd.DataFrame(
            rows,
            columns=["transformation", "count", "mean_delta", "median_delta", "std_delta"],
        )
        if not df.empty:
            df = df.sort_values(
                ["count", "mean_delta"], ascending=[False, False]
            ).reset_index(drop=True)
        return df

    def to_dataframe(self, pairs: Sequence[MatchedPair]) -> "pd.DataFrame":
        """Render pairs as a table.

        Parameters
        ----------
        pairs : sequence of MatchedPair

        Returns
        -------
        pandas.DataFrame
            Columns ``smiles_a``, ``smiles_b``, ``core``,
            ``transformation``, ``delta_activity``, ``index_a``, ``index_b``.
        """
        import pandas as pd
        from rdkit import Chem

        return pd.DataFrame(
            [
                {
                    "smiles_a": Chem.MolToSmiles(p.mol_a),
                    "smiles_b": Chem.MolToSmiles(p.mol_b),
                    "core": p.core,
                    "transformation": p.transformation,
                    "delta_activity": p.delta_activity,
                    "index_a": p.index_a,
                    "index_b": p.index_b,
                }
                for p in pairs
            ],
            columns=[
                "smiles_a", "smiles_b", "core", "transformation",
                "delta_activity", "index_a", "index_b",
            ],
        )
