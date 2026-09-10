"""Duplicate structure detection with activity-agreement checking."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from rdkit.Chem import Mol

__all__ = ["DuplicateGroup", "DuplicateDetector", "merge_replicates"]

_Level = Literal["inchikey", "smiles", "connectivity", "scaffold"]


@dataclass
class DuplicateGroup:
    """One set of records judged to be the same structure.

    Attributes
    ----------
    key : str
        The identity key the members share.
    indices : list of int
        Positions of the members in the input sequence.
    activities : list of float
        Their activity values, where supplied.
    spread : float
        ``max - min`` of the activities; ``0.0`` when fewer than two are
        known.
    consistent : bool
        Whether ``spread`` is within the detector's tolerance.
    """

    key: str
    indices: List[int]
    activities: List[float] = field(default_factory=list)
    spread: float = 0.0
    consistent: bool = True

    def __len__(self) -> int:
        return len(self.indices)


def _identity_key(mol: Any, level: _Level) -> Optional[str]:
    """Identity string for a molecule at the requested strictness."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    if mol is None:
        return None
    try:
        if level == "inchikey":
            return str(Chem.MolToInchiKey(mol))
        if level == "smiles":
            return str(Chem.MolToSmiles(mol))
        if level == "connectivity":
            # First InChIKey block: the skeleton hash, which ignores
            # stereochemistry and protonation. Two records differing only
            # in a drawn stereocentre are the same compound for most
            # activity data, where stereochemistry is often unreported.
            return str(Chem.MolToInchiKey(mol)).split("-")[0]
        if level == "scaffold":
            return str(
                Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
            )
    except Exception:
        return None
    raise ValueError(
        "level must be 'inchikey', 'smiles', 'connectivity' or 'scaffold', "
        f"got {level!r}."
    )


class DuplicateDetector:
    """Find duplicate structures and check whether their activities agree.

    Duplicates are endemic in public activity data: the same compound is
    re-measured across papers, deposited under different salt forms, or
    drawn with different tautomers. Removing them blindly discards
    replicate information; keeping them inflates cross-validated
    performance, because the same structure lands in both the training
    and test folds.

    The useful question is not "are these duplicates?" but "do the
    duplicates *agree*?" A pair reported as 5 nM and 50 uM is not a
    replicate — it is a data error, an assay difference, or a wrong
    structure, and averaging the two produces a value that describes
    neither.

    Parameters
    ----------
    level : {"inchikey", "smiles", "connectivity", "scaffold"}, default "inchikey"
        What counts as the same structure. ``"connectivity"`` ignores
        stereochemistry and charge; ``"scaffold"`` collapses whole
        Bemis-Murcko series and is a deliberately blunt instrument.
    activity_tolerance : float, default 1.0
        Maximum activity spread, in the units supplied, for a duplicate
        group to be called consistent. On a log scale 1.0 means a
        ten-fold disagreement.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mols = [Chem.MolFromSmiles(s) for s in ("CCO", "OCC", "c1ccccc1")]
    >>> groups = DuplicateDetector().find_duplicates(mols, [5.0, 5.2, 7.0])
    >>> len(groups)
    1
    >>> groups[0].indices
    [0, 1]

    References
    ----------
    - Fourches, D., Muratov, E. & Tropsha, A. (2010). "Trust, But Verify:
      On the Importance of Chemical Structure Curation in
      Cheminformatics and QSAR Modeling Research." J. Chem. Inf. Model.,
      50(7), 1189-1204. https://doi.org/10.1021/ci100176x
    - Fourches, D., Muratov, E. & Tropsha, A. (2016). "Trust, but Verify
      II: A Practical Guide to Chemogenomics Data Curation." J. Chem.
      Inf. Model., 56(7), 1243-1252.
      https://doi.org/10.1021/acs.jcim.6b00129
    - Heller, S. R. et al. (2015). "InChI, the IUPAC International
      Chemical Identifier." J. Cheminform., 7, 23.
      https://doi.org/10.1186/s13321-015-0068-4
    - Kramer, C. et al. (2012). "The Experimental Uncertainty of
      Heterogeneous Public Ki Data." J. Med. Chem., 55(11), 5165-5173.
      https://doi.org/10.1021/jm300131x
    """

    def __init__(
        self,
        level: _Level = "inchikey",
        activity_tolerance: float = 1.0,
    ) -> None:
        self.level = level
        self.activity_tolerance = activity_tolerance

    def find_duplicates(
        self,
        mols: Sequence[Any],
        activities: Optional[npt.ArrayLike] = None,
    ) -> List[DuplicateGroup]:
        """Group records that share a structure.

        Parameters
        ----------
        mols : sequence of Mol
            Molecules to check. ``None`` entries are skipped.
        activities : array-like, optional
            Parallel activity values. When given, each group reports its
            spread and whether it is consistent.

        Returns
        -------
        list of DuplicateGroup
            Only groups with two or more members, ordered by first
            appearance.
        """
        if self.activity_tolerance < 0:
            raise ValueError(
                f"activity_tolerance must be non-negative, got "
                f"{self.activity_tolerance}."
            )
        values = None if activities is None else np.asarray(activities, dtype=np.float64)
        if values is not None and len(values) != len(mols):
            raise ValueError(
                f"activities has length {len(values)} but there are "
                f"{len(mols)} molecules."
            )

        buckets: Dict[str, List[int]] = defaultdict(list)
        for i, mol in enumerate(mols):
            key = _identity_key(mol, self.level)
            if key is not None:
                buckets[key].append(i)

        groups: List[DuplicateGroup] = []
        for key, indices in buckets.items():
            if len(indices) < 2:
                continue
            group = DuplicateGroup(key=key, indices=sorted(indices))
            if values is not None:
                group.activities = [float(values[i]) for i in group.indices]
                finite = [a for a in group.activities if np.isfinite(a)]
                group.spread = float(max(finite) - min(finite)) if len(finite) > 1 else 0.0
                group.consistent = group.spread <= self.activity_tolerance
            groups.append(group)
        groups.sort(key=lambda g: g.indices[0])
        return groups

    def report(
        self,
        mols: Sequence[Any],
        activities: Optional[npt.ArrayLike] = None,
    ) -> Dict[str, Any]:
        """Summarize the duplicate content of a dataset.

        Parameters
        ----------
        mols : sequence of Mol
        activities : array-like, optional

        Returns
        -------
        dict
            ``n_records``, ``n_unique``, ``n_duplicate_groups``,
            ``n_duplicate_records`` (records that are not the first of
            their group), ``duplicate_fraction``, ``n_inconsistent``
            (groups whose activities disagree beyond tolerance) and
            ``max_spread``.
        """
        groups = self.find_duplicates(mols, activities)
        n_records = len(mols)
        redundant = sum(len(g) - 1 for g in groups)
        spreads = [g.spread for g in groups if g.activities]
        return {
            "n_records": n_records,
            "n_unique": n_records - redundant,
            "n_duplicate_groups": len(groups),
            "n_duplicate_records": redundant,
            "duplicate_fraction": redundant / n_records if n_records else 0.0,
            "n_inconsistent": sum(1 for g in groups if not g.consistent),
            "max_spread": max(spreads) if spreads else 0.0,
        }

    def to_dataframe(self, groups: Sequence[DuplicateGroup]) -> "pd.DataFrame":
        """Render duplicate groups as a table.

        Parameters
        ----------
        groups : sequence of DuplicateGroup

        Returns
        -------
        pandas.DataFrame
            Columns ``key``, ``n_records``, ``indices``, ``activities``,
            ``spread``, ``consistent``.
        """
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "key": g.key,
                    "n_records": len(g),
                    "indices": g.indices,
                    "activities": g.activities,
                    "spread": g.spread,
                    "consistent": g.consistent,
                }
                for g in groups
            ],
            columns=[
                "key", "n_records", "indices", "activities", "spread", "consistent"
            ],
        )


def merge_replicates(
    mols: Sequence[Any],
    activities: npt.ArrayLike,
    level: _Level = "inchikey",
    method: Literal["mean", "median", "geometric_mean", "min", "max"] = "median",
    max_spread: Optional[float] = 1.0,
    log_scale: bool = True,
) -> Tuple[List[Any], npt.NDArray[np.float64], Dict[str, Any]]:
    """Collapse replicate measurements into one value per structure.

    Parameters
    ----------
    mols : sequence of Mol
        Molecules, one per record.
    activities : array-like
        Activity values, one per record.
    level : {"inchikey", "smiles", "connectivity", "scaffold"}, default "inchikey"
        Identity level used to group replicates.
    method : {"mean", "median", "geometric_mean", "min", "max"}, default "median"
        How to combine agreeing replicates. The median is the safer
        default: activity data carries occasional order-of-magnitude
        transcription errors, and one of those moves a mean far more
        than it moves a median.
    max_spread : float, optional
        Discard groups disagreeing by more than this. ``None`` keeps all.
    log_scale : bool, default True
        Whether ``activities`` are already logarithmic (pIC50 and the
        like). ``geometric_mean`` requires linear, positive values, so
        it is refused when this is True — averaging log values
        arithmetically already *is* the geometric mean.

    Returns
    -------
    mols : list of Mol
        One representative molecule per retained group, in order of
        first appearance.
    activities : ndarray
        The merged values.
    report : dict
        ``n_input``, ``n_output``, ``n_merged``, ``n_discarded`` and
        ``discarded_keys``.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mols = [Chem.MolFromSmiles(s) for s in ("CCO", "OCC", "c1ccccc1")]
    >>> merged, y, report = merge_replicates(mols, [5.0, 5.4, 7.0])
    >>> len(merged), report["n_merged"]
    (2, 1)

    References
    ----------
    - Fourches, D., Muratov, E. & Tropsha, A. (2010). J. Chem. Inf.
      Model., 50(7), 1189-1204. https://doi.org/10.1021/ci100176x
    - Kramer, C. et al. (2012). "The Experimental Uncertainty of
      Heterogeneous Public Ki Data." J. Med. Chem., 55(11), 5165-5173.
      https://doi.org/10.1021/jm300131x
    - Kalliokoski, T. et al. (2013). "Comparability of Mixed IC50 Data."
      PLoS ONE, 8(4), e61007. https://doi.org/10.1371/journal.pone.0061007
    """
    values = np.asarray(activities, dtype=np.float64)
    if len(values) != len(mols):
        raise ValueError(
            f"activities has length {len(values)} but there are {len(mols)} molecules."
        )
    if method == "geometric_mean":
        if log_scale:
            raise ValueError(
                "geometric_mean is for linear concentrations; on a log scale "
                "the arithmetic mean already is the geometric mean. Pass "
                "log_scale=False, or use method='mean'."
            )
        if np.any(values <= 0):
            raise ValueError(
                "geometric_mean requires strictly positive activities."
            )

    combiners = {
        "mean": lambda v: float(np.mean(v)),
        "median": lambda v: float(np.median(v)),
        "geometric_mean": lambda v: float(np.exp(np.mean(np.log(v)))),
        "min": lambda v: float(np.min(v)),
        "max": lambda v: float(np.max(v)),
    }
    if method not in combiners:
        raise ValueError(
            f"method must be one of {sorted(combiners)}, got {method!r}."
        )

    buckets: Dict[str, List[int]] = defaultdict(list)
    unkeyed: List[int] = []
    for i, mol in enumerate(mols):
        key = _identity_key(mol, level)
        (unkeyed if key is None else buckets[key]).append(i)

    kept: List[Tuple[int, float]] = []
    discarded: List[str] = []
    n_merged = 0
    for key, indices in buckets.items():
        group = values[np.asarray(indices, dtype=int)]
        if len(indices) > 1:
            spread = float(np.nanmax(group) - np.nanmin(group))
            if max_spread is not None and spread > max_spread:
                discarded.append(key)
                continue
            n_merged += 1
            kept.append((indices[0], combiners[method](group)))
        else:
            kept.append((indices[0], float(group[0])))

    for i in unkeyed:
        kept.append((i, float(values[i])))

    kept.sort(key=lambda pair: pair[0])
    return (
        [mols[i] for i, _ in kept],
        np.array([v for _, v in kept], dtype=np.float64),
        {
            "n_input": len(mols),
            "n_output": len(kept),
            "n_merged": n_merged,
            "n_discarded": len(discarded),
            "discarded_keys": discarded,
        },
    )
