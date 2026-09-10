"""QSAR-aware train/test splitters.

A random split flatters a QSAR model. Public activity data is dense with
near-duplicate analogues, so a random split puts close relatives on both
sides and measures interpolation rather than the generalization to new
chemistry that actually matters. Every splitter here exists to make the
evaluation harder in a specific, defensible way.

All splitters expose two interfaces:

- ``split(X, y=None, groups=None)`` — the scikit-learn protocol, yielding
  ``(train_idx, test_idx)`` arrays, usable directly as a ``cv=`` argument.
- ``split_mols(mols, y=None)`` — the same, but taking RDKit molecules for
  the structure-aware splitters that need them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover
    from rdkit.Chem import Mol

__all__ = [
    "BaseSplitter",
    "RandomSplitter",
    "ScaffoldSplitter",
    "StratifiedScaffoldSplitter",
    "ButinaClusterSplitter",
    "SphereExclusionSplitter",
    "MaxMinSplitter",
    "TimeSplitter",
    "KennardStoneSplitter",
    "PerimeterSplitter",
]

_Indices = Tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]


def _scaffold_of(mol: Any, include_chirality: bool = False) -> str:
    """Bemis-Murcko scaffold SMILES, or ``""`` for acyclic molecules."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return str(
            Chem.MolToSmiles(scaffold, isomericSmiles=include_chirality)
        )
    except Exception:
        return ""


def _fingerprints(mols: Sequence[Any], n_bits: int = 2048) -> npt.NDArray[np.float64]:
    """ECFP4 bit matrix, with all-zero rows for invalid molecules."""
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    rows = [
        np.zeros(n_bits, dtype=np.float64)
        if m is None
        else np.asarray(list(gen.GetFingerprint(m)), dtype=np.float64)
        for m in mols
    ]
    return np.asarray(rows, dtype=np.float64)



def _finalize_groups(
    train: List[int], test: List[int], n_train_target: int
) -> _Indices:
    """Guarantee a non-empty train and test set after group assignment.

    Group-based splitting assigns whole scaffolds or clusters, so a
    dataset dominated by a single group can leave one side empty. When
    that happens the largest side is cut at the target boundary, which
    breaks the group-disjointness guarantee for exactly one group - the
    only alternative being an unusable split.
    """
    if train and test:
        return (
            np.array(sorted(train), dtype=np.intp),
            np.array(sorted(test), dtype=np.intp),
        )
    everything = sorted([*train, *test])
    cut = min(max(n_train_target, 1), len(everything) - 1)
    return (
        np.array(everything[:cut], dtype=np.intp),
        np.array(everything[cut:], dtype=np.intp),
    )


class BaseSplitter(ABC):
    """Common machinery: one train/test split, in scikit-learn's shape.

    Parameters
    ----------
    test_size : float, default 0.2
        Fraction of the dataset assigned to the test set.
    random_state : int, optional
        Seed, where the splitter has a stochastic component.

    References
    ----------
    - Wu, Z. et al. (2018). "MoleculeNet: A Benchmark for Molecular
      Machine Learning." Chem. Sci., 9, 513-530.
      https://doi.org/10.1039/C7SC02664A
    - Sheridan, R. P. (2013). "Time-Split Cross-Validation as a Method for
      Estimating the Goodness of Prospective Prediction." J. Chem. Inf.
      Model., 53(4), 783-790. https://doi.org/10.1021/ci400084k
    - scikit-learn cross-validation documentation:
      https://scikit-learn.org/stable/modules/cross_validation.html
    """

    def __init__(
        self, test_size: float = 0.2, random_state: Optional[int] = None
    ) -> None:
        self.test_size = test_size
        self.random_state = random_state

    def _n_test(self, n_samples: int) -> int:
        if not 0.0 < self.test_size < 1.0:
            raise ValueError(
                f"test_size must be in (0, 1), got {self.test_size}."
            )
        if n_samples < 2:
            raise ValueError(
                f"Need at least 2 samples to split, got {n_samples}."
            )
        return max(1, int(round(n_samples * self.test_size)))

    @abstractmethod
    def _split_indices(
        self,
        n_samples: int,
        mols: Optional[Sequence[Any]],
        X: Optional[npt.NDArray[np.float64]],
        y: Optional[npt.NDArray[Any]],
        groups: Optional[npt.NDArray[Any]],
    ) -> _Indices:
        """Return ``(train_idx, test_idx)``."""

    def split(
        self,
        X: Any,
        y: Optional[npt.ArrayLike] = None,
        groups: Optional[npt.ArrayLike] = None,
    ) -> Iterator[_Indices]:
        """Yield one ``(train_idx, test_idx)`` pair.

        Parameters
        ----------
        X : array-like or sequence of Mol
            Feature matrix, or molecules for structure-aware splitters.
        y : array-like, optional
            Labels, used by the stratified splitters.
        groups : array-like, optional
            Group labels, used by :class:`TimeSplitter`.

        Yields
        ------
        train_idx, test_idx : ndarray of int
        """
        mols, arr = self._coerce(X)
        n = len(mols) if mols is not None else len(arr)  # type: ignore[arg-type]
        self._n_test(n)  # validate test_size before any splitter-specific work
        yield self._split_indices(
            n,
            mols,
            arr,
            None if y is None else np.asarray(y),
            None if groups is None else np.asarray(groups),
        )

    def split_mols(
        self, mols: Sequence[Any], y: Optional[npt.ArrayLike] = None
    ) -> Iterator[_Indices]:
        """Yield one split, taking RDKit molecules directly.

        Parameters
        ----------
        mols : sequence of Mol
        y : array-like, optional

        Yields
        ------
        train_idx, test_idx : ndarray of int
        """
        self._n_test(len(mols))  # validate test_size up front
        yield self._split_indices(
            len(mols), list(mols), None, None if y is None else np.asarray(y), None
        )

    def get_n_splits(
        self,
        X: Any = None,
        y: Any = None,
        groups: Any = None,
    ) -> int:
        """Number of splits produced (always 1 for these splitters)."""
        return 1

    @staticmethod
    def _coerce(
        X: Any,
    ) -> Tuple[Optional[List[Any]], Optional[npt.NDArray[np.float64]]]:
        """Detect whether ``X`` holds molecules or a numeric matrix."""
        from rdkit import Chem

        seq = list(X)
        if seq and all(m is None or isinstance(m, Chem.Mol) for m in seq):
            return seq, None
        return None, np.asarray(X, dtype=np.float64)

    def _require_mols(self, mols: Optional[Sequence[Any]]) -> Sequence[Any]:
        if mols is None:
            raise ValueError(
                f"{type(self).__name__} needs RDKit molecules; pass them to "
                "split() or use split_mols()."
            )
        return mols


class RandomSplitter(BaseSplitter):
    """Uniformly random split.

    The baseline every other splitter should be compared against — and,
    on molecular data, almost always the optimistic one.

    Parameters
    ----------
    test_size : float, default 0.2
    random_state : int, optional

    Examples
    --------
    >>> import numpy as np
    >>> X = np.arange(20).reshape(10, 2)
    >>> train, test = next(RandomSplitter(random_state=0).split(X))
    >>> len(train), len(test)
    (8, 2)

    References
    ----------
    - Pedregosa, F. et al. (2011). "Scikit-learn." J. Mach. Learn. Res.,
      12, 2825-2830. https://jmlr.org/papers/v12/pedregosa11a.html
    """

    def _split_indices(
        self,
        n_samples: int,
        mols: Optional[Sequence[Any]],
        X: Optional[npt.NDArray[np.float64]],
        y: Optional[npt.NDArray[Any]],
        groups: Optional[npt.NDArray[Any]],
    ) -> _Indices:
        n_test = self._n_test(n_samples)
        rng = np.random.RandomState(self.random_state)
        order = rng.permutation(n_samples)
        return np.sort(order[n_test:]), np.sort(order[:n_test])


class ScaffoldSplitter(BaseSplitter):
    """Split by Bemis-Murcko scaffold, largest scaffold group first.

    Guarantees that no scaffold appears on both sides, so the test set
    contains only chemistry the model has never seen. This is the
    standard hard split in molecular machine learning, and typically
    drops reported performance substantially relative to a random split —
    which is the point: the gap is the size of the illusion.

    Assigning the largest scaffold groups to training first is
    deterministic (no seed needed) and keeps the rarest, most distinct
    chemotypes in the test set.

    Parameters
    ----------
    test_size : float, default 0.2
    include_chirality : bool, default False
        Treat enantiomers as different scaffolds.
    random_state : int, optional
        Unused; accepted for interface symmetry.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mols = [Chem.MolFromSmiles(s) for s in
    ...         ("c1ccccc1C", "c1ccccc1CC", "c1ccncc1C", "CCO", "CCN")]
    >>> train, test = next(ScaffoldSplitter(test_size=0.4).split_mols(mols))
    >>> set(train) & set(test)
    set()

    References
    ----------
    - Bemis, G. W. & Murcko, M. A. (1996). "The Properties of Known Drugs.
      1. Molecular Frameworks." J. Med. Chem., 39(15), 2887-2893.
      https://doi.org/10.1021/jm9602928
    - Wu, Z. et al. (2018). "MoleculeNet." Chem. Sci., 9, 513-530.
      https://doi.org/10.1039/C7SC02664A
    """

    def __init__(
        self,
        test_size: float = 0.2,
        include_chirality: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(test_size=test_size, random_state=random_state)
        self.include_chirality = include_chirality

    def _scaffold_groups(self, mols: Sequence[Any]) -> List[List[int]]:
        groups: Dict[str, List[int]] = defaultdict(list)
        for i, mol in enumerate(mols):
            groups[_scaffold_of(mol, self.include_chirality)].append(i)
        # Largest first, then by scaffold SMILES so ties break deterministically.
        return [
            members
            for _, members in sorted(
                groups.items(), key=lambda kv: (-len(kv[1]), kv[0])
            )
        ]

    def _split_indices(
        self,
        n_samples: int,
        mols: Optional[Sequence[Any]],
        X: Optional[npt.NDArray[np.float64]],
        y: Optional[npt.NDArray[Any]],
        groups: Optional[npt.NDArray[Any]],
    ) -> _Indices:
        mol_list = self._require_mols(mols)
        n_test = self._n_test(n_samples)
        n_train_target = n_samples - n_test

        train: List[int] = []
        test: List[int] = []
        for members in self._scaffold_groups(mol_list):
            if len(train) + len(members) <= n_train_target:
                train.extend(members)
            else:
                test.extend(members)
        return _finalize_groups(train, test, n_train_target)


class StratifiedScaffoldSplitter(ScaffoldSplitter):
    """Scaffold split that also balances the label distribution.

    Assigns scaffold groups greedily to whichever side is currently
    furthest from its target label mean (regression) or class balance
    (classification). Keeps the scaffold-disjointness guarantee while
    avoiding the common failure where the test set ends up composed
    entirely of inactives.

    Parameters
    ----------
    test_size : float, default 0.2
    include_chirality : bool, default False
    random_state : int, optional

    Examples
    --------
    >>> from rdkit import Chem
    >>> mols = [Chem.MolFromSmiles(s) for s in
    ...         ("c1ccccc1C", "c1ccncc1C", "CCO", "CCN", "c1ccccc1CC")]
    >>> y = [1, 0, 1, 0, 1]
    >>> train, test = next(
    ...     StratifiedScaffoldSplitter(test_size=0.4).split_mols(mols, y)
    ... )
    >>> set(train) & set(test)
    set()

    References
    ----------
    - Wu, Z. et al. (2018). "MoleculeNet." Chem. Sci., 9, 513-530.
      https://doi.org/10.1039/C7SC02664A
    - Sheridan, R. P. (2013). J. Chem. Inf. Model., 53(4), 783-790.
      https://doi.org/10.1021/ci400084k
    """

    def _split_indices(
        self,
        n_samples: int,
        mols: Optional[Sequence[Any]],
        X: Optional[npt.NDArray[np.float64]],
        y: Optional[npt.NDArray[Any]],
        groups: Optional[npt.NDArray[Any]],
    ) -> _Indices:
        if y is None:
            return super()._split_indices(n_samples, mols, X, y, groups)

        mol_list = self._require_mols(mols)
        n_test = self._n_test(n_samples)
        n_train_target = n_samples - n_test
        values = np.asarray(y, dtype=np.float64)
        overall_mean = float(values.mean())

        train: List[int] = []
        test: List[int] = []
        for members in self._scaffold_groups(mol_list):
            if len(train) >= n_train_target:
                test.extend(members)
                continue
            if len(test) >= n_test:
                train.extend(members)
                continue
            # Put the group wherever it pulls that side's mean closer to
            # the overall mean, so neither side drifts to one activity end.
            group_mean = float(values[members].mean())
            train_gap = abs(
                (np.mean(values[train]) if train else overall_mean) - overall_mean
            )
            test_gap = abs(
                (np.mean(values[test]) if test else overall_mean) - overall_mean
            )
            if train_gap >= test_gap:
                train.extend(members)
            else:
                test.extend(members)

        return _finalize_groups(train, test, n_train_target)


class ButinaClusterSplitter(BaseSplitter):
    """Split by Taylor-Butina cluster, keeping whole clusters together.

    A softer alternative to a scaffold split: it groups by overall
    fingerprint similarity rather than exact scaffold identity, so it
    also separates molecules that share no scaffold but are still very
    similar — which a scaffold split happily puts on opposite sides.

    Parameters
    ----------
    test_size : float, default 0.2
    cutoff : float, default 0.35
        Butina distance cutoff (Tanimoto similarity ``1 - cutoff``).
    n_bits : int, default 2048
        Fingerprint length.
    random_state : int, optional

    Examples
    --------
    >>> from rdkit import Chem
    >>> mols = [Chem.MolFromSmiles(s) for s in
    ...         ("CCO", "CCN", "c1ccccc1", "c1ccccc1C", "CCCCCC")]
    >>> train, test = next(ButinaClusterSplitter(test_size=0.4).split_mols(mols))
    >>> set(train) & set(test)
    set()

    References
    ----------
    - Butina, D. (1999). "Unsupervised Data Base Clustering Based on
      Daylight's Fingerprint and Tanimoto Similarity." J. Chem. Inf.
      Comput. Sci., 39(4), 747-750. https://doi.org/10.1021/ci9803381
    """

    def __init__(
        self,
        test_size: float = 0.2,
        cutoff: float = 0.35,
        n_bits: int = 2048,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(test_size=test_size, random_state=random_state)
        self.cutoff = cutoff
        self.n_bits = n_bits

    def _cluster_labels(
        self, mols: Optional[Sequence[Any]], X: Optional[npt.NDArray[np.float64]]
    ) -> npt.NDArray[np.intp]:
        from qsarkit.cluster import ButinaClustering

        features = _fingerprints(mols, self.n_bits) if mols is not None else X
        assert features is not None
        return ButinaClustering(cutoff=self.cutoff).fit(features).labels_

    def _split_indices(
        self,
        n_samples: int,
        mols: Optional[Sequence[Any]],
        X: Optional[npt.NDArray[np.float64]],
        y: Optional[npt.NDArray[Any]],
        groups: Optional[npt.NDArray[Any]],
    ) -> _Indices:
        n_test = self._n_test(n_samples)
        labels = self._cluster_labels(mols, X)

        clusters: Dict[int, List[int]] = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[int(label)].append(i)
        ordered = sorted(clusters.values(), key=lambda m: (-len(m), m[0]))

        train: List[int] = []
        test: List[int] = []
        for members in ordered:
            if len(train) + len(members) <= n_samples - n_test:
                train.extend(members)
            else:
                test.extend(members)
        return _finalize_groups(train, test, n_samples - n_test)


class SphereExclusionSplitter(ButinaClusterSplitter):
    """Split by sphere-exclusion cluster, keeping whole clusters together.

    Like :class:`ButinaClusterSplitter` but using leader-based sphere
    exclusion, which guarantees a minimum distance between cluster
    centres and scales to much larger libraries.

    Parameters
    ----------
    test_size : float, default 0.2
    cutoff : float, default 0.35
    n_bits : int, default 2048
    random_state : int, optional

    Examples
    --------
    >>> from rdkit import Chem
    >>> mols = [Chem.MolFromSmiles(s) for s in ("CCO", "CCN", "c1ccccc1", "CCCC")]
    >>> train, test = next(SphereExclusionSplitter(test_size=0.5).split_mols(mols))
    >>> set(train) & set(test)
    set()

    References
    ----------
    - Hudson, B. D. et al. (1996). "Parameter Based Methods for Compound
      Selection from Chemical Databases." Quant. Struct.-Act. Relat.,
      15(4), 285-289. https://doi.org/10.1002/qsar.19960150402
    - Gobbi, A. & Lee, M.-L. (2003). "DISE: Directed Sphere Exclusion."
      J. Chem. Inf. Comput. Sci., 43(1), 317-323.
      https://doi.org/10.1021/ci025554v
    """

    def _cluster_labels(
        self, mols: Optional[Sequence[Any]], X: Optional[npt.NDArray[np.float64]]
    ) -> npt.NDArray[np.intp]:
        from qsarkit.cluster import SphereExclusionClustering

        features = _fingerprints(mols, self.n_bits) if mols is not None else X
        assert features is not None
        return SphereExclusionClustering(cutoff=self.cutoff).fit(features).labels_


class MaxMinSplitter(BaseSplitter):
    """Put a maximally diverse subset in the *training* set.

    Uses MaxMin picking to choose training compounds that span the
    chemical space as widely as possible, leaving the denser regions for
    testing. This is the split to use when the question is "how few
    compounds do I need to measure?" rather than "how well does this
    extrapolate?" — it is deliberately the *optimistic* structure-aware
    split, and pairs well with a scaffold split as the pessimistic bound.

    Parameters
    ----------
    test_size : float, default 0.2
    n_bits : int, default 2048
    random_state : int, optional
        Seed for the initial pick.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mols = [Chem.MolFromSmiles(s) for s in
    ...         ("CCO", "CCN", "c1ccccc1", "c1ccccc1C", "CCCCCC")]
    >>> train, test = next(MaxMinSplitter(test_size=0.4).split_mols(mols))
    >>> len(train) + len(test)
    5

    References
    ----------
    - Ashton, M. et al. (2002). "Identification of Diverse Database
      Subsets." Quant. Struct.-Act. Relat., 21(6), 598-604.
      https://doi.org/10.1002/qsar.200290002
    """

    def __init__(
        self,
        test_size: float = 0.2,
        n_bits: int = 2048,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(test_size=test_size, random_state=random_state)
        self.n_bits = n_bits

    def _split_indices(
        self,
        n_samples: int,
        mols: Optional[Sequence[Any]],
        X: Optional[npt.NDArray[np.float64]],
        y: Optional[npt.NDArray[Any]],
        groups: Optional[npt.NDArray[Any]],
    ) -> _Indices:
        from qsarkit.cluster import MaxMinPicker

        n_test = self._n_test(n_samples)
        n_train = n_samples - n_test
        features = _fingerprints(mols, self.n_bits) if mols is not None else X
        assert features is not None

        picks = (
            MaxMinPicker(n_to_pick=n_train, seed_index=self.random_state)
            .fit(features)
            .picks_
        )
        train = np.sort(picks)
        test = np.setdiff1d(np.arange(n_samples, dtype=np.intp), train)
        return train.astype(np.intp), test.astype(np.intp)


class TimeSplitter(BaseSplitter):
    """Split chronologically: earliest compounds train, latest test.

    The most honest evaluation available, because it reproduces the real
    prospective task — predicting compounds that had not been made yet.
    Sheridan showed time-split validation gives markedly lower, and much
    more realistic, performance estimates than random or even
    scaffold-based splits.

    Parameters
    ----------
    test_size : float, default 0.2
    random_state : int, optional
        Unused; the split is fully determined by the dates.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.arange(20).reshape(10, 2)
    >>> dates = np.arange(10)
    >>> train, test = next(TimeSplitter(test_size=0.3).split(X, groups=dates))
    >>> bool(dates[train].max() <= dates[test].min())
    True

    References
    ----------
    - Sheridan, R. P. (2013). "Time-Split Cross-Validation as a Method for
      Estimating the Goodness of Prospective Prediction." J. Chem. Inf.
      Model., 53(4), 783-790. https://doi.org/10.1021/ci400084k
    """

    def _split_indices(
        self,
        n_samples: int,
        mols: Optional[Sequence[Any]],
        X: Optional[npt.NDArray[np.float64]],
        y: Optional[npt.NDArray[Any]],
        groups: Optional[npt.NDArray[Any]],
    ) -> _Indices:
        if groups is None:
            raise ValueError(
                "TimeSplitter needs dates or ordinal timestamps in `groups`."
            )
        if len(groups) != n_samples:
            raise ValueError(
                f"groups has length {len(groups)} but there are {n_samples} samples."
            )
        n_test = self._n_test(n_samples)
        order = np.argsort(groups, kind="stable")
        return (
            np.sort(order[: n_samples - n_test]).astype(np.intp),
            np.sort(order[n_samples - n_test :]).astype(np.intp),
        )


class KennardStoneSplitter(BaseSplitter):
    """Kennard-Stone: training set covers the descriptor space uniformly.

    Selects training points to be maximally far apart, starting from the
    two most distant compounds. Deterministic, and produces a training
    set whose convex hull encloses most of the test set — which makes it
    the natural companion to a leverage-based applicability domain,
    since almost every test compound ends up inside it.

    Parameters
    ----------
    test_size : float, default 0.2
    metric : str, default "euclidean"
        Any metric accepted by ``scipy.spatial.distance.cdist``.
    random_state : int, optional
        Unused; the algorithm is deterministic.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.RandomState(0).normal(size=(20, 3))
    >>> train, test = next(KennardStoneSplitter(test_size=0.25).split(X))
    >>> len(train), len(test)
    (15, 5)

    References
    ----------
    - Kennard, R. W. & Stone, L. A. (1969). "Computer Aided Design of
      Experiments." Technometrics, 11(1), 137-148.
      https://doi.org/10.1080/00401706.1969.10490666
    - Snee, R. D. (1977). "Validation of Regression Models: Methods and
      Examples." Technometrics, 19(4), 415-428.
      https://doi.org/10.1080/00401706.1977.10489581
    """

    def __init__(
        self,
        test_size: float = 0.2,
        metric: str = "euclidean",
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(test_size=test_size, random_state=random_state)
        self.metric = metric

    def _split_indices(
        self,
        n_samples: int,
        mols: Optional[Sequence[Any]],
        X: Optional[npt.NDArray[np.float64]],
        y: Optional[npt.NDArray[Any]],
        groups: Optional[npt.NDArray[Any]],
    ) -> _Indices:
        from scipy.spatial.distance import cdist

        features = _fingerprints(mols) if mols is not None else X
        assert features is not None
        n_test = self._n_test(n_samples)
        n_train = n_samples - n_test

        dist = cdist(features, features, metric=self.metric)
        # Seed with the two most distant compounds - the extremes of the space.
        i, j = np.unravel_index(np.argmax(dist), dist.shape)
        selected = [int(i), int(j)]
        remaining = set(range(n_samples)) - set(selected)

        while len(selected) < n_train and remaining:
            rest = np.array(sorted(remaining), dtype=int)
            # Each candidate's distance to its nearest already-selected point;
            # take the candidate that maximizes it.
            nearest = dist[np.ix_(rest, np.array(selected, dtype=int))].min(axis=1)
            selected.append(int(rest[int(np.argmax(nearest))]))
            remaining.discard(selected[-1])

        train = np.array(sorted(selected), dtype=np.intp)
        test = np.setdiff1d(np.arange(n_samples, dtype=np.intp), train)
        return train, test


class PerimeterSplitter(BaseSplitter):
    """Put the outermost compounds in the training set.

    Selects the points furthest from the dataset centroid for training,
    leaving the interior for testing. The mirror image of a scaffold
    split: it makes every prediction an *interpolation*, giving the most
    favourable honest estimate of a model's performance inside its own
    domain.

    Parameters
    ----------
    test_size : float, default 0.2
    random_state : int, optional
        Unused; deterministic.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.RandomState(0).normal(size=(20, 3))
    >>> train, test = next(PerimeterSplitter(test_size=0.25).split(X))
    >>> len(train), len(test)
    (15, 5)

    References
    ----------
    - Martin, T. M. et al. (2012). "Does Rational Selection of Training
      and Test Sets Improve the Outcome of QSAR Modeling?" J. Chem. Inf.
      Model., 52(10), 2570-2578. https://doi.org/10.1021/ci300338w
    """

    def _split_indices(
        self,
        n_samples: int,
        mols: Optional[Sequence[Any]],
        X: Optional[npt.NDArray[np.float64]],
        y: Optional[npt.NDArray[Any]],
        groups: Optional[npt.NDArray[Any]],
    ) -> _Indices:
        features = _fingerprints(mols) if mols is not None else X
        assert features is not None
        n_test = self._n_test(n_samples)
        n_train = n_samples - n_test

        distance = np.linalg.norm(features - features.mean(axis=0), axis=1)
        order = np.argsort(-distance, kind="stable")
        return (
            np.sort(order[:n_train]).astype(np.intp),
            np.sort(order[n_train:]).astype(np.intp),
        )
