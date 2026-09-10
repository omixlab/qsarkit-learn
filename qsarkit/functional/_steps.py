"""Curation steps for the functional pipe API.

Every step here is written as ``(X, y=None, **params) -> (X, y)`` and
wrapped with :func:`~qsarkit.functional.step`, so each one works both as a
deferred pipe stage (``desalt()``) and as a plain function called directly
(``desalt(mols, y)``).

Steps that drop molecules drop the matching ``y`` entries too, so the two
stay index-aligned all the way down the pipe.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt

from qsarkit.functional._core import step

if TYPE_CHECKING:  # pragma: no cover
    from rdkit.Chem import Mol

__all__ = [
    "standardize",
    "desalt",
    "neutralize",
    "canonicalize_tautomers",
    "deglycate",
    "remove_protecting_groups",
    "drop_invalid",
    "remove_duplicates",
    "balance",
    "keep_if",
    "drop_if",
    "filter_by_property",
    "to_pactivity",
    "sample",
    "shuffle",
    "apply",
]

_Payload = Tuple[List[Any], Optional[npt.NDArray[Any]]]


def _subset(
    mols: List[Any], y: Optional[npt.NDArray[Any]], keep: List[int]
) -> _Payload:
    """Select positions ``keep`` from both molecules and labels."""
    return [mols[i] for i in keep], (None if y is None else y[np.asarray(keep, dtype=int)])


@step
def standardize(
    X: List[Any],
    y: Optional[npt.NDArray[Any]] = None,
    remove_salts: bool = True,
    neutralize: bool = True,
    normalize_tautomers: bool = True,
    handle_stereochemistry: str = "retain",
    normalize_hydrogens: bool = True,
) -> _Payload:
    """Run the full standardization pipeline over the set.

    Sanitizes, strips salts and solvates, neutralizes charges,
    canonicalizes tautomers, handles stereochemistry and normalizes
    hydrogens. Molecules that fail become ``None`` rather than raising,
    so one bad record cannot abort a pipeline; follow with
    :func:`drop_invalid` to remove them.

    Parameters
    ----------
    X : list of Mol
    y : ndarray, optional
    remove_salts, neutralize, normalize_tautomers, normalize_hydrogens : bool
        Stage toggles, passed to
        :class:`~qsarkit.chemistry.MolecularStandardizer`.
    handle_stereochemistry : {"retain", "remove"}
        Whether to keep stereochemistry.

    Returns
    -------
    (list of Mol, ndarray or None)

    Examples
    --------
    >>> from qsarkit.functional import molecules, standardize
    >>> mols, _ = molecules(["CC(=O)[O-].[Na+]"]) >> standardize()
    >>> from rdkit import Chem
    >>> Chem.MolToSmiles(mols[0])
    'CC(=O)O'

    References
    ----------
    - RDKit MolStandardize documentation:
      https://www.rdkit.org/docs/source/rdkit.Chem.MolStandardize.html
    - OECD (2007). Guidance Document No. 69, ENV/JM/MONO(2007)2.
      https://doi.org/10.1787/9789264085442-en
    """
    from qsarkit.chemistry import MolecularStandardizer

    standardizer = MolecularStandardizer(
        remove_salts=remove_salts,
        neutralize=neutralize,
        normalize_tautomers=normalize_tautomers,
        handle_stereochemistry=handle_stereochemistry,
        normalize_hydrogens=normalize_hydrogens,
    )
    return list(standardizer.transform(X)), y


@step
def desalt(X: List[Any], y: Optional[npt.NDArray[Any]] = None) -> _Payload:
    """Keep only the largest organic fragment of each molecule.

    Strips counter-ions, solvates and hydrates. Activity data is
    routinely reported for salt forms while the activity belongs to the
    parent, so this is usually the first curation step.

    Parameters
    ----------
    X : list of Mol
    y : ndarray, optional

    Returns
    -------
    (list of Mol, ndarray or None)

    Examples
    --------
    >>> from qsarkit.functional import desalt, molecules
    >>> mols, _ = molecules(["CC(=O)[O-].[Na+]"]) >> desalt()
    >>> from rdkit import Chem
    >>> Chem.MolToSmiles(mols[0])
    'CC(=O)[O-]'

    References
    ----------
    - RDKit ``LargestFragmentChooser``:
      https://www.rdkit.org/docs/source/rdkit.Chem.MolStandardize.rdMolStandardize.html
    """
    from rdkit.Chem.MolStandardize import rdMolStandardize

    chooser = rdMolStandardize.LargestFragmentChooser()
    out: List[Any] = []
    for mol in X:
        if mol is None:
            out.append(None)
            continue
        try:
            out.append(chooser.choose(mol))
        except Exception:
            out.append(None)
    return out, y


@step
def neutralize(X: List[Any], y: Optional[npt.NDArray[Any]] = None) -> _Payload:
    """Neutralize charges where a neutral form exists.

    Leaves permanent charges (quaternary ammonium, for instance)
    untouched.

    Parameters
    ----------
    X : list of Mol
    y : ndarray, optional

    Returns
    -------
    (list of Mol, ndarray or None)

    Examples
    --------
    >>> from qsarkit.functional import molecules, neutralize
    >>> mols, _ = molecules(["CC(=O)[O-]"]) >> neutralize()
    >>> from rdkit import Chem
    >>> Chem.MolToSmiles(mols[0])
    'CC(=O)O'

    References
    ----------
    - RDKit ``Uncharger``:
      https://www.rdkit.org/docs/source/rdkit.Chem.MolStandardize.rdMolStandardize.html
    """
    from rdkit.Chem.MolStandardize import rdMolStandardize

    uncharger = rdMolStandardize.Uncharger()
    out: List[Any] = []
    for mol in X:
        if mol is None:
            out.append(None)
            continue
        try:
            out.append(uncharger.uncharge(mol))
        except Exception:
            out.append(None)
    return out, y


@step
def canonicalize_tautomers(
    X: List[Any], y: Optional[npt.NDArray[Any]] = None
) -> _Payload:
    """Map each molecule to its canonical tautomer.

    Without this, the same compound drawn in two tautomeric forms counts
    as two distinct structures, which silently defeats duplicate removal.

    Parameters
    ----------
    X : list of Mol
    y : ndarray, optional

    Returns
    -------
    (list of Mol, ndarray or None)

    Examples
    --------
    >>> from qsarkit.functional import canonicalize_tautomers, molecules
    >>> mols, _ = molecules(["Oc1ccccn1"]) >> canonicalize_tautomers()
    >>> mols[0] is not None
    True

    References
    ----------
    - Sitzmann, M. et al. (2010). "Tautomerism in Large Databases."
      J. Comput. Aided Mol. Des., 24, 521-551.
      https://doi.org/10.1007/s10822-010-9346-4
    """
    from rdkit.Chem.MolStandardize import rdMolStandardize

    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetRemoveSp3Stereo(False)
    out: List[Any] = []
    for mol in X:
        if mol is None:
            out.append(None)
            continue
        try:
            out.append(enumerator.Canonicalize(mol))
        except Exception:
            out.append(None)
    return out, y


@step
def deglycate(
    X: List[Any],
    y: Optional[npt.NDArray[Any]] = None,
    keep_originals: bool = False,
) -> _Payload:
    """Remove sugar moieties, keeping the aglycone.

    Natural-product datasets are full of glycosides whose activity
    belongs to the aglycone. With ``keep_originals=True`` the glycoside
    is kept alongside its aglycone (and its label duplicated), which is
    what you want when you are augmenting a training set rather than
    replacing entries.

    Parameters
    ----------
    X : list of Mol
    y : ndarray, optional
    keep_originals : bool, default False
        Keep the original glycoside in addition to the aglycone.

    Returns
    -------
    (list of Mol, ndarray or None)

    Examples
    --------
    >>> from qsarkit.functional import deglycate, molecules
    >>> glucoside = "OC[C@H]1O[C@@H](Oc2ccccc2)[C@H](O)[C@@H](O)[C@@H]1O"
    >>> mols, _ = molecules([glucoside]) >> deglycate()
    >>> from rdkit import Chem
    >>> Chem.MolToSmiles(mols[0])
    'c1ccccc1'
    >>> mols, _ = molecules([glucoside]) >> deglycate(keep_originals=True)
    >>> len(mols)
    2

    References
    ----------
    - Fischer, J. et al. (2020). "The Sugar Removal Utility (SRU)."
      Molecules, 25(8), 1988. https://doi.org/10.3390/molecules25081988
    """
    from qsarkit.chemistry import GlycanRemover

    remover = GlycanRemover()
    out: List[Any] = []
    keep_index: List[int] = []
    for i, mol in enumerate(X):
        if mol is None:
            out.append(None)
            keep_index.append(i)
            continue
        result = remover.remove(mol)
        aglycone = result["aglycone"]
        if keep_originals and result["removed_fragments"]:
            out.append(mol)
            keep_index.append(i)
        out.append(aglycone)
        keep_index.append(i)
    return out, (None if y is None else y[np.asarray(keep_index, dtype=int)])


@step
def remove_protecting_groups(
    X: List[Any], y: Optional[npt.NDArray[Any]] = None
) -> _Payload:
    """Strip protecting groups, linkers, tags and click handles.

    Parameters
    ----------
    X : list of Mol
    y : ndarray, optional

    Returns
    -------
    (list of Mol, ndarray or None)

    Examples
    --------
    >>> from qsarkit.functional import molecules, remove_protecting_groups
    >>> mols, _ = molecules(["CC(C)(C)OC(=O)NCc1ccccc1"]) >> remove_protecting_groups()
    >>> from rdkit import Chem
    >>> Chem.MolToSmiles(mols[0])
    'NCc1ccccc1'

    References
    ----------
    - Wuts, P. G. M. & Greene, T. W. (2014). "Greene's Protective Groups
      in Organic Synthesis," 5th ed. Wiley.
      https://doi.org/10.1002/9781118978075
    """
    from qsarkit.chemistry import FragmentRemover

    return list(FragmentRemover().transform(X)), y


@step
def drop_invalid(X: List[Any], y: Optional[npt.NDArray[Any]] = None) -> _Payload:
    """Drop ``None`` entries, and their labels with them.

    Parameters
    ----------
    X : list of Mol
    y : ndarray, optional

    Returns
    -------
    (list of Mol, ndarray or None)

    Examples
    --------
    >>> from qsarkit.functional import drop_invalid, molecules
    >>> mols, y = molecules(["CCO", "!!bad!!"], [1.0, 2.0]) >> drop_invalid()
    >>> len(mols), y.tolist()
    (1, [1.0])
    """
    keep = [i for i, m in enumerate(X) if m is not None]
    return _subset(X, y, keep)


@step
def remove_duplicates(
    X: List[Any],
    y: Optional[npt.NDArray[Any]] = None,
    on: Literal["inchikey", "smiles", "scaffold"] = "inchikey",
    agg: Literal["mean", "median", "min", "max", "first"] = "mean",
    max_spread: Optional[float] = None,
) -> _Payload:
    """Collapse duplicate structures, aggregating their labels.

    Public activity data is full of the same compound measured several
    times. Dropping duplicates blindly throws away replicate information;
    averaging them without checking hides disagreement. ``max_spread``
    lets you do both — average the consistent ones and discard the pairs
    that disagree by more than you are willing to accept.

    Parameters
    ----------
    X : list of Mol
    y : ndarray, optional
    on : {"inchikey", "smiles", "scaffold"}, default "inchikey"
        What counts as "the same molecule". InChIKey is the most robust;
        ``scaffold`` collapses whole Bemis-Murcko series and is a
        deliberately blunt instrument.
    agg : {"mean", "median", "min", "max", "first"}, default "mean"
        How to combine the labels of duplicates. Ignored when unlabelled.
    max_spread : float, optional
        Discard duplicate groups whose labels span more than this. For
        log-scale activities, 1.0 (a ten-fold disagreement) is a common
        cutoff.

    Returns
    -------
    (list of Mol, ndarray or None)

    Examples
    --------
    >>> from qsarkit.functional import molecules, remove_duplicates
    >>> mols, y = (
    ...     molecules(["CCO", "CCO", "c1ccccc1"], [1.0, 3.0, 5.0])
    ...     >> remove_duplicates(agg="mean")
    ... )
    >>> len(mols), sorted(y.tolist())
    (2, [2.0, 5.0])

    References
    ----------
    - Fourches, D., Muratov, E. & Tropsha, A. (2010). "Trust, But Verify."
      J. Chem. Inf. Model., 50(7), 1189-1204.
      https://doi.org/10.1021/ci100176x
    - Heller, S. R. et al. (2015). "InChI, the IUPAC International
      Chemical Identifier." J. Cheminform., 7, 23.
      https://doi.org/10.1186/s13321-015-0068-4
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    def key_of(mol: Any) -> Optional[str]:
        if mol is None:
            return None
        try:
            if on == "inchikey":
                return str(Chem.MolToInchiKey(mol))
            if on == "smiles":
                return str(Chem.MolToSmiles(mol))
            if on == "scaffold":
                return str(
                    Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
                )
        except Exception:
            return None
        raise ValueError(
            f"on must be 'inchikey', 'smiles' or 'scaffold', got {on!r}."
        )

    if on not in ("inchikey", "smiles", "scaffold"):
        raise ValueError(
            f"on must be 'inchikey', 'smiles' or 'scaffold', got {on!r}."
        )
    aggregators: Dict[str, Callable[[npt.NDArray[Any]], Any]] = {
        "mean": np.mean,
        "median": np.median,
        "min": np.min,
        "max": np.max,
        "first": lambda v: v[0],
    }
    if agg not in aggregators:
        raise ValueError(
            f"agg must be one of {sorted(aggregators)}, got {agg!r}."
        )

    groups: Dict[Any, List[int]] = defaultdict(list)
    unkeyed: List[int] = []
    for i, mol in enumerate(X):
        key = key_of(mol)
        if key is None:
            unkeyed.append(i)
        else:
            groups[key].append(i)

    keep: List[int] = []
    new_y: List[Any] = []
    for indices in groups.values():
        if y is None:
            keep.append(indices[0])
            continue
        values = y[np.asarray(indices, dtype=int)]
        if max_spread is not None and len(values) > 1:
            numeric = np.asarray(values, dtype=np.float64)
            if float(numeric.max() - numeric.min()) > max_spread:
                continue
        keep.append(indices[0])
        new_y.append(aggregators[agg](values) if len(values) > 1 else values[0])

    # Unparseable molecules have no key, so they cannot be de-duplicated;
    # keep them for drop_invalid() to deal with explicitly.
    for i in unkeyed:
        keep.append(i)
        if y is not None:
            new_y.append(y[i])

    order = np.argsort(np.asarray(keep, dtype=int), kind="stable")
    keep_sorted = [keep[i] for i in order]
    if y is None:
        return [X[i] for i in keep_sorted], None
    y_sorted = np.array([new_y[i] for i in order])
    return [X[i] for i in keep_sorted], y_sorted


@step
def balance(
    X: List[Any],
    y: Optional[npt.NDArray[Any]] = None,
    method: Literal["undersample", "oversample"] = "undersample",
    random_state: Optional[int] = None,
) -> _Payload:
    """Balance a classification set across its label values.

    Parameters
    ----------
    X : list of Mol
    y : ndarray
        Class labels. Required — balancing an unlabelled set is
        meaningless, so this raises rather than silently doing nothing.
    method : {"undersample", "oversample"}, default "undersample"
        Undersampling discards majority-class molecules; oversampling
        duplicates minority-class ones.
    random_state : int, optional
        Seed, for reproducibility.

    Returns
    -------
    (list of Mol, ndarray)

    Examples
    --------
    >>> from qsarkit.functional import balance, molecules
    >>> mols, y = (
    ...     molecules(["CCO", "CCN", "CCC", "c1ccccc1"], [0, 0, 0, 1])
    ...     >> balance(random_state=0)
    ... )
    >>> sorted(y.tolist())
    [0, 1]

    References
    ----------
    - He, H. & Garcia, E. A. (2009). "Learning from Imbalanced Data."
      IEEE Trans. Knowl. Data Eng., 21(9), 1263-1284.
      https://doi.org/10.1109/TKDE.2008.239
    - Chawla, N. V. et al. (2002). "SMOTE: Synthetic Minority
      Over-sampling Technique." J. Artif. Intell. Res., 16, 321-357.
      https://doi.org/10.1613/jair.953
    """
    if y is None:
        raise ValueError(
            "balance() needs labels; the MoleculeSet is unlabelled."
        )
    if method not in ("undersample", "oversample"):
        raise ValueError(
            f"method must be 'undersample' or 'oversample', got {method!r}."
        )

    rng = np.random.RandomState(random_state)
    classes, counts = np.unique(y, return_counts=True)
    target = counts.min() if method == "undersample" else counts.max()

    keep: List[int] = []
    for cls, count in zip(classes, counts):
        idx = np.flatnonzero(y == cls)
        if count == target:
            chosen = idx
        elif method == "undersample":
            chosen = rng.choice(idx, size=target, replace=False)
        else:
            chosen = rng.choice(idx, size=target, replace=True)
        keep.extend(int(i) for i in chosen)

    keep.sort()
    return _subset(X, y, keep)


@step
def keep_if(
    X: List[Any],
    y: Optional[npt.NDArray[Any]] = None,
    predicate: Optional[Callable[[Any], bool]] = None,
) -> _Payload:
    """Keep molecules satisfying a predicate.

    Parameters
    ----------
    X : list of Mol
    y : ndarray, optional
    predicate : callable
        ``f(mol) -> bool``. ``None`` molecules are always dropped.

    Returns
    -------
    (list of Mol, ndarray or None)

    Examples
    --------
    >>> from qsarkit.functional import keep_if, molecules
    >>> mols, _ = molecules(["CCO", "c1ccccc1"]) >> keep_if(
    ...     predicate=lambda m: m.GetNumAtoms() > 3
    ... )
    >>> len(mols)
    1
    """
    if predicate is None:
        raise ValueError("keep_if() requires a `predicate`.")
    keep = [i for i, m in enumerate(X) if m is not None and predicate(m)]
    return _subset(X, y, keep)


@step
def drop_if(
    X: List[Any],
    y: Optional[npt.NDArray[Any]] = None,
    predicate: Optional[Callable[[Any], bool]] = None,
) -> _Payload:
    """Drop molecules satisfying a predicate.

    Parameters
    ----------
    X : list of Mol
    y : ndarray, optional
    predicate : callable
        ``f(mol) -> bool``. ``None`` molecules are kept, so that
        :func:`drop_invalid` remains the one place invalid entries are
        removed.

    Returns
    -------
    (list of Mol, ndarray or None)

    Examples
    --------
    >>> from qsarkit.functional import drop_if, molecules
    >>> mols, _ = molecules(["CCO", "c1ccccc1"]) >> drop_if(
    ...     predicate=lambda m: m.GetNumAtoms() > 3
    ... )
    >>> len(mols)
    1
    """
    if predicate is None:
        raise ValueError("drop_if() requires a `predicate`.")
    keep = [i for i, m in enumerate(X) if m is None or not predicate(m)]
    return _subset(X, y, keep)


@step
def filter_by_property(
    X: List[Any],
    y: Optional[npt.NDArray[Any]] = None,
    mw: Optional[Tuple[float, float]] = None,
    logp: Optional[Tuple[float, float]] = None,
    heavy_atoms: Optional[Tuple[int, int]] = None,
    rotatable_bonds: Optional[Tuple[int, int]] = None,
) -> _Payload:
    """Keep molecules whose physicochemical properties fall in given ranges.

    Each bound is an inclusive ``(low, high)`` tuple; ``None`` disables
    that filter.

    Parameters
    ----------
    X : list of Mol
    y : ndarray, optional
    mw : (float, float), optional
        Molecular-weight range.
    logp : (float, float), optional
        Wildman-Crippen logP range.
    heavy_atoms : (int, int), optional
        Heavy-atom count range.
    rotatable_bonds : (int, int), optional
        Rotatable-bond count range.

    Returns
    -------
    (list of Mol, ndarray or None)

    Examples
    --------
    >>> from qsarkit.functional import filter_by_property, molecules
    >>> mols, _ = molecules(["CCO", "CCCCCCCCCCCCCCCCCC"]) >> filter_by_property(
    ...     mw=(0, 100)
    ... )
    >>> len(mols)
    1

    References
    ----------
    - Wildman, S. A. & Crippen, G. M. (1999). "Prediction of
      Physicochemical Parameters by Atomic Contributions." J. Chem. Inf.
      Comput. Sci., 39(5), 868-873. https://doi.org/10.1021/ci990307l
    - Lipinski, C. A. et al. (2001). Adv. Drug Deliv. Rev., 46(1-3), 3-26.
      https://doi.org/10.1016/S0169-409X(96)00423-1
    """
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    checks: List[Tuple[Callable[[Any], float], Tuple[float, float]]] = []
    if mw is not None:
        checks.append((Descriptors.MolWt, mw))
    if logp is not None:
        checks.append((Crippen.MolLogP, logp))
    if heavy_atoms is not None:
        checks.append((lambda m: m.GetNumHeavyAtoms(), heavy_atoms))
    if rotatable_bonds is not None:
        checks.append((rdMolDescriptors.CalcNumRotatableBonds, rotatable_bonds))

    keep = [
        i
        for i, m in enumerate(X)
        if m is not None
        and all(low <= fn(m) <= high for fn, (low, high) in checks)
    ]
    return _subset(X, y, keep)


@step
def to_pactivity(
    X: List[Any],
    y: Optional[npt.NDArray[Any]] = None,
    unit: Literal["M", "mM", "uM", "nM", "pM"] = "nM",
) -> _Payload:
    """Convert concentration labels to pActivity (``-log10`` molar).

    QSAR models should be fitted on a log scale: potency spans orders of
    magnitude, and the activity-cliff and SALI thresholds throughout
    qsarkit are all expressed in log units.

    Parameters
    ----------
    X : list of Mol
    y : ndarray
        Concentrations in ``unit``. Non-positive values become ``nan``,
        since their logarithm is undefined.
    unit : {"M", "mM", "uM", "nM", "pM"}, default "nM"
        Unit of the incoming values.

    Returns
    -------
    (list of Mol, ndarray)

    Examples
    --------
    >>> from qsarkit.functional import molecules, to_pactivity
    >>> _, y = molecules(["CCO"], [1.0]) >> to_pactivity(unit="nM")
    >>> round(float(y[0]), 2)
    9.0
    """
    if y is None:
        raise ValueError("to_pactivity() needs labels; the set is unlabelled.")
    factors = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "nM": 1e-9, "pM": 1e-12}
    if unit not in factors:
        raise ValueError(f"unit must be one of {sorted(factors)}, got {unit!r}.")

    molar = np.asarray(y, dtype=np.float64) * factors[unit]
    with np.errstate(divide="ignore", invalid="ignore"):
        converted = -np.log10(np.where(molar > 0, molar, np.nan))
    return X, converted


@step
def sample(
    X: List[Any],
    y: Optional[npt.NDArray[Any]] = None,
    n: Optional[int] = None,
    fraction: Optional[float] = None,
    random_state: Optional[int] = None,
) -> _Payload:
    """Take a random subset.

    Parameters
    ----------
    X : list of Mol
    y : ndarray, optional
    n : int, optional
        Number of molecules to keep. Mutually exclusive with ``fraction``.
    fraction : float, optional
        Fraction of the set to keep, in (0, 1].
    random_state : int, optional
        Seed.

    Returns
    -------
    (list of Mol, ndarray or None)

    Examples
    --------
    >>> from qsarkit.functional import molecules, sample
    >>> mols, _ = molecules(["CCO", "CCN", "CCC"]) >> sample(n=2, random_state=0)
    >>> len(mols)
    2
    """
    if (n is None) == (fraction is None):
        raise ValueError("sample() takes exactly one of `n` or `fraction`.")
    if fraction is not None:
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}.")
        n = max(1, int(round(len(X) * fraction)))
    assert n is not None
    if n > len(X):
        raise ValueError(f"Cannot sample {n} from a set of {len(X)}.")

    rng = np.random.RandomState(random_state)
    keep = sorted(int(i) for i in rng.choice(len(X), size=n, replace=False))
    return _subset(X, y, keep)


@step
def shuffle(
    X: List[Any],
    y: Optional[npt.NDArray[Any]] = None,
    random_state: Optional[int] = None,
) -> _Payload:
    """Shuffle the set, keeping molecules and labels aligned.

    Parameters
    ----------
    X : list of Mol
    y : ndarray, optional
    random_state : int, optional
        Seed.

    Returns
    -------
    (list of Mol, ndarray or None)

    Examples
    --------
    >>> from qsarkit.functional import molecules, shuffle
    >>> mols, y = molecules(["CCO", "CCN"], [1.0, 2.0]) >> shuffle(random_state=0)
    >>> len(mols)
    2
    """
    rng = np.random.RandomState(random_state)
    order = rng.permutation(len(X))
    return _subset(X, y, [int(i) for i in order])


@step
def apply(
    X: List[Any],
    y: Optional[npt.NDArray[Any]] = None,
    func: Optional[Callable[[Any], Any]] = None,
) -> _Payload:
    """Apply an arbitrary per-molecule function — the escape hatch.

    Parameters
    ----------
    X : list of Mol
    y : ndarray, optional
    func : callable
        ``f(mol) -> mol``. ``None`` molecules are passed through
        untouched.

    Returns
    -------
    (list of Mol, ndarray or None)

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.functional import apply, molecules
    >>> mols, _ = molecules(["CCO"]) >> apply(func=Chem.AddHs)
    >>> mols[0].GetNumAtoms()
    9
    """
    if func is None:
        raise ValueError("apply() requires a `func`.")
    return [None if m is None else func(m) for m in X], y
