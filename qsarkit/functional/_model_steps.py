"""Modelling steps for the functional pipe API.

Where :mod:`qsarkit.functional._steps` curates molecules, these steps
carry a pipeline through the rest of the QSAR workflow: featurization,
scaling, feature selection, splitting, model fitting, cross-validation
and applicability domain.

The point is that one notation reaches the whole package. Any qsarkit (or
scikit-learn) transformer can be dropped into :func:`featurize`, any
selector into :func:`select_features`, any estimator into :func:`fit` --
so the pipe is a way of *writing* the workflow, not a reimplementation of
it, and nothing in the library is out of its reach.

Examples
--------
>>> from qsarkit.functional import *
>>> from qsarkit.representation import MorganFingerprint
>>> result = (
...     molecules(DEMO_SMILES, DEMO_Y)             # doctest: +SKIP
...     >> standardize()
...     >> drop_invalid()
...     >> remove_duplicates(agg="mean")
...     >> featurize(MorganFingerprint(n_bits=512))
...     >> select_features(k=64)
...     >> fit("rf")
... )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from qsarkit.functional._core import (
    FeatureSet,
    MoleculeSet,
    PipeStep,
    feature_step,
)

if TYPE_CHECKING:  # pragma: no cover
    from sklearn.base import BaseEstimator

__all__ = [
    "featurize",
    "describe",
    "fingerprint",
    "scale",
    "impute",
    "select_features",
    "drop_correlated",
    "drop_constant",
    "split",
    "fit",
    "cross_validate",
    "applicability_domain",
    "collect",
]

_FeaturePayload = Tuple[
    "npt.NDArray[Any]", Optional["npt.NDArray[Any]"], Optional[List[Any]]
]


def _feature_names(transformer: Any, n_columns: int) -> Optional[List[str]]:
    """Column names from a transformer, or ``None`` if it does not provide them."""
    getter = getattr(transformer, "get_feature_names_out", None)
    if getter is None:
        return None
    try:
        names = [str(n) for n in getter()]
    except Exception:
        return None
    return names if len(names) == n_columns else None


class _Featurize(PipeStep):
    """The transition step: molecules in, feature matrix out.

    Wraps any transformer with a scikit-learn ``fit``/``transform``
    interface -- every class in :mod:`qsarkit.representation`, and
    equally a plain scikit-learn transformer or a
    :class:`sklearn.pipeline.Pipeline` -- so the pipe inherits the whole
    representation catalogue instead of duplicating it.

    Parameters
    ----------
    transformer : object
        Anything implementing ``transform(mols)``. ``fit`` is called
        first when the transformer has one, so learned representations
        (Mol2Vec, a fitted vocabulary) work unchanged.
    keep_mols : bool, default True
        Keep the molecules alongside the matrix, so a later step can
        still reach the chemistry (an applicability domain on Tanimoto
        distance, a scaffold split, an atom-level explanation).
    """

    __slots__ = ("transformer", "keep_mols")

    def __init__(self, transformer: Any, keep_mols: bool = True) -> None:
        super().__init__(name="featurize", params={"transformer": transformer})
        self.transformer = transformer
        self.keep_mols = keep_mols

    def _describe(self, n_before: int = 0) -> str:
        return f"featurize({type(self.transformer).__name__})"

    def __call__(self, data: Any) -> FeatureSet:
        """Featurize a :class:`MoleculeSet` (or a raw molecule list)."""
        if isinstance(data, FeatureSet):
            raise TypeError(
                "featurize() works on molecules, but it received a "
                "FeatureSet -- the pipeline has already been featurized."
            )
        molecule_set = data if isinstance(data, MoleculeSet) else MoleculeSet(data)

        if any(m is None for m in molecule_set.mols):
            n_bad = sum(1 for m in molecule_set.mols if m is None)
            raise ValueError(
                f"Cannot featurize: {n_bad} of {len(molecule_set)} entries are "
                "None (they failed parsing or curation). Add a drop_invalid() "
                "step before featurize() to remove them, which drops the "
                "matching labels too."
            )

        transformer = self.transformer
        if hasattr(transformer, "fit"):
            transformer.fit(molecule_set.mols, molecule_set.y)
        X = np.asarray(transformer.transform(molecule_set.mols))

        return FeatureSet(
            X,
            molecule_set.y,
            molecule_set.mols if self.keep_mols else None,
            history=[*molecule_set.history, self._describe()],
            feature_names=_feature_names(transformer, X.shape[1]),
        )


def featurize(transformer: Any, keep_mols: bool = True) -> PipeStep:
    """Turn molecules into a feature matrix, crossing into the modelling half.

    This is the hinge of the functional API. Everything before it works
    on molecules; everything after it works on a matrix.

    Parameters
    ----------
    transformer : object
        Any transformer with ``transform(mols)`` -- typically one from
        :mod:`qsarkit.representation`, but a scikit-learn
        :class:`~sklearn.pipeline.Pipeline` or
        :class:`~sklearn.pipeline.FeatureUnion` of them works too.
    keep_mols : bool, default True
        Keep the molecules alongside the matrix for later steps.

    Returns
    -------
    PipeStep
        A step producing a :class:`~qsarkit.functional.FeatureSet`.

    Examples
    --------
    >>> from qsarkit.functional import featurize, molecules
    >>> from qsarkit.representation import MorganFingerprint
    >>> fs = molecules(["CCO", "c1ccccc1"], [1.0, 2.0]) >> featurize(
    ...     MorganFingerprint(n_bits=128))
    >>> fs.shape
    (2, 128)

    Because it takes any transformer, combining representations needs no
    new syntax:

    >>> from qsarkit.representation import FingerprintCombiner, MACCSKeysFingerprint
    >>> combined = FingerprintCombiner([
    ...     ("morgan", MorganFingerprint(n_bits=128)),
    ...     ("maccs", MACCSKeysFingerprint()),
    ... ])
    >>> (molecules(["CCO", "c1ccccc1"]) >> featurize(combined)).shape
    (2, 295)

    Invalid molecules are refused rather than silently producing junk
    rows, because a ``None`` here would break the alignment with ``y``
    that the rest of the pipe maintains:

    >>> molecules(["CCO", "not-a-molecule"]) >> featurize(MorganFingerprint())
    Traceback (most recent call last):
        ...
    ValueError: Cannot featurize: 1 of 2 entries are None...

    References
    ----------
    - Pedregosa et al. (2011). "Scikit-learn: Machine Learning in
      Python." JMLR, 12, 2825-2830.
      https://jmlr.org/papers/v12/pedregosa11a.html
    """
    return _Featurize(transformer, keep_mols=keep_mols)


def fingerprint(
    kind: str = "morgan", keep_mols: bool = True, **kwargs: Any
) -> PipeStep:
    """Featurize with a named fingerprint, for the common case.

    A shorthand for ``featurize(MorganFingerprint(...))`` and friends,
    so a quick pipeline does not need a second import.

    Parameters
    ----------
    kind : {"morgan", "ecfp", "fcfp", "rdkit", "maccs", "atom_pair", \
"topological_torsion", "avalon", "pattern", "layered"}, default "morgan"
        Which fingerprint to compute. ``"ecfp"`` is an alias for
        ``"morgan"``; ``"fcfp"`` selects the feature-based variant.
    keep_mols : bool, default True
        Keep the molecules alongside the matrix, as for :func:`featurize`.
    **kwargs
        Passed to the underlying transformer (``n_bits``, ``radius``, ...).

    Returns
    -------
    PipeStep
        A step producing a :class:`~qsarkit.functional.FeatureSet`.

    Examples
    --------
    >>> from qsarkit.functional import fingerprint, molecules
    >>> (molecules(["CCO", "c1ccccc1"]) >> fingerprint("morgan", n_bits=64)).shape
    (2, 64)
    >>> (molecules(["CCO"]) >> fingerprint("maccs")).shape
    (1, 167)

    References
    ----------
    - Rogers, D. & Hahn, M. (2010). "Extended-Connectivity Fingerprints."
      J. Chem. Inf. Model., 50(5), 742-754.
      https://doi.org/10.1021/ci100050t
    """
    from qsarkit import representation as rep

    registry: Dict[str, Any] = {
        "morgan": rep.MorganFingerprint,
        "ecfp": rep.MorganFingerprint,
        "fcfp": rep.FeatureMorganFingerprint,
        "rdkit": rep.RDKitFingerprint,
        "maccs": rep.MACCSKeysFingerprint,
        "atom_pair": rep.AtomPairFingerprint,
        "topological_torsion": rep.TopologicalTorsionFingerprint,
        "avalon": rep.AvalonFingerprint,
        "pattern": rep.PatternFingerprint,
        "layered": rep.LayeredFingerprint,
    }
    if kind not in registry:
        raise ValueError(
            f"Unknown fingerprint {kind!r}. Choose from {sorted(registry)}."
        )
    return _Featurize(registry[kind](**kwargs), keep_mols=keep_mols)


def describe(
    kind: str = "physicochemical", keep_mols: bool = True, **kwargs: Any
) -> PipeStep:
    """Featurize with a named descriptor block.

    Parameters
    ----------
    kind : {"physicochemical", "rdkit", "constitutional", "lipinski", \
"fragment", "3d"}, default "physicochemical"
        Which descriptor set to compute.
    keep_mols : bool, default True
        Keep the molecules alongside the matrix, as for :func:`featurize`.
    **kwargs
        Passed to the underlying transformer.

    Returns
    -------
    PipeStep
        A step producing a :class:`~qsarkit.functional.FeatureSet`.

    Examples
    --------
    >>> from qsarkit.functional import describe, molecules
    >>> fs = molecules(["CCO", "c1ccccc1"]) >> describe("lipinski")
    >>> fs.feature_names is not None
    True

    Descriptors are continuous and on wildly different scales (molecular
    weight in the hundreds, logP in single digits), so they almost always
    want a :func:`scale` step before a distance-based model:

    >>> from qsarkit.functional import scale
    >>> (molecules(["CCO", "c1ccccc1", "CCN"]) >> describe() >> scale()).shape
    (3, 9)

    References
    ----------
    - Todeschini, R. & Consonni, V. (2009). "Molecular Descriptors for
      Chemoinformatics." Wiley. https://doi.org/10.1002/9783527628766
    """
    from qsarkit import representation as rep

    registry: Dict[str, Any] = {
        "physicochemical": rep.PhysicochemicalDescriptors,
        "rdkit": rep.RDKitDescriptors,
        "constitutional": rep.ConstitutionalDescriptors,
        "lipinski": rep.LipinskiDescriptors,
        "fragment": rep.FragmentDescriptors,
        "3d": rep.Descriptors3D,
    }
    if kind not in registry:
        raise ValueError(
            f"Unknown descriptor set {kind!r}. Choose from {sorted(registry)}."
        )
    return _Featurize(registry[kind](**kwargs), keep_mols=keep_mols)


# ---------------------------------------------------------------- matrix ----


@feature_step
def scale(
    X: "npt.NDArray[Any]",
    y: Optional["npt.NDArray[Any]"] = None,
    mols: Optional[List[Any]] = None,
    method: Literal["standard", "minmax", "robust", "none"] = "standard",
) -> _FeaturePayload:
    """Scale the feature matrix.

    Parameters
    ----------
    X : ndarray
        Feature matrix.
    y : ndarray, optional
        Labels, passed through untouched.
    mols : list of Mol, optional
        Molecules, passed through untouched.
    method : {"standard", "minmax", "robust", "none"}, default "standard"
        ``"standard"`` centres and scales to unit variance,
        ``"robust"`` uses the median and IQR (resistant to the outliers
        that descriptor blocks routinely contain), ``"minmax"`` maps onto
        [0, 1], and ``"none"`` is a no-op for parametrized pipelines.

    Returns
    -------
    tuple
        ``(X, y, mols)``.

    Notes
    -----
    Scaling inside a pipe like this fits on whatever data is flowing
    through it. That is correct for a single curated dataset, but if you
    are holding out a test set, scale *after* :func:`split` or inside a
    :class:`sklearn.pipeline.Pipeline` given to :func:`cross_validate` --
    otherwise the test set's statistics leak into the transform.

    Examples
    --------
    >>> from qsarkit.functional import describe, molecules, scale
    >>> fs = molecules(["CCO", "c1ccccc1", "CCN"]) >> describe() >> scale("robust")
    >>> fs.shape
    (3, 9)

    References
    ----------
    - Pedregosa et al. (2011). "Scikit-learn: Machine Learning in
      Python." JMLR, 12, 2825-2830.
      https://jmlr.org/papers/v12/pedregosa11a.html
    """
    if method == "none":
        return X, y, mols
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }
    if method not in scalers:
        raise ValueError(
            f"Unknown scaling method {method!r}. "
            f"Choose from {sorted(scalers)} or 'none'."
        )
    return np.asarray(scalers[method]().fit_transform(X)), y, mols


@feature_step
def impute(
    X: "npt.NDArray[Any]",
    y: Optional["npt.NDArray[Any]"] = None,
    mols: Optional[List[Any]] = None,
    strategy: Literal["mean", "median", "most_frequent", "zero", "drop"] = "median",
) -> _FeaturePayload:
    """Fill or remove non-finite values in the feature matrix.

    Descriptor calculators emit NaN for undefined quantities (a 3D
    descriptor on a molecule with no conformer, a ratio with a zero
    denominator), and most estimators refuse to fit on them.

    Parameters
    ----------
    X : ndarray
        Feature matrix.
    y : ndarray, optional
        Labels; subset alongside ``X`` when ``strategy="drop"``.
    mols : list of Mol, optional
        Molecules; subset alongside ``X`` when ``strategy="drop"``.
    strategy : {"mean", "median", "most_frequent", "zero", "drop"}, default "median"
        How to handle them. ``"drop"`` removes offending *rows* (and the
        matching labels and molecules); the others fill column-wise.

    Returns
    -------
    tuple
        ``(X, y, mols)``.

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.functional import impute
    >>> X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    >>> filled, _, _ = impute(X, strategy="median")
    >>> float(filled[0, 1])
    5.0

    Dropping instead keeps labels aligned with the surviving rows:

    >>> kept, y, _ = impute(X, np.array([1.0, 2.0, 3.0]), strategy="drop")
    >>> kept.shape, y.tolist()
    ((2, 2), [2.0, 3.0])

    References
    ----------
    - Pedregosa et al. (2011). "Scikit-learn: Machine Learning in
      Python." JMLR, 12, 2825-2830.
      https://jmlr.org/papers/v12/pedregosa11a.html
    """
    X = np.asarray(X, dtype=float)
    finite = np.isfinite(X)
    if finite.all():
        return X, y, mols

    if strategy == "drop":
        keep = np.flatnonzero(finite.all(axis=1))
        return (
            X[keep],
            None if y is None else y[keep],
            None if mols is None else [mols[i] for i in keep],
        )

    if strategy == "zero":
        return np.where(finite, X, 0.0), y, mols

    from sklearn.impute import SimpleImputer

    if strategy not in {"mean", "median", "most_frequent"}:
        raise ValueError(
            f"Unknown imputation strategy {strategy!r}. Choose from "
            "'mean', 'median', 'most_frequent', 'zero' or 'drop'."
        )
    cleaned = np.where(finite, X, np.nan)
    imputed = SimpleImputer(strategy=strategy).fit_transform(cleaned)
    # SimpleImputer drops all-NaN columns; restore them as zeros so the
    # matrix keeps its width and feature names stay meaningful.
    if imputed.shape[1] != X.shape[1]:
        out = np.zeros_like(X)
        kept = np.flatnonzero(~np.isnan(cleaned).all(axis=0))
        out[:, kept] = imputed
        imputed = out
    return np.asarray(imputed), y, mols


@feature_step
def drop_constant(
    X: "npt.NDArray[Any]",
    y: Optional["npt.NDArray[Any]"] = None,
    mols: Optional[List[Any]] = None,
    threshold: float = 0.0,
) -> _FeaturePayload:
    """Remove features whose variance is at or below ``threshold``.

    A bit that is set in every molecule, or in none, cannot separate
    them. Fingerprint blocks are mostly this: a 2048-bit Morgan
    fingerprint over a few hundred compounds typically has fewer than
    300 columns that vary at all.

    Parameters
    ----------
    X : ndarray
        Feature matrix.
    y : ndarray, optional
        Labels, passed through untouched.
    mols : list of Mol, optional
        Molecules, passed through untouched.
    threshold : float, default 0.0
        Variance at or below which a column is dropped.

    Returns
    -------
    tuple
        ``(X, y, mols)``.

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.functional import drop_constant
    >>> X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    >>> reduced, _, _ = drop_constant(X)
    >>> reduced.shape
    (3, 1)

    References
    ----------
    - Pedregosa et al. (2011). "Scikit-learn: Machine Learning in
      Python." JMLR, 12, 2825-2830.
      https://jmlr.org/papers/v12/pedregosa11a.html
    """
    from qsarkit.feature_selection import VarianceFilter

    selector = VarianceFilter(threshold=threshold)
    return np.asarray(selector.fit_transform(X)), y, mols


@feature_step
def drop_correlated(
    X: "npt.NDArray[Any]",
    y: Optional["npt.NDArray[Any]"] = None,
    mols: Optional[List[Any]] = None,
    threshold: float = 0.95,
    method: Literal["pearson", "spearman"] = "pearson",
) -> _FeaturePayload:
    """Remove one of every pair of features correlated above ``threshold``.

    Parameters
    ----------
    X : ndarray
        Feature matrix.
    y : ndarray, optional
        Labels, passed through untouched.
    mols : list of Mol, optional
        Molecules, passed through untouched.
    threshold : float, default 0.95
        Absolute correlation above which one of the pair is dropped.
    method : {"pearson", "spearman"}, default "pearson"
        Correlation coefficient to use.

    Returns
    -------
    tuple
        ``(X, y, mols)``.

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.functional import drop_correlated
    >>> X = np.array([[1.0, 2.0, 9.0], [2.0, 4.0, 1.0], [3.0, 6.0, 5.0]])
    >>> reduced, _, _ = drop_correlated(X, threshold=0.99)
    >>> reduced.shape                # columns 0 and 1 are perfectly correlated
    (3, 2)

    References
    ----------
    - Todeschini, R. & Consonni, V. (2009). "Molecular Descriptors for
      Chemoinformatics." Wiley. https://doi.org/10.1002/9783527628766
    """
    from qsarkit.feature_selection import CorrelationFilter

    selector = CorrelationFilter(threshold=threshold, method=method)
    return np.asarray(selector.fit_transform(X, y)), y, mols


@feature_step
def select_features(
    X: "npt.NDArray[Any]",
    y: Optional["npt.NDArray[Any]"] = None,
    mols: Optional[List[Any]] = None,
    method: Literal["mutual_info", "rfe", "boruta", "variance"] = "mutual_info",
    k: int = 20,
    task: Literal["regression", "classification"] = "regression",
    estimator: Optional[Any] = None,
    **kwargs: Any,
) -> _FeaturePayload:
    """Select the ``k`` most informative features.

    Parameters
    ----------
    X : ndarray
        Feature matrix.
    y : ndarray
        Labels. Required for every method except ``"variance"``.
    mols : list of Mol, optional
        Molecules, passed through untouched.
    method : {"mutual_info", "rfe", "boruta", "variance"}, default "mutual_info"
        Selection strategy, from :mod:`qsarkit.feature_selection`.
    k : int, default 20
        Number of features to keep. Ignored by ``"variance"``, and by
        ``"boruta"``, which determines the count itself.
    task : {"regression", "classification"}, default "regression"
        Whether ``y`` is continuous or categorical. Chooses the
        underlying scoring function.
    estimator : object, optional
        Base estimator for ``"rfe"`` and ``"boruta"``.
    **kwargs
        Passed to the underlying selector.

    Returns
    -------
    tuple
        ``(X, y, mols)``.

    Notes
    -----
    Selecting features on the full dataset and *then* splitting is
    selection bias: the choice of columns has already seen the test
    labels, and the held-out score is optimistic. Put this step after
    :func:`split`, or inside a pipeline handed to :func:`cross_validate`.

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.functional import select_features
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(30, 10))
    >>> y = X[:, 0] * 2 + rng.normal(scale=0.1, size=30)
    >>> reduced, _, _ = select_features(X, y, k=3)
    >>> reduced.shape
    (30, 3)

    References
    ----------
    - Kursa, M. B. & Rudnicki, W. R. (2010). "Feature Selection with the
      Boruta Package." J. Stat. Softw., 36(11), 1-13.
      https://doi.org/10.18637/jss.v036.i11
    - Guyon, I. et al. (2002). "Gene Selection for Cancer Classification
      using Support Vector Machines." Mach. Learn., 46, 389-422.
      https://doi.org/10.1023/A:1012487302797
    """
    from qsarkit import feature_selection as fs

    if method != "variance" and y is None:
        raise ValueError(
            f"select_features(method={method!r}) needs labels. Either supply y "
            "to molecules(), or use method='variance', which is unsupervised."
        )

    if method == "variance":
        selector: Any = fs.VarianceFilter(**kwargs)
    elif method == "mutual_info":
        selector = fs.MutualInformationSelector(task=task, k=k, **kwargs)
    elif method == "rfe":
        selector = fs.RFESelector(
            estimator=estimator, task=task, n_features_to_select=k, **kwargs
        )
    elif method == "boruta":
        selector = fs.BorutaSelector(estimator=estimator, task=task, **kwargs)
    else:
        raise ValueError(
            f"Unknown selection method {method!r}. Choose from "
            "'mutual_info', 'rfe', 'boruta' or 'variance'."
        )
    return np.asarray(selector.fit_transform(X, y)), y, mols


# ------------------------------------------------------------- terminals ----


class _Terminal(PipeStep):
    """Base for steps that end a pipe, returning a result rather than a set.

    Splitting, fitting, cross-validating and defining an applicability
    domain all consume a :class:`FeatureSet` and produce something that
    is not one. They are still steps -- they compose, they appear in the
    flowchart -- but a pipe ending in one stops there.
    """

    __slots__ = ()

    def _require_features(self, data: Any) -> FeatureSet:
        if isinstance(data, FeatureSet):
            return data
        what = "molecules" if isinstance(data, MoleculeSet) else type(data).__name__
        raise TypeError(
            f"Step '{self.name}' works on a feature matrix, but it received "
            f"{what}. Insert a featurize(...) step first."
        )


class _Split(_Terminal):
    """Split a :class:`FeatureSet` into train and test halves."""

    __slots__ = ("splitter", "method", "test_size", "random_state", "kwargs")

    def __init__(
        self,
        method: str = "scaffold",
        test_size: float = 0.2,
        random_state: Optional[int] = None,
        splitter: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="split",
            params={"method": method, "test_size": test_size},
        )
        self.method = method
        self.test_size = test_size
        self.random_state = random_state
        self.splitter = splitter
        self.kwargs = kwargs

    def _build(self) -> Any:
        if self.splitter is not None:
            return self.splitter
        from qsarkit import model_selection as ms

        registry: Dict[str, Any] = {
            "scaffold": ms.ScaffoldSplitter,
            "stratified_scaffold": ms.StratifiedScaffoldSplitter,
            "random": ms.RandomSplitter,
            "butina": ms.ButinaClusterSplitter,
            "sphere_exclusion": ms.SphereExclusionSplitter,
            "maxmin": ms.MaxMinSplitter,
            "kennard_stone": ms.KennardStoneSplitter,
            "perimeter": ms.PerimeterSplitter,
            "time": ms.TimeSplitter,
        }
        if self.method not in registry:
            raise ValueError(
                f"Unknown split method {self.method!r}. "
                f"Choose from {sorted(registry)}."
            )
        return registry[self.method](
            test_size=self.test_size,
            random_state=self.random_state,
            **self.kwargs,
        )

    def __call__(self, data: Any) -> Tuple[FeatureSet, FeatureSet]:
        """Return ``(train, test)`` as two :class:`FeatureSet` objects."""
        features = self._require_features(data)
        splitter = self._build()

        # The chemistry-aware splitters need molecules, not just the matrix.
        needs_mols = self.method in {
            "scaffold", "stratified_scaffold", "butina", "sphere_exclusion",
        }
        if needs_mols:
            if features.mols is None:
                raise ValueError(
                    f"split(method={self.method!r}) needs the molecules, but "
                    "they were discarded by featurize(keep_mols=False)."
                )
            train_idx, test_idx = next(
                splitter.split_mols(features.mols, features.y)
            )
        else:
            train_idx, test_idx = next(
                splitter.split(features.X, features.y, groups=features.mols)
            )

        def _take(idx: "npt.NDArray[Any]", label: str) -> FeatureSet:
            idx = np.asarray(idx, dtype=int)
            return FeatureSet(
                features.X[idx],
                None if features.y is None else features.y[idx],
                None if features.mols is None else [features.mols[i] for i in idx],
                history=[*features.history, f"{self._describe()} -> {label}"],
                feature_names=features.feature_names,
            )

        return _take(train_idx, "train"), _take(test_idx, "test")


def split(
    method: str = "scaffold",
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    splitter: Optional[Any] = None,
    **kwargs: Any,
) -> PipeStep:
    """Split into train and test sets, unpacking as ``train, test``.

    Parameters
    ----------
    method : str, default "scaffold"
        One of ``"scaffold"``, ``"stratified_scaffold"``, ``"random"``,
        ``"butina"``, ``"sphere_exclusion"``, ``"maxmin"``,
        ``"kennard_stone"``, ``"perimeter"`` or ``"time"``. Ignored when
        ``splitter`` is given.
    test_size : float, default 0.2
        Fraction held out.
    random_state : int, optional
        Seed, where the splitter uses one.
    splitter : object, optional
        A splitter instance to use instead of building one from
        ``method`` -- any of :mod:`qsarkit.model_selection`, or a
        scikit-learn splitter.
    **kwargs
        Passed to the splitter's constructor.

    Returns
    -------
    PipeStep
        A step returning ``(train, test)``, each a
        :class:`~qsarkit.functional.FeatureSet`.

    Notes
    -----
    The default is a scaffold split, not a random one, because a random
    split of a QSAR dataset measures interpolation: public datasets are
    dense with near-duplicate analogues, so random assignment scatters a
    congeneric series across both sides and the model is scored on
    compounds whose close relatives it has memorized. A scaffold split
    keeps whole series together and reports what you actually want to
    know.

    Examples
    --------
    >>> from qsarkit.functional import fingerprint, molecules, split
    >>> smiles = ["c1ccccc1C", "c1ccccc1CC", "c1ccncc1C", "CCO", "CCN"]
    >>> train, test = (
    ...     molecules(smiles, [1.0, 2.0, 3.0, 4.0, 5.0])
    ...     >> fingerprint(n_bits=64)
    ...     >> split(test_size=0.4)
    ... )
    >>> len(train) + len(test)
    5

    References
    ----------
    - Bemis, G. W. & Murcko, M. A. (1996). "The Properties of Known
      Drugs. 1. Molecular Frameworks." J. Med. Chem., 39(15), 2887-2893.
      https://doi.org/10.1021/jm9602928
    - Sheridan, R. P. (2013). "Time-Split Cross-Validation as a Method
      for Estimating the Goodness of Prospective Prediction."
      J. Chem. Inf. Model., 53(4), 783-790.
      https://doi.org/10.1021/ci400084k
    """
    return _Split(
        method=method,
        test_size=test_size,
        random_state=random_state,
        splitter=splitter,
        **kwargs,
    )


class _Fit(_Terminal):
    """Fit an estimator on a :class:`FeatureSet`."""

    __slots__ = ("estimator", "task", "kwargs")

    def __init__(
        self,
        estimator: Any = "rf",
        task: Literal["regression", "classification", "auto"] = "auto",
        **kwargs: Any,
    ) -> None:
        label = estimator if isinstance(estimator, str) else type(estimator).__name__
        super().__init__(name="fit", params={"estimator": label})
        self.estimator = estimator
        self.task = task
        self.kwargs = kwargs

    def _build(self, y: Optional["npt.NDArray[Any]"]) -> Any:
        if not isinstance(self.estimator, str):
            return self.estimator
        from qsarkit.models import QSARClassifier, QSARRegressor

        task = self.task
        if task == "auto":
            if y is None:
                raise ValueError(
                    "fit() needs labels to choose between regression and "
                    "classification. Supply y to molecules(), or pass an "
                    "estimator instance."
                )
            # Integer-like labels with few distinct values are categories;
            # anything else is treated as a continuous endpoint.
            distinct = np.unique(y)
            looks_categorical = (
                distinct.size <= max(2, int(np.sqrt(len(y))))
                and np.allclose(distinct, distinct.astype(int))
            )
            task = "classification" if looks_categorical else "regression"
        cls = QSARClassifier if task == "classification" else QSARRegressor
        return cls(self.estimator, **self.kwargs)

    def __call__(self, data: Any) -> Any:
        """Fit and return the estimator."""
        features = self._require_features(data)
        if features.y is None:
            raise ValueError("fit() needs labels; none were supplied to molecules().")
        model = self._build(features.y)
        model.fit(features.X, features.y)
        return model


def fit(
    estimator: Any = "rf",
    task: Literal["regression", "classification", "auto"] = "auto",
    **kwargs: Any,
) -> PipeStep:
    """Fit a model on the features, ending the pipe with a fitted estimator.

    Parameters
    ----------
    estimator : str or object, default "rf"
        A backend name for :class:`~qsarkit.models.QSARRegressor` /
        :class:`~qsarkit.models.QSARClassifier` (``"rf"``, ``"svm"``,
        ``"gbm"``, ``"xgboost"``, ``"lightgbm"``, ``"knn"``, ``"pls"``,
        ``"gp"``, ``"mlp"``, ...), or any estimator instance -- including
        a plain scikit-learn one.
    task : {"auto", "regression", "classification"}, default "auto"
        Which facade to build when ``estimator`` is a name. ``"auto"``
        infers it from ``y``: few distinct integer labels means
        classification, anything else regression.
    **kwargs
        Passed to the facade's constructor (``random_state``,
        ``model_params``).

    Returns
    -------
    PipeStep
        A step returning the fitted estimator.

    Examples
    --------
    >>> from qsarkit.functional import fingerprint, fit, molecules
    >>> smiles = ["CCO", "CCN", "CCC", "CCCl", "c1ccccc1", "c1ccncc1"]
    >>> model = (
    ...     molecules(smiles, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ...     >> fingerprint(n_bits=64)
    ...     >> fit("rf", random_state=0)
    ... )
    >>> model.predict(fingerprint(n_bits=64)(molecules(["CCO"])).X).shape
    (1,)

    Any estimator instance works, so the pipe is not limited to the
    facades:

    >>> from sklearn.linear_model import Ridge
    >>> model = (
    ...     molecules(smiles, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ...     >> fingerprint(n_bits=64)
    ...     >> fit(Ridge())
    ... )
    >>> type(model).__name__
    'Ridge'

    References
    ----------
    - Pedregosa et al. (2011). "Scikit-learn: Machine Learning in
      Python." JMLR, 12, 2825-2830.
      https://jmlr.org/papers/v12/pedregosa11a.html
    """
    return _Fit(estimator=estimator, task=task, **kwargs)


class _CrossValidate(_Terminal):
    """Cross-validate an estimator on a :class:`FeatureSet`."""

    __slots__ = ("estimator", "task", "validator_kwargs")

    def __init__(
        self,
        estimator: Any = "rf",
        task: Literal["regression", "classification", "auto"] = "auto",
        **validator_kwargs: Any,
    ) -> None:
        label = estimator if isinstance(estimator, str) else type(estimator).__name__
        super().__init__(name="cross_validate", params={"estimator": label})
        self.estimator = estimator
        self.task = task
        self.validator_kwargs = validator_kwargs

    def __call__(self, data: Any) -> Dict[str, Any]:
        """Run cross-validation and return the report."""
        features = self._require_features(data)
        if features.y is None:
            raise ValueError(
                "cross_validate() needs labels; none were supplied to molecules()."
            )
        from qsarkit.validation import CrossValidator

        model = _Fit(self.estimator, self.task)._build(features.y)
        validator = CrossValidator(**self.validator_kwargs)
        return validator.evaluate(model, features.X, features.y)


def cross_validate(
    estimator: Any = "rf",
    task: Literal["regression", "classification", "auto"] = "auto",
    **kwargs: Any,
) -> PipeStep:
    """Cross-validate on the features, ending the pipe with a score report.

    Parameters
    ----------
    estimator : str or object, default "rf"
        As for :func:`fit`.
    task : {"auto", "regression", "classification"}, default "auto"
        As for :func:`fit`.
    **kwargs
        Passed to :class:`~qsarkit.validation.CrossValidator`
        (``method``, ``n_splits``, ``random_state``).

    Returns
    -------
    PipeStep
        A step returning the cross-validation report as a dict.

    Examples
    --------
    >>> from qsarkit.functional import cross_validate, fingerprint, molecules
    >>> smiles = ["CCO", "CCN", "CCC", "CCCl", "c1ccccc1", "c1ccncc1"]
    >>> report = (
    ...     molecules(smiles, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ...     >> fingerprint(n_bits=64)
    ...     >> cross_validate("rf", n_splits=3, random_state=0)
    ... )
    >>> "q2" in report or "r2" in report
    True

    References
    ----------
    - OECD (2007). "Guidance Document on the Validation of (Quantitative)
      Structure-Activity Relationship [(Q)SAR] Models,"
      ENV/JM/MONO(2007)2. https://doi.org/10.1787/9789264085442-en
    """
    return _CrossValidate(estimator=estimator, task=task, **kwargs)


class _ApplicabilityDomain(_Terminal):
    """Fit an applicability domain on a :class:`FeatureSet`."""

    __slots__ = ("method", "kwargs")

    def __init__(self, method: str = "knn", **kwargs: Any) -> None:
        super().__init__(name="applicability_domain", params={"method": method})
        self.method = method
        self.kwargs = kwargs

    def __call__(self, data: Any) -> Any:
        """Fit and return the domain."""
        features = self._require_features(data)
        from qsarkit import applicability as ad

        registry: Dict[str, Any] = {
            "knn": ad.KNNApplicabilityDomain,
            "leverage": ad.LeverageAD,
            "range": ad.RangeAD,
            "bounding_box": ad.BoundingBoxAD,
            "pca": ad.PCABoundingBoxAD,
            "convex_hull": ad.ConvexHullAD,
            "tanimoto": ad.TanimotoSimilarityAD,
            "kde": ad.KernelDensityAD,
            "isolation_forest": ad.IsolationForestAD,
            "ensemble": ad.EnsembleAD,
        }
        if self.method not in registry:
            raise ValueError(
                f"Unknown applicability domain {self.method!r}. "
                f"Choose from {sorted(registry)}."
            )
        return registry[self.method](**self.kwargs).fit(features.X, features.y)


def applicability_domain(method: str = "knn", **kwargs: Any) -> PipeStep:
    """Fit an applicability domain on the features, ending the pipe.

    Parameters
    ----------
    method : str, default "knn"
        One of ``"knn"``, ``"leverage"``, ``"range"``, ``"bounding_box"``,
        ``"pca"``, ``"convex_hull"``, ``"tanimoto"``, ``"kde"``,
        ``"isolation_forest"`` or ``"ensemble"``.
    **kwargs
        Passed to the domain's constructor.

    Returns
    -------
    PipeStep
        A step returning the fitted domain.

    Examples
    --------
    >>> from qsarkit.functional import applicability_domain, fingerprint, molecules
    >>> smiles = ["CCO", "CCN", "CCC", "CCCl", "c1ccccc1", "c1ccncc1"]
    >>> domain = (
    ...     molecules(smiles)
    ...     >> fingerprint(n_bits=64)
    ...     >> applicability_domain("tanimoto", threshold=0.3)
    ... )
    >>> domain.predict(fingerprint(n_bits=64)(molecules(["CCO"])).X).tolist()
    [True]

    References
    ----------
    - Sahigara, F. et al. (2012). "Comparison of Different Approaches to
      Define the Applicability Domain of QSAR Models." Molecules, 17(5),
      4791-4810. https://doi.org/10.3390/molecules17054791
    """
    return _ApplicabilityDomain(method=method, **kwargs)


class _Collect(_Terminal):
    """Return the flowing value itself, ending a pipe explicitly."""

    __slots__ = ("as_frame",)

    def __init__(self, as_frame: bool = False) -> None:
        super().__init__(name="collect", params={"as_frame": as_frame} if as_frame else None)
        self.as_frame = as_frame

    def __call__(self, data: Any) -> Any:
        """Return the set, or its DataFrame rendering."""
        if not isinstance(data, (MoleculeSet, FeatureSet)):
            data = MoleculeSet(data)
        return data.to_frame() if self.as_frame else data


def collect(as_frame: bool = False) -> PipeStep:
    """End a pipe explicitly, returning the set or a DataFrame of it.

    Useful when a pipeline is built programmatically and you want the
    terminal stage to be a step like any other, and when you want the
    result as a table rather than as arrays.

    Parameters
    ----------
    as_frame : bool, default False
        Return ``to_frame()`` instead of the set itself.

    Returns
    -------
    PipeStep
        A step returning the value flowing into it.

    Examples
    --------
    >>> from qsarkit.functional import collect, desalt, molecules
    >>> frame = molecules(["CCO", "CC(=O)[O-].[Na+]"]) >> desalt() >> collect(as_frame=True)
    >>> list(frame.columns)
    ['smiles']
    >>> len(frame)
    2
    """
    return _Collect(as_frame=as_frame)
