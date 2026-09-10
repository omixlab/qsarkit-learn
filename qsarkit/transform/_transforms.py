"""scikit-learn glue for composing molecule transformers into pipelines."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Iterable, List, Literal, Optional, Sequence

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin

from qsarkit.base.exceptions import ModelNotFittedError

if TYPE_CHECKING:  # pragma: no cover
    from rdkit.Chem import Mol
    from sklearn.pipeline import Pipeline

__all__ = [
    "SmilesToMol",
    "MolToSmiles",
    "MoleculeFeatureUnion",
    "NaNHandler",
    "VarianceThresholdMol",
    "DescriptorScaler",
    "make_qsar_pipeline",
]


class SmilesToMol(BaseEstimator, TransformerMixin):
    """Parse SMILES strings into RDKit molecules.

    The entry point that lets a pipeline start from a raw SMILES column,
    so the whole workflow — parsing, featurizing, modeling — is one
    fitted object that can be pickled and reapplied.

    Parameters
    ----------
    sanitize : bool, default True
        Sanitize on parse. Turning this off keeps structures RDKit would
        reject, which is occasionally useful for diagnostics but unsafe
        for modeling.
    on_error : {"none", "raise"}, default "none"
        ``"none"`` yields ``None`` for unparseable input, preserving
        positional alignment with ``y``; ``"raise"`` stops at the first
        failure.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mols = SmilesToMol().transform(["CCO", "c1ccccc1"])
    >>> Chem.MolToSmiles(mols[0])
    'CCO'

    References
    ----------
    - Weininger, D. (1988). "SMILES, a Chemical Language and Information
      System." J. Chem. Inf. Comput. Sci., 28(1), 31-36.
      https://doi.org/10.1021/ci00057a005
    - RDKit documentation: https://www.rdkit.org/docs/
    """

    def __init__(
        self,
        sanitize: bool = True,
        on_error: Literal["none", "raise"] = "none",
    ) -> None:
        self.sanitize = sanitize
        self.on_error = on_error

    def fit(
        self, X: Iterable[str], y: Optional[npt.ArrayLike] = None
    ) -> "SmilesToMol":
        """No-op; parsing is stateless.

        Parameters
        ----------
        X : iterable of str
        y : ignored

        Returns
        -------
        SmilesToMol
        """
        return self

    def transform(self, X: Iterable[str]) -> List[Any]:
        """Parse each SMILES string.

        Parameters
        ----------
        X : iterable of str

        Returns
        -------
        list of Mol
            ``None`` where parsing failed and ``on_error="none"``.
        """
        from rdkit import Chem

        from qsarkit.base.exceptions import InvalidMoleculeError

        out: List[Any] = []
        for i, smiles in enumerate(X):
            mol = Chem.MolFromSmiles(smiles, sanitize=self.sanitize)
            if mol is None and self.on_error == "raise":
                raise InvalidMoleculeError(
                    f"Could not parse SMILES at position {i}: {smiles!r}"
                )
            out.append(mol)
        return out


class MolToSmiles(BaseEstimator, TransformerMixin):
    """Serialize RDKit molecules back to canonical SMILES.

    Parameters
    ----------
    isomeric : bool, default True
        Include stereochemistry.
    canonical : bool, default True
        Emit RDKit's canonical form, so identical structures produce
        identical strings.

    Examples
    --------
    >>> from rdkit import Chem
    >>> MolToSmiles().transform([Chem.MolFromSmiles("OCC")])
    ['CCO']

    References
    ----------
    - Weininger, D., Weininger, A. & Weininger, J. L. (1989). "SMILES 2.
      Algorithm for Generation of Unique SMILES Notation." J. Chem. Inf.
      Comput. Sci., 29(2), 97-101. https://doi.org/10.1021/ci00062a008
    """

    def __init__(self, isomeric: bool = True, canonical: bool = True) -> None:
        self.isomeric = isomeric
        self.canonical = canonical

    def fit(
        self, X: Iterable[Any], y: Optional[npt.ArrayLike] = None
    ) -> "MolToSmiles":
        """No-op; serialization is stateless."""
        return self

    def transform(self, X: Iterable[Any]) -> List[Optional[str]]:
        """Serialize each molecule.

        Parameters
        ----------
        X : iterable of Mol

        Returns
        -------
        list of str
            ``None`` where the input molecule was ``None``.
        """
        from rdkit import Chem

        return [
            None
            if m is None
            else str(
                Chem.MolToSmiles(
                    m, isomericSmiles=self.isomeric, canonical=self.canonical
                )
            )
            for m in X
        ]


class MoleculeFeatureUnion(BaseEstimator, TransformerMixin):
    """Concatenate several molecule transformers into one feature matrix.

    scikit-learn's own ``FeatureUnion`` cannot be used here because it
    validates ``X`` as a numeric array before the transformers run, which
    rejects a list of RDKit molecules outright. This does the same job
    while leaving the input untouched.

    Parameters
    ----------
    transformers : sequence
        Molecule transformers, or ``(name, transformer)`` pairs.

    Attributes
    ----------
    transformers_ : list
        The fitted transformers.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.representation import MorganFingerprint, MACCSKeysFingerprint
    >>> union = MoleculeFeatureUnion([
    ...     MorganFingerprint(n_bits=64), MACCSKeysFingerprint(),
    ... ])
    >>> union.fit_transform([Chem.MolFromSmiles("CCO")]).shape
    (1, 231)

    References
    ----------
    - Pedregosa, F. et al. (2011). "Scikit-learn." J. Mach. Learn. Res.,
      12, 2825-2830. https://jmlr.org/papers/v12/pedregosa11a.html
    """

    transformers_: List[Any]

    def __init__(self, transformers: Sequence[Any]) -> None:
        self.transformers = transformers

    @staticmethod
    def _unwrap(entry: Any) -> Any:
        return entry[1] if isinstance(entry, tuple) else entry

    def fit(
        self, X: Iterable[Any], y: Optional[npt.ArrayLike] = None
    ) -> "MoleculeFeatureUnion":
        """Fit every member on the same molecules.

        Parameters
        ----------
        X : iterable of Mol
        y : array-like, optional

        Returns
        -------
        MoleculeFeatureUnion
        """
        if not self.transformers:
            raise ValueError("MoleculeFeatureUnion needs at least one transformer.")
        mols = list(X)
        self.transformers_ = [
            self._unwrap(t).fit(mols, y) for t in self.transformers
        ]
        return self

    def transform(self, X: Iterable[Any]) -> npt.NDArray[np.float64]:
        """Horizontally stack every member's output.

        Parameters
        ----------
        X : iterable of Mol

        Returns
        -------
        ndarray of shape (n_molecules, total_features)
        """
        if not hasattr(self, "transformers_"):
            raise ModelNotFittedError(
                "MoleculeFeatureUnion must be fitted before transform()."
            )
        mols = list(X)
        blocks = [np.asarray(t.transform(mols), dtype=np.float64) for t in self.transformers_]
        return np.hstack(blocks)

    def get_feature_names_out(
        self, input_features: Optional[Any] = None
    ) -> npt.NDArray[np.str_]:
        """Concatenated feature names, prefixed by member.

        Returns
        -------
        ndarray of str
        """
        if not hasattr(self, "transformers_"):
            raise ModelNotFittedError(
                "MoleculeFeatureUnion must be fitted before "
                "get_feature_names_out()."
            )
        names: List[str] = []
        for entry, fitted in zip(self.transformers, self.transformers_):
            prefix = entry[0] if isinstance(entry, tuple) else type(fitted).__name__
            if hasattr(fitted, "get_feature_names_out"):
                names.extend(
                    f"{prefix}__{n}" for n in fitted.get_feature_names_out()
                )
            else:  # pragma: no cover - transformers here all provide names
                names.append(prefix)
        return np.asarray(names, dtype=object).astype(str)


class NaNHandler(BaseEstimator, TransformerMixin):
    """Replace or drop non-finite descriptor values.

    RDKit descriptors return ``NaN`` or ``inf`` for molecules where they
    are undefined — a logP contribution for an unparameterized element, a
    ring descriptor for an acyclic molecule. Left alone these propagate
    silently through scaling and crash the estimator much later, with an
    error that names neither the descriptor nor the molecule.

    Parameters
    ----------
    strategy : {"mean", "median", "constant", "drop_columns"}, default "median"
        How to handle them. ``"drop_columns"`` removes any column
        containing a non-finite value, which is the honest choice when a
        descriptor is undefined for a whole class of molecule rather
        than a stray one.
    fill_value : float, default 0.0
        Used by ``strategy="constant"``.
    max_nan_fraction : float, default 0.5
        Columns with a greater fraction of non-finite values are dropped
        regardless of strategy — imputing most of a column invents data.

    Attributes
    ----------
    statistics_ : ndarray
        Per-column fill values.
    support_ : ndarray of bool
        Columns retained.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1.0, np.nan], [3.0, 2.0]])
    >>> NaNHandler().fit_transform(X)
    array([[1., 2.],
           [3., 2.]])

    References
    ----------
    - Little, R. J. A. & Rubin, D. B. (2019). "Statistical Analysis with
      Missing Data," 3rd ed. Wiley.
      https://doi.org/10.1002/9781119482260
    - scikit-learn imputation documentation:
      https://scikit-learn.org/stable/modules/impute.html
    """

    statistics_: npt.NDArray[np.float64]
    support_: npt.NDArray[np.bool_]
    n_features_in_: int

    def __init__(
        self,
        strategy: Literal["mean", "median", "constant", "drop_columns"] = "median",
        fill_value: float = 0.0,
        max_nan_fraction: float = 0.5,
    ) -> None:
        self.strategy = strategy
        self.fill_value = fill_value
        self.max_nan_fraction = max_nan_fraction

    def fit(
        self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None
    ) -> "NaNHandler":
        """Learn per-column fill values and which columns to keep.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        NaNHandler
        """
        if self.strategy not in ("mean", "median", "constant", "drop_columns"):
            raise ValueError(
                "strategy must be 'mean', 'median', 'constant' or "
                f"'drop_columns', got {self.strategy!r}."
            )
        if not 0.0 <= self.max_nan_fraction <= 1.0:
            raise ValueError(
                f"max_nan_fraction must be in [0, 1], got {self.max_nan_fraction}."
            )

        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}.")
        self.n_features_in_ = arr.shape[1]

        bad = ~np.isfinite(arr)
        fraction = bad.mean(axis=0)
        if self.strategy == "drop_columns":
            self.support_ = np.asarray(~bad.any(axis=0), dtype=np.bool_)
        else:
            self.support_ = np.asarray(
                fraction <= self.max_nan_fraction, dtype=np.bool_
            )

        clean = np.where(bad, np.nan, arr)
        # An all-NaN column makes nanmean/nanmedian warn; that column is
        # dropped below, so the warning is noise rather than information.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if self.strategy == "mean":
                stats = np.nanmean(clean, axis=0)
            elif self.strategy == "median":
                stats = np.nanmedian(clean, axis=0)
            else:
                stats = np.full(arr.shape[1], self.fill_value, dtype=np.float64)
        # An all-NaN column has no usable statistic; it is dropped anyway,
        # but leave a finite value so the imputation itself cannot emit NaN.
        self.statistics_ = np.nan_to_num(stats, nan=self.fill_value)
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Impute and drop columns as learned during fit.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_retained_features)
        """
        if not hasattr(self, "statistics_"):
            raise ModelNotFittedError("NaNHandler must be fitted before transform().")
        arr = np.asarray(X, dtype=np.float64)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {arr.shape[1]} features, expected {self.n_features_in_}."
            )
        filled = np.where(np.isfinite(arr), arr, self.statistics_[None, :])
        return np.asarray(filled[:, self.support_], dtype=np.float64)

    def get_feature_names_out(
        self, input_features: Optional[Sequence[str]] = None
    ) -> npt.NDArray[np.str_]:
        """Names of the retained columns."""
        if not hasattr(self, "support_"):
            raise ModelNotFittedError(
                "NaNHandler must be fitted before get_feature_names_out()."
            )
        names = (
            list(input_features)
            if input_features is not None
            else [f"x{i}" for i in range(self.n_features_in_)]
        )
        return np.asarray(
            [n for n, keep in zip(names, self.support_) if keep], dtype=object
        ).astype(str)


class VarianceThresholdMol(BaseEstimator, TransformerMixin):
    """Drop near-constant descriptor columns.

    A descriptor that takes the same value for every molecule carries no
    information but still costs a parameter, and constant columns break
    correlation-based selection and scaling by producing zero variance.

    Parameters
    ----------
    threshold : float, default 0.0
        Columns with variance at or below this are dropped. The default
        removes exactly-constant columns.

    Attributes
    ----------
    variances_ : ndarray
        Per-column variance.
    support_ : ndarray of bool
        Columns retained.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    >>> VarianceThresholdMol().fit_transform(X).shape
    (3, 1)

    References
    ----------
    - scikit-learn feature selection documentation:
      https://scikit-learn.org/stable/modules/feature_selection.html
    """

    variances_: npt.NDArray[np.float64]
    support_: npt.NDArray[np.bool_]
    n_features_in_: int

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold

    def fit(
        self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None
    ) -> "VarianceThresholdMol":
        """Measure per-column variance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        VarianceThresholdMol
        """
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}.")
        if self.threshold < 0:
            raise ValueError(
                f"threshold must be non-negative, got {self.threshold}."
            )
        self.n_features_in_ = arr.shape[1]
        self.variances_ = np.asarray(np.nanvar(arr, axis=0), dtype=np.float64)
        self.support_ = self.variances_ > self.threshold
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Keep only the columns that passed the variance threshold."""
        if not hasattr(self, "support_"):
            raise ModelNotFittedError(
                "VarianceThresholdMol must be fitted before transform()."
            )
        arr = np.asarray(X, dtype=np.float64)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {arr.shape[1]} features, expected {self.n_features_in_}."
            )
        return np.asarray(arr[:, self.support_], dtype=np.float64)

    def get_support(self, indices: bool = False) -> npt.NDArray[Any]:
        """Mask (or indices) of the retained columns."""
        if not hasattr(self, "support_"):
            raise ModelNotFittedError(
                "VarianceThresholdMol must be fitted before get_support()."
            )
        return np.flatnonzero(self.support_) if indices else self.support_


class DescriptorScaler(BaseEstimator, TransformerMixin):
    """Standardize descriptors while keeping their names.

    Descriptors span wildly different scales — molecular weight in the
    hundreds, Fsp3 in [0, 1] — so any distance- or penalty-based method
    (SVM, k-NN, ridge, PCA, most applicability domains) is dominated by
    whichever descriptor happens to have the largest units unless they
    are scaled first. Fingerprints, being already 0/1, should *not* be
    scaled.

    Parameters
    ----------
    method : {"standard", "minmax", "robust"}, default "standard"
        ``"robust"`` centres on the median and scales by the IQR, which
        is the right choice when the descriptor distribution has the long
        tail typical of counts.
    clip : bool, default False
        Clip transformed values to the range seen during fit, preventing
        a single extreme test molecule from dominating downstream.

    Attributes
    ----------
    scaler_ : sklearn scaler
        The fitted scikit-learn scaler.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
    >>> scaled = DescriptorScaler().fit_transform(X)
    >>> bool(np.allclose(scaled.mean(axis=0), 0.0))
    True

    References
    ----------
    - scikit-learn preprocessing documentation:
      https://scikit-learn.org/stable/modules/preprocessing.html
    - Todeschini, R. & Consonni, V. (2009). "Molecular Descriptors for
      Chemoinformatics." Wiley-VCH.
      https://doi.org/10.1002/9783527628766
    """

    scaler_: Any
    n_features_in_: int

    def __init__(
        self,
        method: Literal["standard", "minmax", "robust"] = "standard",
        clip: bool = False,
    ) -> None:
        self.method = method
        self.clip = clip

    def fit(
        self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None
    ) -> "DescriptorScaler":
        """Fit the underlying scaler.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        DescriptorScaler
        """
        from sklearn.preprocessing import (
            MinMaxScaler,
            RobustScaler,
            StandardScaler,
        )

        scalers = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler,
        }
        if self.method not in scalers:
            raise ValueError(
                f"method must be one of {sorted(scalers)}, got {self.method!r}."
            )
        arr = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = arr.shape[1]
        self.scaler_ = scalers[self.method]().fit(arr)
        if self.clip:
            transformed = self.scaler_.transform(arr)
            self._low = transformed.min(axis=0)
            self._high = transformed.max(axis=0)
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Scale the descriptors."""
        if not hasattr(self, "scaler_"):
            raise ModelNotFittedError(
                "DescriptorScaler must be fitted before transform()."
            )
        out = np.asarray(
            self.scaler_.transform(np.asarray(X, dtype=np.float64)),
            dtype=np.float64,
        )
        if self.clip:
            out = np.clip(out, self._low, self._high)
        return out

    def inverse_transform(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Map scaled values back to the original units."""
        if not hasattr(self, "scaler_"):
            raise ModelNotFittedError(
                "DescriptorScaler must be fitted before inverse_transform()."
            )
        return np.asarray(
            self.scaler_.inverse_transform(np.asarray(X, dtype=np.float64)),
            dtype=np.float64,
        )

    def get_feature_names_out(
        self, input_features: Optional[Sequence[str]] = None
    ) -> npt.NDArray[np.str_]:
        """Pass feature names through unchanged (scaling is column-wise)."""
        if not hasattr(self, "scaler_"):
            raise ModelNotFittedError(
                "DescriptorScaler must be fitted before get_feature_names_out()."
            )
        names = (
            list(input_features)
            if input_features is not None
            else [f"x{i}" for i in range(self.n_features_in_)]
        )
        return np.asarray(names, dtype=object).astype(str)


def make_qsar_pipeline(
    representation: Any,
    model: Any,
    scale: bool = False,
    handle_nan: bool = True,
    from_smiles: bool = False,
) -> "Pipeline":
    """Assemble the standard QSAR pipeline.

    Chains, in order: optional SMILES parsing, the representation, NaN
    handling, optional scaling, and the estimator. Building it as one
    ``Pipeline`` matters for more than tidiness — it is what keeps the
    imputation and scaling *fitted on training folds only*, which is
    exactly the leak that inflates cross-validated QSAR scores when
    preprocessing is done up-front on the whole dataset.

    Parameters
    ----------
    representation : transformer
        A molecule featurizer from :mod:`qsarkit.representation`.
    model : estimator
        The final regressor or classifier.
    scale : bool, default False
        Insert a :class:`DescriptorScaler`. Leave off for fingerprints,
        which are already 0/1; turn on for descriptors.
    handle_nan : bool, default True
        Insert a :class:`NaNHandler`.
    from_smiles : bool, default False
        Prepend a :class:`SmilesToMol` so the pipeline accepts SMILES.

    Returns
    -------
    sklearn.pipeline.Pipeline

    Examples
    --------
    >>> from qsarkit.representation import MorganFingerprint
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> pipe = make_qsar_pipeline(
    ...     MorganFingerprint(n_bits=64), RandomForestRegressor(n_estimators=5),
    ...     from_smiles=True,
    ... )
    >>> _ = pipe.fit(["CCO", "CCN", "c1ccccc1"], [1.0, 2.0, 3.0])
    >>> pipe.predict(["CCO"]).shape
    (1,)

    References
    ----------
    - Pedregosa, F. et al. (2011). "Scikit-learn." J. Mach. Learn. Res.,
      12, 2825-2830. https://jmlr.org/papers/v12/pedregosa11a.html
    - Cawley, G. C. & Talbot, N. L. C. (2010). "On Over-fitting in Model
      Selection and Subsequent Selection Bias in Performance Evaluation."
      J. Mach. Learn. Res., 11, 2079-2107.
      https://jmlr.org/papers/v11/cawley10a.html
    """
    from sklearn.pipeline import Pipeline

    steps: List[Any] = []
    if from_smiles:
        steps.append(("smiles_to_mol", SmilesToMol()))
    steps.append(("representation", representation))
    if handle_nan:
        steps.append(("nan", NaNHandler()))
    if scale:
        steps.append(("scale", DescriptorScaler()))
    steps.append(("model", model))
    return Pipeline(steps)
