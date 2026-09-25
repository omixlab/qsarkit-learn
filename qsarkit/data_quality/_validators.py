"""Structure validation and activity-outlier detection."""

from __future__ import annotations

from qsarkit.base.exceptions import RDKIT_MOLECULE_ERRORS
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from rdkit.Chem import Mol

__all__ = [
    "ValidationIssue",
    "StructureValidator",
    "ActivityOutlierDetector",
    "check_activity_units",
]

# Elements that make a record unsuitable for a standard organic QSAR model:
# anything outside common organic chemistry plus the halogens.
_ORGANIC_ELEMENTS = {
    "H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I",
}


@dataclass
class ValidationIssue:
    """One problem found with one record.

    Attributes
    ----------
    index : int
        Position of the offending record.
    code : str
        Machine-readable issue code, e.g. ``"inorganic"``.
    message : str
        Human-readable explanation.
    fatal : bool
        Whether the record should be dropped rather than merely flagged.
    """

    index: int
    code: str
    message: str
    fatal: bool = True


class StructureValidator:
    """Flag records that are not usable molecules for QSAR modeling.

    Runs after standardization and before modeling. Each check
    corresponds to a category of record that routinely appears in public
    datasets and quietly degrades a model: unparseable structures,
    mixtures whose activity cannot be attributed to one component,
    inorganics and organometallics outside the applicability of organic
    descriptors, isotopically labelled tracers, and molecules far outside
    the size range the descriptors were designed for.

    Parameters
    ----------
    allow_inorganic : bool, default False
        Keep molecules containing elements outside common organic
        chemistry.
    allow_mixtures : bool, default False
        Keep multi-component records. These are usually salts that
        survived desalting, or genuine mixtures whose activity belongs to
        no single structure.
    allow_isotopes : bool, default False
        Keep isotopically labelled molecules.
    min_heavy_atoms : int, default 3
        Smallest acceptable molecule. Fragments below this carry almost
        no descriptor signal.
    max_heavy_atoms : int, default 150
        Largest acceptable molecule, excluding peptides and polymers that
        most descriptor sets were never calibrated on.
    require_carbon : bool, default True
        Require at least one carbon atom.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mols = [Chem.MolFromSmiles(s) for s in ("CCO", "[Na+].[Cl-]", "O")]
    >>> validator = StructureValidator()
    >>> issues = validator.validate(mols)
    >>> sorted({i.code for i in issues})
    ['inorganic', 'mixture', 'no_carbon', 'too_small']

    Ethanol passes; sodium chloride is a mixture of inorganic ions and
    water is too small to carry descriptor signal.

    The ``charged`` check looks at *net* charge, so a zwitterion is not
    mistaken for a record that escaped neutralization:

    >>> glycine = Chem.MolFromSmiles("[NH3+]CC(=O)[O-]")
    >>> [i.code for i in validator.validate([glycine])]
    []
    >>> acetate = Chem.MolFromSmiles("CC(=O)[O-]")
    >>> [i.code for i in validator.validate([acetate])]
    ['charged']

    References
    ----------
    - Fourches, D., Muratov, E. & Tropsha, A. (2010). "Trust, But Verify."
      J. Chem. Inf. Model., 50(7), 1189-1204.
      https://doi.org/10.1021/ci100176x
    - Fourches, D., Muratov, E. & Tropsha, A. (2016). "Trust, but Verify
      II." J. Chem. Inf. Model., 56(7), 1243-1252.
      https://doi.org/10.1021/acs.jcim.6b00129
    - Young, D. et al. (2008). "Are the Chemical Structures in Your QSAR
      Correct?" QSAR Comb. Sci., 27(11-12), 1337-1345.
      https://doi.org/10.1002/qsar.200810084
    """

    def __init__(
        self,
        allow_inorganic: bool = False,
        allow_mixtures: bool = False,
        allow_isotopes: bool = False,
        min_heavy_atoms: int = 3,
        max_heavy_atoms: int = 150,
        require_carbon: bool = True,
    ) -> None:
        self.allow_inorganic = allow_inorganic
        self.allow_mixtures = allow_mixtures
        self.allow_isotopes = allow_isotopes
        self.min_heavy_atoms = min_heavy_atoms
        self.max_heavy_atoms = max_heavy_atoms
        self.require_carbon = require_carbon

    def _check_one(self, index: int, mol: Any) -> List[ValidationIssue]:
        from rdkit import Chem

        if mol is None:
            return [
                ValidationIssue(index, "unparseable", "Molecule is None.", True)
            ]

        issues: List[ValidationIssue] = []
        try:
            Chem.SanitizeMol(Chem.Mol(mol))
        except RDKIT_MOLECULE_ERRORS as exc:
            issues.append(
                ValidationIssue(index, "sanitization_failed", str(exc), True)
            )
            return issues

        symbols = {a.GetSymbol() for a in mol.GetAtoms()}
        heavy = mol.GetNumHeavyAtoms()

        if not self.allow_mixtures and len(Chem.GetMolFrags(mol)) > 1:
            issues.append(
                ValidationIssue(
                    index, "mixture",
                    f"Record has {len(Chem.GetMolFrags(mol))} disconnected "
                    "components; activity cannot be attributed to one structure.",
                    True,
                )
            )
        if not self.allow_inorganic and not symbols <= _ORGANIC_ELEMENTS:
            issues.append(
                ValidationIssue(
                    index, "inorganic",
                    f"Contains non-organic elements: "
                    f"{sorted(symbols - _ORGANIC_ELEMENTS)}.",
                    True,
                )
            )
        if self.require_carbon and "C" not in symbols:
            issues.append(
                ValidationIssue(index, "no_carbon", "Contains no carbon.", True)
            )
        if not self.allow_isotopes and any(a.GetIsotope() for a in mol.GetAtoms()):
            issues.append(
                ValidationIssue(
                    index, "isotope", "Contains isotopic labels.", True
                )
            )
        if heavy < self.min_heavy_atoms:
            issues.append(
                ValidationIssue(
                    index, "too_small",
                    f"{heavy} heavy atoms, below the minimum of "
                    f"{self.min_heavy_atoms}.",
                    True,
                )
            )
        if heavy > self.max_heavy_atoms:
            issues.append(
                ValidationIssue(
                    index, "too_large",
                    f"{heavy} heavy atoms, above the maximum of "
                    f"{self.max_heavy_atoms}.",
                    True,
                )
            )
        # Net charge, not per-atom charge. A zwitterion (glycine,
        # ciprofloxacin, any betaine) carries formal charges on individual
        # atoms while summing to zero; it is a perfectly valid neutral
        # record and flagging it would bury the real signal, which is an
        # unbalanced ion left behind because neutralization did not run.
        net_charge = Chem.GetFormalCharge(mol)
        if net_charge != 0:
            issues.append(
                ValidationIssue(
                    index, "charged",
                    f"Molecule carries a net formal charge of {net_charge:+d}; "
                    "check that neutralization ran.",
                    False,
                )
            )
        if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
            issues.append(
                ValidationIssue(
                    index, "radical", "Molecule contains radical electrons.", True
                )
            )
        return issues

    def validate(self, mols: Sequence[Any]) -> List[ValidationIssue]:
        """Check every molecule and return all issues found.

        Parameters
        ----------
        mols : sequence of Mol

        Returns
        -------
        list of ValidationIssue
            Ordered by record index; a record may raise several issues.
        """
        return [
            issue
            for i, mol in enumerate(mols)
            for issue in self._check_one(i, mol)
        ]

    def valid_mask(self, mols: Sequence[Any]) -> npt.NDArray[np.bool_]:
        """Boolean mask of records with no fatal issue.

        Parameters
        ----------
        mols : sequence of Mol

        Returns
        -------
        ndarray of bool of shape (n_molecules,)
        """
        mask = np.ones(len(mols), dtype=np.bool_)
        for issue in self.validate(mols):
            if issue.fatal:
                mask[issue.index] = False
        return mask

    def to_dataframe(self, issues: Sequence[ValidationIssue]) -> "pd.DataFrame":
        """Render issues as a table.

        Parameters
        ----------
        issues : sequence of ValidationIssue

        Returns
        -------
        pandas.DataFrame
            Columns ``index``, ``code``, ``message``, ``fatal``.
        """
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "index": i.index, "code": i.code,
                    "message": i.message, "fatal": i.fatal,
                }
                for i in issues
            ],
            columns=["index", "code", "message", "fatal"],
        )


class ActivityOutlierDetector:
    """Flag activity values that look like errors rather than chemistry.

    Distinguishes two different things that both get called "outliers":

    - **Distributional outliers** — values far from the rest of the
      dataset, found by z-score, modified z-score or the IQR rule.
    - **Structure-activity outliers** — values far from what the
      compound's nearest structural neighbours would predict. These are
      the interesting ones, but note that a genuine activity cliff looks
      exactly like a data error from this angle, so use
      :mod:`qsarkit.sar` to tell them apart before deleting anything.

    Parameters
    ----------
    method : {"zscore", "modified_zscore", "iqr", "neighbor"}, default "modified_zscore"
        Detection rule. The modified z-score uses the median and MAD,
        so a few extreme values cannot inflate the scale and hide
        themselves — which is exactly what happens with a plain z-score.
    threshold : float, default 3.5
        Cutoff. 3.5 is the conventional modified-z-score limit; use ~3
        for ``"zscore"`` and 1.5 for ``"iqr"``.
    n_neighbors : int, default 5
        Neighbours used by ``method="neighbor"``.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.concatenate([np.full(20, 5.0), [50.0]])
    >>> detector = ActivityOutlierDetector()
    >>> bool(detector.detect(y)[-1])
    True

    References
    ----------
    - Iglewicz, B. & Hoaglin, D. C. (1993). "How to Detect and Handle
      Outliers." ASQC Quality Press. (modified z-score, MAD-based)
    - Tukey, J. W. (1977). "Exploratory Data Analysis." Addison-Wesley.
      (the IQR rule)
    - Fourches, D., Muratov, E. & Tropsha, A. (2010). J. Chem. Inf.
      Model., 50(7), 1189-1204. https://doi.org/10.1021/ci100176x
    - Maggiora, G. M. (2006). "On Outliers and Activity Cliffs."
      J. Chem. Inf. Model., 46(4), 1535. https://doi.org/10.1021/ci060117s
    """

    def __init__(
        self,
        method: Literal["zscore", "modified_zscore", "iqr", "neighbor"] = (
            "modified_zscore"
        ),
        threshold: float = 3.5,
        n_neighbors: int = 5,
    ) -> None:
        self.method = method
        self.threshold = threshold
        self.n_neighbors = n_neighbors

    def scores(
        self,
        activities: npt.ArrayLike,
        mols: Optional[Sequence[Any]] = None,
    ) -> npt.NDArray[np.float64]:
        """Per-record outlier score (larger = more anomalous).

        Parameters
        ----------
        activities : array-like of shape (n_samples,)
        mols : sequence of Mol, optional
            Required for ``method="neighbor"``.

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        y = np.asarray(activities, dtype=np.float64).ravel()
        if y.size == 0:
            return np.zeros(0, dtype=np.float64)

        if self.method == "zscore":
            std = float(y.std(ddof=1)) if y.size > 1 else 0.0
            if std == 0.0:
                return np.zeros_like(y)
            return np.abs((y - y.mean()) / std)

        if self.method == "modified_zscore":
            median = float(np.median(y))
            mad = float(np.median(np.abs(y - median)))
            if mad == 0.0:
                # Every value identical apart from a few; fall back to the
                # mean absolute deviation so those few are still visible.
                mean_ad = float(np.mean(np.abs(y - median)))
                if mean_ad == 0.0:
                    return np.zeros_like(y)
                return np.abs(y - median) / (1.253314 * mean_ad)
            return 0.6745 * np.abs(y - median) / mad

        if self.method == "iqr":
            q1, q3 = np.percentile(y, [25, 75])
            iqr = float(q3 - q1)
            if iqr == 0.0:
                return np.zeros_like(y)
            below = (q1 - y) / iqr
            above = (y - q3) / iqr
            return np.maximum(np.maximum(below, above), 0.0)

        if self.method == "neighbor":
            if mols is None:
                raise ValueError(
                    "method='neighbor' needs molecules to find structural "
                    "neighbours; pass `mols`."
                )
            return self._neighbor_scores(y, mols)

        raise ValueError(
            "method must be 'zscore', 'modified_zscore', 'iqr' or 'neighbor', "
            f"got {self.method!r}."
        )

    def _neighbor_scores(
        self, y: npt.NDArray[np.float64], mols: Sequence[Any]
    ) -> npt.NDArray[np.float64]:
        """Absolute deviation from the similarity-weighted neighbour mean."""
        from rdkit.Chem import rdFingerprintGenerator

        from qsarkit.neighbors import tanimoto_similarity_matrix

        if len(mols) != len(y):
            raise ValueError(
                f"activities has length {len(y)} but there are {len(mols)} molecules."
            )
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fps = np.array(
            [
                np.zeros(2048) if m is None else np.asarray(list(gen.GetFingerprint(m)))
                for m in mols
            ],
            dtype=np.float64,
        )
        sim = tanimoto_similarity_matrix(fps)
        np.fill_diagonal(sim, -np.inf)

        k = min(self.n_neighbors, len(y) - 1)
        if k < 1:
            return np.zeros_like(y)
        neighbors = np.argsort(-sim, axis=1)[:, :k]
        rows = np.arange(len(y))[:, None]
        weights = np.clip(sim[rows, neighbors], 0.0, None)
        totals = weights.sum(axis=1)
        expected = np.where(
            totals > 0,
            (weights * y[neighbors]).sum(axis=1) / np.where(totals > 0, totals, 1.0),
            y.mean(),
        )
        residual = np.abs(y - expected)
        scale = float(np.median(residual))
        return residual / scale if scale > 0 else residual

    def detect(
        self,
        activities: npt.ArrayLike,
        mols: Optional[Sequence[Any]] = None,
    ) -> npt.NDArray[np.bool_]:
        """Boolean mask of records flagged as outliers.

        Parameters
        ----------
        activities : array-like of shape (n_samples,)
        mols : sequence of Mol, optional

        Returns
        -------
        ndarray of bool of shape (n_samples,)
        """
        return np.asarray(
            self.scores(activities, mols) > self.threshold, dtype=np.bool_
        )


def check_activity_units(
    activities: npt.ArrayLike,
    unit: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Sanity-check an activity column before modeling.

    Catches the most common and most damaging data error in QSAR: a
    column that mixes units, or is on a linear concentration scale when
    the modeling assumes a logarithmic one. A dataset spanning six orders
    of magnitude in raw nM will be dominated by its largest values, and
    the resulting model is fitted almost entirely to the inactives.

    Parameters
    ----------
    activities : array-like of shape (n_samples,)
        The values to inspect.
    unit : str, optional
        Declared unit, e.g. ``"nM"``. Used to sharpen the advice.
    endpoint : str, optional
        Declared endpoint, e.g. ``"IC50"``.

    Returns
    -------
    dict
        ``n``, ``n_missing``, ``n_non_positive``, ``min``, ``max``,
        ``median``, ``log_range`` (orders of magnitude spanned),
        ``looks_logarithmic`` and ``warnings`` (a list of plain-language
        problems found).

    Examples
    --------
    A raw nanomolar column spanning six decades is flagged:

    >>> report = check_activity_units([1.0, 10.0, 1000.0, 1e6], unit="nM")
    >>> report["looks_logarithmic"]
    False
    >>> bool(report["warnings"])
    True

    A pActivity column is recognised and passes clean:

    >>> report = check_activity_units([5.1, 6.2, 7.3], endpoint="IC50")
    >>> report["looks_logarithmic"], report["warnings"]
    (True, [])

    Molar values spanning little range are *not* mistaken for a p-scale
    column, even though their numeric range is narrow -- a pActivity of
    1e-9 would mean an IC50 near 1 M:

    >>> report = check_activity_units([1e-9, 5e-8], endpoint="IC50")
    >>> report["looks_logarithmic"]
    False
    >>> report["warnings"][0].startswith("All values are below 1")
    True

    References
    ----------
    - Fourches, D., Muratov, E. & Tropsha, A. (2016). "Trust, but Verify
      II." J. Chem. Inf. Model., 56(7), 1243-1252.
      https://doi.org/10.1021/acs.jcim.6b00129
    - Kalliokoski, T. et al. (2013). "Comparability of Mixed IC50 Data."
      PLoS ONE, 8(4), e61007.
      https://doi.org/10.1371/journal.pone.0061007
    """
    y = np.asarray(activities, dtype=np.float64).ravel()
    finite = y[np.isfinite(y)]
    warnings: List[str] = []

    n_missing = int(y.size - finite.size)
    if n_missing:
        warnings.append(f"{n_missing} missing or non-finite values.")
    if finite.size == 0:
        return {
            "n": int(y.size), "n_missing": n_missing, "n_non_positive": 0,
            "min": float("nan"), "max": float("nan"), "median": float("nan"),
            "log_range": float("nan"), "looks_logarithmic": False,
            "warnings": warnings + ["No usable values."],
        }

    n_non_positive = int(np.sum(finite <= 0))
    positive = finite[finite > 0]
    log_range = (
        float(np.log10(positive.max()) - np.log10(positive.min()))
        if positive.size > 1
        else 0.0
    )
    # A logarithmic activity column sits in roughly 0-15 and spans a
    # narrow numeric range; a raw concentration column spans decades.
    #
    # The `max >= 1` term is what separates a p-scale column from molar
    # values that happen to span little range: [1e-9, 5e-8] is narrow and
    # small, but a pActivity of 1e-9 would mean an IC50 of about 1 M. Real
    # p-scale data sits around 4-10, so at least one value clears 1.
    looks_logarithmic = bool(
        finite.max() <= 20
        and finite.max() >= 1.0
        and finite.min() >= -5
        and log_range < 2.5
    )

    if not looks_logarithmic and log_range > 3:
        warnings.append(
            f"Values span {log_range:.1f} orders of magnitude, which suggests a "
            "raw concentration scale. Convert to pActivity (-log10 molar) "
            "before modeling."
        )
    if positive.size and positive.max() < 1.0:
        warnings.append(
            f"All values are below 1 (max {positive.max():.3g}), which suggests "
            "raw molar concentrations rather than a pActivity scale. Convert "
            "with qsarkit.utils.to_pactivity before modeling."
        )
    if n_non_positive:
        warnings.append(
            f"{n_non_positive} non-positive values, which cannot be "
            "log-transformed."
        )
    if unit and unit.lower() in {"nm", "um", "µm", "mm", "m", "pm"} and looks_logarithmic:
        warnings.append(
            f"Unit is declared as {unit} but the values look already "
            "logarithmic; check whether the unit label is stale."
        )
    if finite.size > 1 and float(np.std(finite)) == 0.0:
        warnings.append("All values are identical; nothing to model.")

    return {
        "n": int(y.size),
        "n_missing": n_missing,
        "n_non_positive": n_non_positive,
        "min": float(finite.min()),
        "max": float(finite.max()),
        "median": float(np.median(finite)),
        "log_range": log_range,
        "looks_logarithmic": looks_logarithmic,
        "endpoint": endpoint,
        "unit": unit,
        "warnings": warnings,
    }
