"""End-to-end dataset curation with a provenance report."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from qsarkit.data_quality._duplicates import DuplicateDetector, merge_replicates
from qsarkit.data_quality._validators import (
    ActivityOutlierDetector,
    StructureValidator,
    check_activity_units,
)

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from rdkit.Chem import Mol

__all__ = ["CurationReport", "DataCurationPipeline"]


@dataclass
class CurationReport:
    """What curation did, and why.

    A curated dataset is only trustworthy if the curation is auditable —
    OECD principle 1 asks for a defined endpoint and a documented
    dataset, and "we cleaned it" is not documentation.

    Attributes
    ----------
    n_input : int
        Records supplied.
    n_output : int
        Records surviving.
    stages : list of dict
        One entry per stage: its name, the counts before and after, and
        the indices it removed.
    removed : dict
        Original index -> reason it was removed.
    activity_check : dict
        Output of :func:`~qsarkit.data_quality.check_activity_units`.
    warnings : list of str
        Problems worth a human's attention.
    """

    n_input: int = 0
    n_output: int = 0
    stages: List[Dict[str, Any]] = field(default_factory=list)
    removed: Dict[int, str] = field(default_factory=dict)
    activity_check: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def n_removed(self) -> int:
        """Number of records removed."""
        return self.n_input - self.n_output

    @property
    def retention(self) -> float:
        """Fraction of records surviving curation."""
        return self.n_output / self.n_input if self.n_input else 0.0

    def to_dataframe(self) -> "pd.DataFrame":
        """Per-stage summary table.

        Returns
        -------
        pandas.DataFrame
            Columns ``stage``, ``n_before``, ``n_after``, ``n_removed``.
        """
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "stage": s["stage"],
                    "n_before": s["n_before"],
                    "n_after": s["n_after"],
                    "n_removed": s["n_before"] - s["n_after"],
                }
                for s in self.stages
            ],
            columns=["stage", "n_before", "n_after", "n_removed"],
        )

    def summary(self) -> str:
        """Human-readable report.

        Returns
        -------
        str
        """
        lines = [
            f"Curation: {self.n_input} -> {self.n_output} records "
            f"({self.retention:.1%} retained)",
        ]
        for stage in self.stages:
            dropped = stage["n_before"] - stage["n_after"]
            lines.append(
                f"  {stage['stage']:<22} {stage['n_before']:>6} -> "
                f"{stage['n_after']:<6} ({dropped} removed)"
            )
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"<CurationReport {self.n_input} -> {self.n_output} "
            f"({self.retention:.1%} retained), {len(self.warnings)} warnings>"
        )


class DataCurationPipeline:
    """Standardize, validate, de-duplicate and screen a QSAR dataset.

    Runs the curation protocol of Fourches, Muratov and Tropsha in the
    order that matters: standardization first (so that duplicate
    detection sees comparable structures), then structural validation,
    then replicate merging, then activity outliers. Running de-duplication
    before standardization is the classic mistake — the same compound
    stored as a salt and as a free base will not be recognised as a
    duplicate, and both copies survive into the model.

    Every stage records what it removed, so the output is auditable.

    Parameters
    ----------
    standardize : bool, default True
        Run :class:`~qsarkit.chemistry.MolecularStandardizer` first.
    validate_structures : bool, default True
        Apply :class:`~qsarkit.data_quality.StructureValidator`.
    remove_duplicates : bool, default True
        Merge replicate structures.
    remove_outliers : bool, default False
        Drop activity outliers. Off by default, because on a congeneric
        series a genuine activity cliff is indistinguishable from a data
        error by statistics alone — see :mod:`qsarkit.sar` before
        enabling this.
    duplicate_level : str, default "inchikey"
        Passed to :class:`DuplicateDetector`.
    merge_method : str, default "median"
        Passed to :func:`merge_replicates`.
    max_spread : float, optional, default 1.0
        Replicate groups disagreeing by more than this are discarded.
    outlier_method : str, default "modified_zscore"
        Passed to :class:`ActivityOutlierDetector`.
    outlier_threshold : float, default 3.5
        Passed to :class:`ActivityOutlierDetector`.
    validator : StructureValidator, optional
        Custom validator.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mols = [Chem.MolFromSmiles(s) for s in
    ...         ("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]", "CCO", "OCC", "[Na+].[Cl-]")]
    >>> curated, y, report = DataCurationPipeline().run(mols, [5.0, 6.0, 6.2, 1.0])
    >>> report.n_input
    4
    >>> report.n_output < report.n_input
    True

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
    - Tropsha, A. (2010). "Best Practices for QSAR Model Development,
      Validation, and Exploitation." Mol. Inform., 29(6-7), 476-488.
      https://doi.org/10.1002/minf.201000061
    - OECD (2007). Guidance Document No. 69, ENV/JM/MONO(2007)2.
      https://doi.org/10.1787/9789264085442-en
    """

    def __init__(
        self,
        standardize: bool = True,
        validate_structures: bool = True,
        remove_duplicates: bool = True,
        remove_outliers: bool = False,
        duplicate_level: str = "inchikey",
        merge_method: str = "median",
        max_spread: Optional[float] = 1.0,
        outlier_method: str = "modified_zscore",
        outlier_threshold: float = 3.5,
        validator: Optional[StructureValidator] = None,
    ) -> None:
        self.standardize = standardize
        self.validate_structures = validate_structures
        self.remove_duplicates = remove_duplicates
        self.remove_outliers = remove_outliers
        self.duplicate_level = duplicate_level
        self.merge_method = merge_method
        self.max_spread = max_spread
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.validator = validator

    def run(
        self,
        mols: Sequence[Any],
        activities: Optional[npt.ArrayLike] = None,
        unit: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> Tuple[List[Any], Optional[npt.NDArray[np.float64]], CurationReport]:
        """Curate a dataset.

        Parameters
        ----------
        mols : sequence of Mol
            Molecules to curate.
        activities : array-like, optional
            Parallel activity values, kept aligned throughout.
        unit : str, optional
            Activity unit, for the sanity check.
        endpoint : str, optional
            Endpoint name, recorded in the report.

        Returns
        -------
        mols : list of Mol
            The curated molecules.
        activities : ndarray or None
            The curated activities.
        report : CurationReport
            What happened at each stage.
        """
        current: List[Any] = list(mols)
        y = None if activities is None else np.asarray(activities, dtype=np.float64)
        if y is not None and len(y) != len(current):
            raise ValueError(
                f"activities has length {len(y)} but there are {len(current)} molecules."
            )

        # Track original positions so the report can name what was dropped.
        origin = list(range(len(current)))
        report = CurationReport(n_input=len(current))

        if y is not None:
            report.activity_check = check_activity_units(y, unit, endpoint)
            report.warnings.extend(report.activity_check["warnings"])

        def record(stage: str, before: int, kept: List[int], reason: str) -> None:
            """Note a stage's effect and update the origin/removal bookkeeping."""
            keep_set = set(kept)
            for pos, orig in enumerate(origin):
                if pos not in keep_set:
                    report.removed[orig] = reason
            report.stages.append(
                {"stage": stage, "n_before": before, "n_after": len(kept)}
            )

        if self.standardize:
            from qsarkit.chemistry import MolecularStandardizer

            before = len(current)
            current = list(MolecularStandardizer().transform(current))
            kept = [i for i, m in enumerate(current) if m is not None]
            record("standardize", before, kept, "standardization failed")
            current = [current[i] for i in kept]
            origin = [origin[i] for i in kept]
            if y is not None:
                y = y[np.asarray(kept, dtype=int)] if kept else y[:0]

        if self.validate_structures:
            validator = self.validator or StructureValidator()
            before = len(current)
            issues = validator.validate(current)
            reasons = {
                i.index: i.code for i in issues if i.fatal
            }
            kept = [i for i in range(len(current)) if i not in reasons]
            keep_set = set(kept)
            for pos, orig in enumerate(origin):
                if pos not in keep_set:
                    report.removed[orig] = f"invalid structure: {reasons[pos]}"
            report.stages.append(
                {"stage": "validate", "n_before": before, "n_after": len(kept)}
            )
            current = [current[i] for i in kept]
            origin = [origin[i] for i in kept]
            if y is not None:
                y = y[np.asarray(kept, dtype=int)] if kept else y[:0]

        if self.remove_duplicates and current:
            before = len(current)
            if y is None:
                detector = DuplicateDetector(level=self.duplicate_level)  # type: ignore[arg-type]
                seen: set[int] = set()
                for group in detector.find_duplicates(current):
                    seen.update(group.indices[1:])
                kept = [i for i in range(len(current)) if i not in seen]
                record("deduplicate", before, kept, "duplicate structure")
                current = [current[i] for i in kept]
                origin = [origin[i] for i in kept]
            else:
                merged, y_merged, merge_report = merge_replicates(
                    current, y,
                    level=self.duplicate_level,  # type: ignore[arg-type]
                    method=self.merge_method,  # type: ignore[arg-type]
                    max_spread=self.max_spread,
                )
                # merge_replicates keeps the first member of each group, so
                # recover which original positions survived by re-running the
                # same grouping over the pre-merge list.
                kept = self._surviving_positions(current, merged)
                keep_set = set(kept)
                for pos, orig in enumerate(origin):
                    if pos not in keep_set:
                        report.removed[orig] = "duplicate structure (merged)"
                report.stages.append(
                    {"stage": "deduplicate", "n_before": before, "n_after": len(merged)}
                )
                if merge_report["n_discarded"]:
                    report.warnings.append(
                        f"{merge_report['n_discarded']} replicate groups discarded "
                        f"for disagreeing by more than {self.max_spread} units."
                    )
                current, y = merged, y_merged
                origin = [origin[i] for i in kept]

        if self.remove_outliers and y is not None and current:
            before = len(current)
            outlier_detector = ActivityOutlierDetector(
                method=self.outlier_method,  # type: ignore[arg-type]
                threshold=self.outlier_threshold,
            )
            flags = outlier_detector.detect(y, current)
            kept = [i for i in range(len(current)) if not flags[i]]
            record("remove_outliers", before, kept, "activity outlier")
            current = [current[i] for i in kept]
            origin = [origin[i] for i in kept]
            y = y[np.asarray(kept, dtype=int)] if kept else y[:0]

        report.n_output = len(current)
        if report.retention < 0.5 and report.n_input:
            report.warnings.append(
                f"Curation removed {1 - report.retention:.0%} of the dataset; "
                "check the per-stage counts before proceeding."
            )
        return current, y, report

    @staticmethod
    def _surviving_positions(
        before: Sequence[Any], after: Sequence[Any]
    ) -> List[int]:
        """Positions in ``before`` corresponding to the members of ``after``.

        ``merge_replicates`` returns the first record of each group, in
        original order, so matching object identity recovers the mapping
        exactly without re-deriving the grouping.
        """
        remaining = list(after)
        positions: List[int] = []
        for i, mol in enumerate(before):
            if remaining and mol is remaining[0]:
                positions.append(i)
                remaining.pop(0)
        return positions
