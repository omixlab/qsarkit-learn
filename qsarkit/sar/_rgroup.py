"""R-group decomposition, SAR tables and Free-Wilson analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from rdkit.Chem import Mol

__all__ = ["RGroupAnalyzer", "SARTable", "FreeWilsonAnalysis"]


class RGroupAnalyzer:
    """Decompose a congeneric series into a core plus R-group substituents.

    Wraps RDKit's ``rdRGroupDecomposition`` to turn a set of analogues
    into the R-group table medicinal chemists actually reason with:
    one row per compound, one column per substitution point.

    Parameters
    ----------
    core : Mol or str, optional
        The scaffold, as a Mol or SMARTS/SMILES string. When ``None``,
        the most common Bemis-Murcko scaffold in the series is used.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mols = [Chem.MolFromSmiles(s) for s in ("c1ccccc1Cl", "c1ccccc1Br")]
    >>> analyzer = RGroupAnalyzer(core="c1ccccc1")
    >>> table = analyzer.decompose(mols)
    >>> "Core" in table.columns
    True

    References
    ----------
    - Bemis, G. W. & Murcko, M. A. (1996). "The Properties of Known
      Drugs. 1. Molecular Frameworks." J. Med. Chem., 39(15), 2887-2893.
      https://doi.org/10.1021/jm9602928
    - RDKit ``rdRGroupDecomposition`` documentation:
      https://www.rdkit.org/docs/source/rdkit.Chem.rdRGroupDecomposition.html
    - Stierand, K. & Rarey, M. (2010). "Drawing the PDB: Protein-Ligand
      Complexes in Two Dimensions." ACS Med. Chem. Lett., 1(9), 540-545.
      https://doi.org/10.1021/ml100164p
    """

    def __init__(self, core: Optional[Any] = None) -> None:
        self.core = core

    def _resolve_core(self, mols: Sequence["Mol"]) -> "Mol":
        from collections import Counter

        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        if self.core is not None:
            if isinstance(self.core, str):
                core = Chem.MolFromSmarts(self.core) or Chem.MolFromSmiles(self.core)
                if core is None:
                    raise ValueError(f"Could not parse core {self.core!r}.")
                return core
            return self.core

        counts = Counter(
            Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
            for m in mols
            if m is not None
        )
        if not counts:
            raise ValueError("Cannot infer a core from an empty molecule set.")
        most_common = counts.most_common(1)[0][0]
        core = Chem.MolFromSmiles(most_common)
        if core is None:
            raise ValueError(f"Inferred core {most_common!r} could not be parsed.")
        return core

    def decompose(self, mols: Sequence["Mol"]) -> "pd.DataFrame":
        """Run R-group decomposition over a series.

        Parameters
        ----------
        mols : sequence of Mol
            Analogues sharing a common core.

        Returns
        -------
        pandas.DataFrame
            One row per successfully decomposed molecule; a ``Core``
            column plus one ``R1``, ``R2``, ... column per attachment
            point, all as SMILES. Molecules that do not match the core
            are omitted, and their positions are recorded in the
            frame's ``attrs["unmatched"]``.
        """
        import pandas as pd
        from rdkit import Chem
        from rdkit.Chem import rdRGroupDecomposition

        core = self._resolve_core(mols)
        valid = [(i, m) for i, m in enumerate(mols) if m is not None]
        decomposition, unmatched_positions = rdRGroupDecomposition.RGroupDecompose(
            [core], [m for _, m in valid], asSmiles=True, asRows=True
        )
        matched_original = [
            valid[i][0] for i in range(len(valid)) if i not in set(unmatched_positions)
        ]
        frame = pd.DataFrame(decomposition)
        frame.attrs["unmatched"] = [
            valid[i][0] for i in set(unmatched_positions)
        ]
        if not frame.empty:
            frame.insert(0, "molecule_index", matched_original)
        return frame

    def r_group_positions(self, table: "pd.DataFrame") -> List[str]:
        """List the R-group column names present in a decomposition table.

        Parameters
        ----------
        table : pandas.DataFrame
            Output of :meth:`decompose`.

        Returns
        -------
        list of str
            e.g. ``["R1", "R2"]``.
        """
        return [c for c in table.columns if c.startswith("R") and c[1:].isdigit()]


class SARTable:
    """R-group x activity table for a congeneric series.

    Joins an R-group decomposition to measured activities so that the
    contribution of each substituent at each position can be read
    directly, and pivoted into the classic two-position SAR grid.

    Parameters
    ----------
    core : Mol or str, optional
        Passed to :class:`RGroupAnalyzer`.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mols = [Chem.MolFromSmiles(s) for s in ("c1ccccc1Cl", "c1ccccc1Br")]
    >>> table = SARTable(core="c1ccccc1").build(mols, [5.0, 6.0])
    >>> "activity" in table.columns
    True

    References
    ----------
    - Agrafiotis, D. K. et al. (2011). "SAR Maps: A New SAR
      Visualization Technique for Medicinal Chemists." J. Med. Chem.,
      50(24), 5926-5937. https://doi.org/10.1021/jm070845m
    - Wassermann, A. M. et al. (2010). J. Med. Chem., 53(23), 8209-8223.
      https://doi.org/10.1021/jm100933w
    """

    def __init__(self, core: Optional[Any] = None) -> None:
        self.core = core
        self._analyzer = RGroupAnalyzer(core=core)

    def build(
        self, mols: Sequence["Mol"], activities: Sequence[float]
    ) -> "pd.DataFrame":
        """Build the R-group + activity table.

        Parameters
        ----------
        mols : sequence of Mol
        activities : sequence of float
            Parallel activity values.

        Returns
        -------
        pandas.DataFrame
            The decomposition table with an ``activity`` column added.
        """
        if len(mols) != len(activities):
            raise ValueError(
                f"mols has length {len(mols)} but activities has {len(activities)}."
            )
        table = self._analyzer.decompose(mols)
        if table.empty:
            return table
        acts = np.asarray(activities, dtype=np.float64)
        table = table.copy()
        table["activity"] = acts[table["molecule_index"].to_numpy()]
        return table

    def pivot(
        self,
        table: "pd.DataFrame",
        row: str = "R1",
        column: str = "R2",
        aggfunc: str = "mean",
    ) -> "pd.DataFrame":
        """Pivot into the classic two-position SAR grid.

        Parameters
        ----------
        table : pandas.DataFrame
            Output of :meth:`build`.
        row, column : str
            R-group columns to use as the grid axes.
        aggfunc : str, default "mean"
            Aggregation for duplicate cells.

        Returns
        -------
        pandas.DataFrame
            Activity grid indexed by ``row`` with ``column`` as columns.
        """
        for name in (row, column):
            if name not in table.columns:
                raise ValueError(
                    f"Column {name!r} not in the SAR table; available R-groups: "
                    f"{self._analyzer.r_group_positions(table)}"
                )
        return table.pivot_table(
            index=row, columns=column, values="activity", aggfunc=aggfunc
        )

    def substituent_effects(
        self, table: "pd.DataFrame", position: str = "R1"
    ) -> "pd.DataFrame":
        """Mean activity and count per substituent at one position.

        Parameters
        ----------
        table : pandas.DataFrame
            Output of :meth:`build`.
        position : str, default "R1"
            R-group column to summarize.

        Returns
        -------
        pandas.DataFrame
            Columns ``substituent``, ``count``, ``mean_activity``,
            ``std_activity``, sorted by descending mean activity.
        """
        if position not in table.columns:
            raise ValueError(f"Column {position!r} not in the SAR table.")
        grouped = table.groupby(position)["activity"].agg(["count", "mean", "std"])
        grouped = grouped.rename(
            columns={"count": "count", "mean": "mean_activity", "std": "std_activity"}
        )
        grouped = grouped.fillna({"std_activity": 0.0})
        return (
            grouped.sort_values("mean_activity", ascending=False)
            .reset_index()
            .rename(columns={position: "substituent"})
        )


class FreeWilsonAnalysis:
    """Free-Wilson additive SAR model over R-group indicator variables.

    The original QSAR method: activity is modelled as a baseline plus an
    additive contribution from each substituent at each position,

    .. math:: A = \\mu + \\sum_{p} \\sum_{s} a_{p,s} X_{p,s}

    fitted by linear regression on one-hot indicators. It is exactly
    interpretable — each coefficient is "what this substituent is worth
    at this position, in log units" — and its residuals are themselves
    informative: large ones mark non-additive SAR, i.e. activity cliffs
    and substituent interactions the additive model cannot represent.

    Parameters
    ----------
    core : Mol or str, optional
        Passed to :class:`RGroupAnalyzer`.
    fit_intercept : bool, default True
        Whether to fit the baseline term.
    alpha : float, default 0.0
        Ridge penalty. Free-Wilson designs are often rank-deficient
        (a substituent appearing once is perfectly confounded with its
        compound), so a small positive alpha is frequently needed.

    Attributes
    ----------
    contributions_ : dict[str, dict[str, float]]
        ``{position: {substituent: contribution}}``.
    intercept_ : float
        Baseline activity.
    r2_ : float
        Coefficient of determination on the training series.
    feature_names_ : list[str]
        Names of the indicator columns, as ``"R1=Cl"``.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mols = [Chem.MolFromSmiles(s) for s in
    ...         ("c1ccccc1Cl", "c1ccccc1Br", "c1ccccc1F")]
    >>> fw = FreeWilsonAnalysis(core="c1ccccc1", alpha=0.1)
    >>> _ = fw.fit(mols, [5.0, 6.0, 4.0])
    >>> isinstance(fw.r2_, float)
    True

    References
    ----------
    - Free, S. M. & Wilson, J. W. (1964). "A Mathematical Contribution to
      Structure-Activity Studies." J. Med. Chem., 7(4), 395-399.
      https://doi.org/10.1021/jm00334a001
    - Kubinyi, H. (1988). "Free Wilson Analysis. Theory, Applications and
      its Relationship to Hansch Analysis." Quant. Struct.-Act. Relat.,
      7(3), 121-133. https://doi.org/10.1002/qsar.19880070303
    - Patel, Y., Gillet, V. J. et al. (2018). "Reinvestigating the
      Free-Wilson Approach." J. Comput. Aided Mol. Des.
      https://doi.org/10.1007/s10822-018-0116-z
    """

    contributions_: Dict[str, Dict[str, float]]
    intercept_: float
    r2_: float
    feature_names_: List[str]

    def __init__(
        self,
        core: Optional[Any] = None,
        fit_intercept: bool = True,
        alpha: float = 0.0,
    ) -> None:
        self.core = core
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self._table_builder = SARTable(core=core)

    def _design_matrix(
        self, table: "pd.DataFrame", positions: Sequence[str]
    ) -> tuple[npt.NDArray[np.float64], List[str]]:
        import pandas as pd

        dummies = pd.get_dummies(
            table[list(positions)], prefix=list(positions), prefix_sep="="
        )
        return dummies.to_numpy(dtype=np.float64), list(dummies.columns)

    def fit(
        self, mols: Sequence["Mol"], activities: Sequence[float]
    ) -> "FreeWilsonAnalysis":
        """Fit substituent contributions by linear regression.

        Parameters
        ----------
        mols : sequence of Mol
        activities : sequence of float
            Log-scale activities.

        Returns
        -------
        FreeWilsonAnalysis
            The fitted analysis.
        """
        from sklearn.linear_model import LinearRegression, Ridge

        table = self._table_builder.build(mols, activities)
        if table.empty:
            raise ValueError("R-group decomposition matched no molecules.")
        positions = RGroupAnalyzer().r_group_positions(table)
        if not positions:
            raise ValueError("R-group decomposition found no substitution points.")

        X, names = self._design_matrix(table, positions)
        y = table["activity"].to_numpy(dtype=np.float64)

        model: Any = (
            Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)
            if self.alpha > 0
            else LinearRegression(fit_intercept=self.fit_intercept)
        )
        model.fit(X, y)

        contributions: Dict[str, Dict[str, float]] = {p: {} for p in positions}
        for name, coef in zip(names, np.asarray(model.coef_, dtype=np.float64).ravel()):
            position, _, substituent = name.partition("=")
            contributions.setdefault(position, {})[substituent] = float(coef)

        self.contributions_ = contributions
        self.intercept_ = float(model.intercept_) if self.fit_intercept else 0.0
        self.r2_ = float(model.score(X, y))
        self.feature_names_ = names
        self._model = model
        self._positions = positions
        self._table = table
        return self

    def predict(self, mols: Sequence["Mol"]) -> npt.NDArray[np.float64]:
        """Predict activity for new analogues of the same core.

        Parameters
        ----------
        mols : sequence of Mol
            Molecules sharing the fitted core.

        Returns
        -------
        ndarray of shape (n_matched,)
            Predictions for the molecules that matched the core.
        """
        import pandas as pd

        if not hasattr(self, "_model"):
            from qsarkit.base.exceptions import ModelNotFittedError

            raise ModelNotFittedError(
                "FreeWilsonAnalysis must be fitted before calling predict()."
            )
        table = RGroupAnalyzer(core=self.core).decompose(mols)
        if table.empty:
            return np.zeros(0, dtype=np.float64)
        dummies = pd.get_dummies(
            table[list(self._positions)],
            prefix=list(self._positions),
            prefix_sep="=",
        )
        # Align to the training design, filling unseen substituents with 0.
        dummies = dummies.reindex(columns=self.feature_names_, fill_value=0)
        return np.asarray(
            self._model.predict(dummies.to_numpy(dtype=np.float64)), dtype=np.float64
        )

    def residuals(self) -> "pd.DataFrame":
        """Training residuals — large values flag non-additive SAR.

        Returns
        -------
        pandas.DataFrame
            Columns ``molecule_index``, ``observed``, ``predicted``,
            ``residual``, sorted by descending absolute residual.
        """
        if not hasattr(self, "_model"):
            from qsarkit.base.exceptions import ModelNotFittedError

            raise ModelNotFittedError(
                "FreeWilsonAnalysis must be fitted before calling residuals()."
            )
        import pandas as pd

        X, _ = self._design_matrix(self._table, self._positions)
        observed = self._table["activity"].to_numpy(dtype=np.float64)
        predicted = np.asarray(self._model.predict(X), dtype=np.float64)
        frame = pd.DataFrame(
            {
                "molecule_index": self._table["molecule_index"].to_numpy(),
                "observed": observed,
                "predicted": predicted,
                "residual": observed - predicted,
            }
        )
        return frame.reindex(
            frame["residual"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)

    def to_dataframe(self) -> "pd.DataFrame":
        """Substituent contributions as a tidy table.

        Returns
        -------
        pandas.DataFrame
            Columns ``position``, ``substituent``, ``contribution``,
            sorted by descending contribution.
        """
        if not hasattr(self, "contributions_"):
            from qsarkit.base.exceptions import ModelNotFittedError

            raise ModelNotFittedError(
                "FreeWilsonAnalysis must be fitted before calling to_dataframe()."
            )
        import pandas as pd

        rows = [
            {"position": position, "substituent": substituent, "contribution": value}
            for position, group in self.contributions_.items()
            for substituent, value in group.items()
        ]
        frame = pd.DataFrame(
            rows, columns=["position", "substituent", "contribution"]
        )
        if not frame.empty:
            frame = frame.sort_values("contribution", ascending=False).reset_index(
                drop=True
            )
        return frame
