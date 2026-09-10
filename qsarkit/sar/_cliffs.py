"""Activity cliff detection and structure-activity landscape analysis."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from qsarkit.neighbors import tanimoto_similarity_matrix

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    import plotly.graph_objects as go
    from rdkit.Chem import Mol

__all__ = [
    "ActivityCliff",
    "ActivityCliffDetector",
    "SALIAnalyzer",
    "SARIAnalyzer",
    "ActivityLandscapePlotter",
    "activity_cliff_report",
]

_SimilarityMethod = Literal["fingerprint", "scaffold", "mmp"]


@dataclass(frozen=True)
class ActivityCliff:
    """A pair of similar molecules with a large activity difference.

    Attributes
    ----------
    index_a, index_b : int
        Positions of the two molecules in the input sequence.
    mol_a, mol_b : Mol
        The two molecules.
    similarity : float
        Structural similarity in [0, 1].
    activity_a, activity_b : float
        Their activities on a logarithmic scale (e.g. pIC50).
    delta : float
        ``abs(activity_a - activity_b)``.
    sali : float
        Structure-Activity Landscape Index for the pair.
    """

    index_a: int
    index_b: int
    mol_a: "Mol"
    mol_b: "Mol"
    similarity: float
    activity_a: float
    activity_b: float
    delta: float
    sali: float

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"ActivityCliff({self.index_a}<->{self.index_b}, "
            f"sim={self.similarity:.2f}, delta={self.delta:.2f}, "
            f"SALI={self.sali:.1f})"
        )


def _fingerprint_matrix(
    mols: Sequence["Mol"], radius: int = 2, n_bits: int = 2048
) -> npt.NDArray[np.float64]:
    """ECFP bit matrix for a molecule sequence."""
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return np.array(
        [list(gen.GetFingerprint(m)) for m in mols], dtype=np.float64
    )


def _scaffold_similarity_matrix(mols: Sequence["Mol"]) -> npt.NDArray[np.float64]:
    """1.0 where two molecules share a Bemis-Murcko scaffold, else 0.0."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffolds = [
        Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) for m in mols
    ]
    n = len(scaffolds)
    sim = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            sim[i, j] = 1.0 if scaffolds[i] == scaffolds[j] else 0.0
    return sim


def _mmp_similarity_matrix(mols: Sequence["Mol"], **kwargs: Any) -> npt.NDArray[np.float64]:
    """1.0 for molecules forming a matched molecular pair, else 0.0."""
    from qsarkit.sar._mmp import MatchedMolecularPairs

    n = len(mols)
    sim = np.eye(n, dtype=np.float64)
    for pair in MatchedMolecularPairs(**kwargs).find_pairs(mols):
        sim[pair.index_a, pair.index_b] = 1.0
        sim[pair.index_b, pair.index_a] = 1.0
    return sim


def _similarity_matrix(
    mols: Sequence["Mol"],
    method: _SimilarityMethod,
    radius: int,
    n_bits: int,
) -> npt.NDArray[np.float64]:
    if method == "fingerprint":
        return tanimoto_similarity_matrix(_fingerprint_matrix(mols, radius, n_bits))
    if method == "scaffold":
        return _scaffold_similarity_matrix(mols)
    if method == "mmp":
        return _mmp_similarity_matrix(mols)
    raise ValueError(
        f"method must be 'fingerprint', 'scaffold' or 'mmp', got {method!r}."
    )


def _sali_matrix(
    similarity: npt.NDArray[np.float64], activities: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """SALI = |Ai - Aj| / (1 - sim(i,j)), with identical structures -> inf."""
    delta = np.abs(activities[:, None] - activities[None, :])
    denom = 1.0 - similarity
    with np.errstate(divide="ignore", invalid="ignore"):
        sali = np.where(denom > 0, delta / np.where(denom > 0, denom, 1.0), np.inf)
    # A molecule against itself has zero activity difference, not a cliff.
    np.fill_diagonal(sali, 0.0)
    return np.asarray(sali, dtype=np.float64)


class ActivityCliffDetector:
    """Detect activity cliffs: similar structures with very different activity.

    Activity cliffs are the single biggest obstacle to QSAR: they violate
    the similarity-property principle that regression models rely on, and
    a model that cannot reproduce them will systematically mispredict the
    most interesting compounds in a series. Detecting them tells you both
    where a model will fail and where the SAR carries real information.

    A pair ``(i, j)`` is a cliff when
    ``similarity(i, j) >= similarity_threshold`` and
    ``|activity_i - activity_j| >= activity_threshold``.

    Parameters
    ----------
    similarity_threshold : float, default 0.85
        Minimum structural similarity. 0.85 on ECFP4 is the conventional
        cutoff in the activity-cliff literature.
    activity_threshold : float, default 2.0
        Minimum absolute activity difference, in log units. 2.0 means a
        100-fold potency change.
    method : {"fingerprint", "scaffold", "mmp"}, default "fingerprint"
        How structural similarity is measured. ``"scaffold"`` and
        ``"mmp"`` give binary similarity (1.0 for same scaffold / a
        matched pair), so with those the similarity threshold acts as a
        simple on/off test.
    radius : int, default 2
        Morgan radius (ECFP4 = radius 2) for the fingerprint method.
    n_bits : int, default 2048
        Fingerprint length for the fingerprint method.

    Examples
    --------
    A 4-Cl / 4-Br swap on the same anilide core, four log units apart:

    >>> from rdkit import Chem
    >>> mols = [Chem.MolFromSmiles(s) for s in
    ...         ("CC(=O)Nc1ccc(Cl)cc1", "CC(=O)Nc1ccc(Br)cc1")]
    >>> detector = ActivityCliffDetector(similarity_threshold=0.6)
    >>> cliffs = detector.detect(mols, [9.0, 5.0])
    >>> len(cliffs)
    1
    >>> round(cliffs[0].similarity, 3), cliffs[0].delta
    (0.615, 4.0)

    Note how low that similarity is for a single-atom change. Morgan
    fingerprints of small molecules score far below intuition, because one
    substituent alters every atom environment within ``radius`` bonds of
    it. A threshold of 0.85 -- the usual figure quoted for cliff analysis,
    and this class's default -- is calibrated for drug-sized molecules with
    a large shared core, and will find nothing in a set of fragments.

    References
    ----------
    - Maggiora, G. M. (2006). "On Outliers and Activity Cliffs - Why QSAR
      Often Disappoints." J. Chem. Inf. Model., 46(4), 1535.
      https://doi.org/10.1021/ci060117s
    - Stumpfe, D. & Bajorath, J. (2012). "Exploring Activity Cliffs in
      Medicinal Chemistry." J. Med. Chem., 55(7), 2932-2942.
      https://doi.org/10.1021/jm300288g
    - Stumpfe, D., Hu, H. & Bajorath, J. (2019). "Evolving Concept of
      Activity Cliffs." ACS Omega / J. Med. Chem., 62(5), 2354-2363.
      https://doi.org/10.1021/acs.jmedchem.9b00004
    - Cruz-Monteagudo, M. et al. (2014). "Activity Cliffs in Drug
      Discovery: Dr Jekyll or Mr Hyde?" Drug Discov. Today, 19(8),
      1069-1080. https://doi.org/10.1016/j.drudis.2014.02.003
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        activity_threshold: float = 2.0,
        method: _SimilarityMethod = "fingerprint",
        radius: int = 2,
        n_bits: int = 2048,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.activity_threshold = activity_threshold
        self.method = method
        self.radius = radius
        self.n_bits = n_bits

    def similarity_matrix(self, mols: Sequence["Mol"]) -> npt.NDArray[np.float64]:
        """Pairwise structural similarity under the configured method.

        Parameters
        ----------
        mols : sequence of Mol

        Returns
        -------
        ndarray of shape (n, n)
        """
        return _similarity_matrix(mols, self.method, self.radius, self.n_bits)

    def detect(
        self, mols: Sequence["Mol"], activities: Sequence[float]
    ) -> List[ActivityCliff]:
        """Find every activity cliff in a dataset.

        Parameters
        ----------
        mols : sequence of Mol
            Molecules, all non-None.
        activities : sequence of float
            Activities on a logarithmic scale (pIC50, pKi, ...). Using a
            linear scale here would make the threshold meaningless.

        Returns
        -------
        list of ActivityCliff
            Sorted by descending SALI, so the sharpest cliffs come first.
        """
        if len(mols) != len(activities):
            raise ValueError(
                f"mols has length {len(mols)} but activities has {len(activities)}."
            )
        if len(mols) < 2:
            return []

        acts = np.asarray(activities, dtype=np.float64)
        sim = self.similarity_matrix(mols)
        sali = _sali_matrix(sim, acts)
        delta = np.abs(acts[:, None] - acts[None, :])

        iu = np.triu_indices(len(mols), k=1)
        mask = (sim[iu] >= self.similarity_threshold) & (
            delta[iu] >= self.activity_threshold
        )
        cliffs = [
            ActivityCliff(
                index_a=int(i),
                index_b=int(j),
                mol_a=mols[i],
                mol_b=mols[j],
                similarity=float(sim[i, j]),
                activity_a=float(acts[i]),
                activity_b=float(acts[j]),
                delta=float(delta[i, j]),
                sali=float(sali[i, j]),
            )
            for i, j, keep in zip(iu[0], iu[1], mask)
            if keep
        ]
        cliffs.sort(key=lambda c: c.sali, reverse=True)
        return cliffs

    def to_dataframe(self, cliffs: Sequence[ActivityCliff]) -> "pd.DataFrame":
        """Render detected cliffs as a table.

        Parameters
        ----------
        cliffs : sequence of ActivityCliff

        Returns
        -------
        pandas.DataFrame
        """
        import pandas as pd
        from rdkit import Chem

        return pd.DataFrame(
            [
                {
                    "index_a": c.index_a,
                    "index_b": c.index_b,
                    "smiles_a": Chem.MolToSmiles(c.mol_a),
                    "smiles_b": Chem.MolToSmiles(c.mol_b),
                    "similarity": c.similarity,
                    "activity_a": c.activity_a,
                    "activity_b": c.activity_b,
                    "delta": c.delta,
                    "sali": c.sali,
                }
                for c in cliffs
            ],
            columns=[
                "index_a", "index_b", "smiles_a", "smiles_b", "similarity",
                "activity_a", "activity_b", "delta", "sali",
            ],
        )


class SALIAnalyzer:
    """Structure-Activity Landscape Index (SALI) analysis.

    SALI quantifies how sharply activity changes with structure::

        SALI(i, j) = |A_i - A_j| / (1 - sim(i, j))

    Large values mark cliffs — small structural change, large activity
    change. Beyond the pairwise matrix, the SALI *curve* scores how well
    a model reproduces the landscape: pairs are ranked by true SALI and
    by predicted SALI, and the fraction of top-ranked true pairs the
    model also ranks highly gives a curve whose area (in [0, 1], 1 =
    perfect) is a landscape-aware model-quality metric that ordinary
    RMSE/R2 completely miss.

    Parameters
    ----------
    method : {"fingerprint", "scaffold", "mmp"}, default "fingerprint"
        Similarity backend.
    radius : int, default 2
        Morgan radius for the fingerprint method.
    n_bits : int, default 2048
        Fingerprint length for the fingerprint method.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mols = [Chem.MolFromSmiles(s) for s in ("CCO", "CCC", "CCN")]
    >>> analyzer = SALIAnalyzer()
    >>> S = analyzer.sali_matrix(mols, [5.0, 6.0, 7.0])
    >>> S.shape
    (3, 3)

    References
    ----------
    - Guha, R. & Van Drie, J. H. (2008). "Structure-Activity Landscape
      Index: Identifying and Quantifying Activity Cliffs."
      J. Chem. Inf. Model., 48(3), 646-658.
      https://doi.org/10.1021/ci7004093
    - Guha, R. (2012). "Exploring Structure-Activity Data Using the
      Landscape Paradigm." WIREs Comput. Mol. Sci. / J. Chem. Inf.
      Model., 52(8), 2181-2191. https://doi.org/10.1021/ci300047k
    - Guha, R. & Van Drie, J. H. (2008). "Assessing How Well a Modeling
      Protocol Captures a Structure-Activity Landscape."
      J. Chem. Inf. Model., 48(8), 1716-1728.
      https://doi.org/10.1021/ci8001414
    """

    def __init__(
        self,
        method: _SimilarityMethod = "fingerprint",
        radius: int = 2,
        n_bits: int = 2048,
    ) -> None:
        self.method = method
        self.radius = radius
        self.n_bits = n_bits

    def sali_matrix(
        self, mols: Sequence["Mol"], activities: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        """Pairwise SALI matrix.

        Parameters
        ----------
        mols : sequence of Mol
        activities : array-like of float
            Log-scale activities.

        Returns
        -------
        ndarray of shape (n, n)
            Symmetric, zero diagonal, ``inf`` where two distinct
            molecules have identical structure fingerprints.
        """
        acts = np.asarray(activities, dtype=np.float64)
        if len(mols) != acts.shape[0]:
            raise ValueError(
                f"mols has length {len(mols)} but activities has {acts.shape[0]}."
            )
        sim = _similarity_matrix(mols, self.method, self.radius, self.n_bits)
        return _sali_matrix(sim, acts)

    def sali_network(
        self,
        mols: Sequence["Mol"],
        activities: Sequence[float],
        percentile: float = 95.0,
    ) -> Any:
        """Build a graph of the highest-SALI pairs.

        Parameters
        ----------
        mols : sequence of Mol
        activities : sequence of float
        percentile : float, default 95.0
            Keep edges whose SALI is at or above this percentile of the
            finite SALI values.

        Returns
        -------
        networkx.Graph
            Nodes carry ``activity``; edges carry ``sali``.
        """
        import networkx as nx

        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"percentile must be in [0, 100], got {percentile}.")

        sali = self.sali_matrix(mols, activities)
        iu = np.triu_indices(len(mols), k=1)
        values = sali[iu]
        finite = values[np.isfinite(values)]
        cutoff = float(np.percentile(finite, percentile)) if finite.size else 0.0

        graph = nx.Graph()
        for i, activity in enumerate(activities):
            graph.add_node(i, activity=float(activity))
        for i, j, value in zip(iu[0], iu[1], values):
            if value >= cutoff:
                graph.add_edge(int(i), int(j), sali=float(value))
        return graph

    def sali_curve(
        self,
        mols: Sequence["Mol"],
        y_true: Sequence[float],
        y_pred: Sequence[float],
        n_points: int = 50,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """SALI curve comparing true and predicted activity landscapes.

        For each cutoff ``X`` (fraction of the highest-SALI true pairs),
        the curve reports the fraction of those pairs whose activity
        *ordering* the model reproduces.

        Parameters
        ----------
        mols : sequence of Mol
        y_true : sequence of float
            Observed activities.
        y_pred : sequence of float
            Predicted activities.
        n_points : int, default 50
            Number of cutoffs sampled along the curve.

        Returns
        -------
        x : ndarray of shape (n_points,)
            Fraction of top-SALI pairs considered, in (0, 1].
        y : ndarray of shape (n_points,)
            Fraction of those pairs ordered correctly, in [0, 1].
        """
        true = np.asarray(y_true, dtype=np.float64)
        pred = np.asarray(y_pred, dtype=np.float64)
        if not (len(mols) == len(true) == len(pred)):
            raise ValueError("mols, y_true and y_pred must all have the same length.")

        sali = self.sali_matrix(mols, true)
        iu = np.triu_indices(len(mols), k=1)
        values = sali[iu]
        finite_mask = np.isfinite(values)
        idx_a, idx_b = iu[0][finite_mask], iu[1][finite_mask]
        values = values[finite_mask]
        if values.size == 0:
            return np.zeros(0), np.zeros(0)

        order = np.argsort(-values)
        idx_a, idx_b = idx_a[order], idx_b[order]
        true_sign = np.sign(true[idx_a] - true[idx_b])
        pred_sign = np.sign(pred[idx_a] - pred[idx_b])
        correct = (true_sign == pred_sign).astype(np.float64)

        n_pairs = len(correct)
        counts = np.unique(
            np.clip(
                np.linspace(1, n_pairs, num=min(n_points, n_pairs)).astype(int), 1, n_pairs
            )
        )
        x = counts / n_pairs
        y = np.array([correct[:k].mean() for k in counts], dtype=np.float64)
        return x, y

    def sali_auc(
        self,
        mols: Sequence["Mol"],
        y_true: Sequence[float],
        y_pred: Sequence[float],
        n_points: int = 50,
    ) -> float:
        """Area under the SALI curve — a landscape-aware model score.

        Parameters
        ----------
        mols : sequence of Mol
        y_true, y_pred : sequence of float
        n_points : int, default 50

        Returns
        -------
        float
            Area in [0, 1]; 1.0 means every cliff's direction is
            predicted correctly, 0.5 is chance.
        """
        x, y = self.sali_curve(mols, y_true, y_pred, n_points)
        if x.size < 2:
            return float(y[0]) if y.size else 0.0
        return float(np.trapezoid(y, x) / (x[-1] - x[0]))


class SARIAnalyzer:
    """Structure-Activity Relationship Index (SARI): continuity vs discontinuity.

    SARI scores a compound set on two orthogonal axes and combines them::

        SARI = 0.5 * ((1 - continuity_norm) + discontinuity_norm)

    The *continuity* score reflects smooth, gradual SAR (similar
    molecules with similar potency, weighted by potency); the
    *discontinuity* score reflects cliffs (similar molecules with very
    different potency). A series can be high in both — a "heterogeneous"
    SAR that is smooth in one region and cliff-ridden in another.

    Parameters
    ----------
    similarity_threshold : float, default 0.6
        Minimum similarity for a pair to contribute to the
        discontinuity term.
    reference_delta : float, default 3.0
        Activity difference, in log units, treated as maximally
        discontinuous when normalizing the discontinuity score. Fixing
        this on an absolute scale (rather than the dataset's own range)
        is what makes SARI comparable between series.
    radius : int, default 2
        Morgan radius.
    n_bits : int, default 2048
        Fingerprint length.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mols = [Chem.MolFromSmiles(s) for s in ("CCO", "CCC", "CCN", "CCCl")]
    >>> scores = SARIAnalyzer().analyze(mols, [5.0, 5.2, 5.1, 8.0])
    >>> set(scores) == {"continuity", "discontinuity", "sari"}
    True

    References
    ----------
    - Peltason, L. & Bajorath, J. (2007). "SAR Index: Quantifying the
      Nature of Structure-Activity Relationships." J. Med. Chem., 50(23),
      5571-5578. https://doi.org/10.1021/jm070562u
    - Wassermann, A. M., Wawer, M. & Bajorath, J. (2010). "Activity
      Landscape Representations for Structure-Activity Relationship
      Analysis." J. Med. Chem., 53(23), 8209-8223.
      https://doi.org/10.1021/jm100933w
    """

    def __init__(
        self,
        similarity_threshold: float = 0.6,
        reference_delta: float = 3.0,
        radius: int = 2,
        n_bits: int = 2048,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.reference_delta = reference_delta
        self.radius = radius
        self.n_bits = n_bits

    def analyze(
        self, mols: Sequence["Mol"], activities: Sequence[float]
    ) -> Dict[str, float]:
        """Compute continuity, discontinuity and the combined SARI score.

        Parameters
        ----------
        mols : sequence of Mol
        activities : sequence of float
            Log-scale activities.

        Returns
        -------
        dict
            Keys ``continuity``, ``discontinuity``, ``sari``.
        """
        if len(mols) != len(activities):
            raise ValueError(
                f"mols has length {len(mols)} but activities has {len(activities)}."
            )
        if len(mols) < 2:
            return {"continuity": 0.0, "discontinuity": 0.0, "sari": 0.0}

        acts = np.asarray(activities, dtype=np.float64)
        sim = tanimoto_similarity_matrix(
            _fingerprint_matrix(mols, self.radius, self.n_bits)
        )
        iu = np.triu_indices(len(mols), k=1)
        sim_pairs = sim[iu]
        delta_pairs = np.abs(acts[iu[0]] - acts[iu[1]])

        # Continuity: potency-weighted mean similarity over pairs whose
        # potency differs little; high when SAR changes gradually.
        weights = (acts[iu[0]] + acts[iu[1]]) / 2.0
        denom = np.sum(weights * (1.0 + delta_pairs))
        continuity = (
            float(np.sum(weights * sim_pairs) / denom) if denom > 0 else 0.0
        )

        # Discontinuity: mean potency difference among similar pairs.
        similar = sim_pairs >= self.similarity_threshold
        discontinuity = (
            float(np.mean(delta_pairs[similar] * sim_pairs[similar]))
            if similar.any()
            else 0.0
        )

        # Normalize against a fixed reference of `reference_delta` log
        # units rather than the dataset's own maximum: dividing by the
        # observed max would make the score scale-invariant, so a series
        # whose activities span 0.1 log units would score as "highly
        # discontinuous" as one spanning 4. The index must be comparable
        # across datasets, which requires an absolute yardstick.
        cont_norm = float(np.clip(continuity, 0.0, 1.0))
        disc_norm = float(np.clip(discontinuity / self.reference_delta, 0.0, 1.0))
        return {
            "continuity": cont_norm,
            "discontinuity": disc_norm,
            "sari": 0.5 * ((1.0 - cont_norm) + disc_norm),
        }


class ActivityLandscapePlotter:
    """Structure-Activity Similarity (SAS) map data and Plotly figure.

    A SAS map plots every compound pair as (structure similarity,
    activity similarity) and reads the four quadrants as distinct SAR
    regimes:

    ================ ================= ==========================
    Structure sim.   Activity sim.     Interpretation
    ================ ================= ==========================
    high             high              smooth / continuous SAR
    high             low               **activity cliff**
    low              high              scaffold hop
    low              low               nondescript
    ================ ================= ==========================

    Parameters
    ----------
    similarity_threshold : float, default 0.6
        Structure-similarity boundary between the left and right halves.
    activity_threshold : float, default 0.6
        Activity-similarity boundary between the top and bottom halves.
    radius : int, default 2
        Morgan radius.
    n_bits : int, default 2048
        Fingerprint length.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mols = [Chem.MolFromSmiles(s) for s in ("CCO", "CCC", "CCN")]
    >>> df = ActivityLandscapePlotter().sas_data(mols, [5.0, 7.0, 5.1])
    >>> sorted(df.columns)
    ['activity_similarity', 'delta_activity', 'index_a', 'index_b',
     'quadrant', 'structure_similarity']

    One row per pair, each assigned to a quadrant of the SAS map:

    >>> len(df)               # three pairs from three molecules
    3
    >>> sorted(set(df["quadrant"]))
    ['nondescript', 'scaffold hop']

    References
    ----------
    - Shanmugasundaram, V. & Maggiora, G. M. (2001). "Characterizing
      Property and Activity Landscapes Using an Information-Theoretic
      Approach." 222nd ACS National Meeting, CINF 77.
    - Wassermann, A. M., Wawer, M. & Bajorath, J. (2010). J. Med. Chem.,
      53(23), 8209-8223. https://doi.org/10.1021/jm100933w
    - Perez-Villanueva, J. et al. (2011). "Comparison of Multiple 2D
      Representations for the Activity Landscape Modeling."
      Bioorg. Med. Chem., 19(21), 6183-6193.
      https://doi.org/10.1016/j.bmc.2011.09.024
    """

    def __init__(
        self,
        similarity_threshold: float = 0.6,
        activity_threshold: float = 0.6,
        radius: int = 2,
        n_bits: int = 2048,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.activity_threshold = activity_threshold
        self.radius = radius
        self.n_bits = n_bits

    def sas_data(
        self, mols: Sequence["Mol"], activities: Sequence[float]
    ) -> "pd.DataFrame":
        """Compute the SAS-map table (one row per compound pair).

        Parameters
        ----------
        mols : sequence of Mol
        activities : sequence of float

        Returns
        -------
        pandas.DataFrame
            Columns ``index_a``, ``index_b``, ``structure_similarity``,
            ``activity_similarity``, ``delta_activity``, ``quadrant``.
            Activity similarity is ``1 - |dA| / max|dA|``.
        """
        import pandas as pd

        if len(mols) != len(activities):
            raise ValueError(
                f"mols has length {len(mols)} but activities has {len(activities)}."
            )
        acts = np.asarray(activities, dtype=np.float64)
        sim = tanimoto_similarity_matrix(
            _fingerprint_matrix(mols, self.radius, self.n_bits)
        )
        iu = np.triu_indices(len(mols), k=1)
        struct_sim = sim[iu]
        delta = np.abs(acts[iu[0]] - acts[iu[1]])
        max_delta = float(delta.max()) if delta.size else 0.0
        act_sim = 1.0 - (delta / max_delta) if max_delta > 0 else np.ones_like(delta)

        quadrants = [
            self._quadrant(s, a) for s, a in zip(struct_sim, act_sim)
        ]
        return pd.DataFrame(
            {
                "index_a": iu[0],
                "index_b": iu[1],
                "structure_similarity": struct_sim,
                "activity_similarity": act_sim,
                "delta_activity": delta,
                "quadrant": quadrants,
            }
        )

    def _quadrant(self, structure_sim: float, activity_sim: float) -> str:
        high_struct = structure_sim >= self.similarity_threshold
        high_act = activity_sim >= self.activity_threshold
        if high_struct and high_act:
            return "smooth SAR"
        if high_struct and not high_act:
            return "activity cliff"
        if not high_struct and high_act:
            return "scaffold hop"
        return "nondescript"

    def plot(
        self, mols: Sequence["Mol"], activities: Sequence[float]
    ) -> "go.Figure":
        """Render the SAS map as a Plotly scatter with quadrant guides.

        Parameters
        ----------
        mols : sequence of Mol
        activities : sequence of float

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go

        data = self.sas_data(mols, activities)
        fig = go.Figure()
        for quadrant, group in data.groupby("quadrant"):
            fig.add_trace(
                go.Scatter(
                    x=group["structure_similarity"],
                    y=group["activity_similarity"],
                    mode="markers",
                    name=str(quadrant),
                    text=[
                        f"{a} vs {b}<br>dA = {d:.2f}"
                        for a, b, d in zip(
                            group["index_a"], group["index_b"], group["delta_activity"]
                        )
                    ],
                    hovertemplate="%{text}<extra></extra>",
                )
            )
        fig.add_vline(x=self.similarity_threshold, line_dash="dash", line_width=1)
        fig.add_hline(y=self.activity_threshold, line_dash="dash", line_width=1)
        fig.update_layout(
            title="Structure-Activity Similarity (SAS) map",
            xaxis_title="Structure similarity (Tanimoto)",
            yaxis_title="Activity similarity",
            xaxis_range=[0, 1],
            yaxis_range=[0, 1],
        )
        return fig


def activity_cliff_report(
    mols: Sequence["Mol"],
    activities: Sequence[float],
    similarity_threshold: float = 0.85,
    activity_threshold: float = 2.0,
    top_n: int = 10,
) -> Dict[str, Any]:
    """Summarize the activity-cliff content of a dataset.

    A one-call diagnostic to run before modeling: a high cliff ratio
    predicts that a regression model will underperform on this series
    no matter how it is tuned, and points at which scaffolds and which
    substituent changes are responsible.

    Parameters
    ----------
    mols : sequence of Mol
        Molecules.
    activities : sequence of float
        Log-scale activities.
    similarity_threshold : float, default 0.85
        Passed to :class:`ActivityCliffDetector`.
    activity_threshold : float, default 2.0
        Passed to :class:`ActivityCliffDetector`.
    top_n : int, default 10
        How many top scaffolds/transformations/cliffs to report.

    Returns
    -------
    dict
        ``n_compounds``, ``n_pairs``, ``n_cliffs``, ``cliff_ratio``
        (cliffs / all pairs), ``cliff_compound_fraction`` (fraction of
        compounds involved in at least one cliff), ``max_sali``,
        ``top_cliffs``, ``top_scaffolds`` (scaffold SMILES -> cliff
        count), ``top_transformations`` (MMP transformation -> cliff
        count), and ``sari``.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mols = [Chem.MolFromSmiles(s) for s in
    ...         ("CC(=O)Nc1ccc(Cl)cc1", "CC(=O)Nc1ccc(Br)cc1")]
    >>> report = activity_cliff_report(mols, [9.0, 5.0], similarity_threshold=0.6)
    >>> report["n_cliffs"]
    1
    >>> report["cliff_ratio"]        # one cliff out of one pair
    1.0

    References
    ----------
    - Stumpfe, D. & Bajorath, J. (2012). J. Med. Chem., 55(7), 2932-2942.
      https://doi.org/10.1021/jm300288g
    - Guha, R. & Van Drie, J. H. (2008). J. Chem. Inf. Model., 48(3),
      646-658. https://doi.org/10.1021/ci7004093
    - van Tilborg, D., Alenicheva, A. & Grisoni, F. (2022). "Exposing the
      Limitations of Molecular Machine Learning with Activity Cliffs."
      J. Chem. Inf. Model., 62(23), 5938-5951.
      https://doi.org/10.1021/acs.jcim.2c01073
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    from qsarkit.sar._mmp import MatchedMolecularPairs

    detector = ActivityCliffDetector(
        similarity_threshold=similarity_threshold,
        activity_threshold=activity_threshold,
    )
    cliffs = detector.detect(mols, activities)
    n = len(mols)
    n_pairs = n * (n - 1) // 2

    involved: set[int] = set()
    scaffold_counts: Counter[str] = Counter()
    for cliff in cliffs:
        involved.update((cliff.index_a, cliff.index_b))
        for mol in (cliff.mol_a, cliff.mol_b):
            scaffold_counts[
                Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
            ] += 1

    cliff_pairs = {(c.index_a, c.index_b) for c in cliffs}
    transformation_counts: Counter[str] = Counter()
    for pair in MatchedMolecularPairs().find_pairs(list(mols)):
        key = (min(pair.index_a, pair.index_b), max(pair.index_a, pair.index_b))
        if key in cliff_pairs:
            transformation_counts[pair.transformation] += 1

    finite_sali = [c.sali for c in cliffs if np.isfinite(c.sali)]
    return {
        "n_compounds": n,
        "n_pairs": n_pairs,
        "n_cliffs": len(cliffs),
        "cliff_ratio": len(cliffs) / n_pairs if n_pairs else 0.0,
        "cliff_compound_fraction": len(involved) / n if n else 0.0,
        "max_sali": max(finite_sali) if finite_sali else 0.0,
        "top_cliffs": cliffs[:top_n],
        "top_scaffolds": dict(scaffold_counts.most_common(top_n)),
        "top_transformations": dict(transformation_counts.most_common(top_n)),
        "sari": SARIAnalyzer().analyze(mols, activities),
    }
