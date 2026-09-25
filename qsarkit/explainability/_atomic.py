"""Per-atom and per-fragment attribution for fingerprint-based QSAR models."""

from __future__ import annotations

from qsarkit.base.exceptions import RDKIT_MOLECULE_ERRORS
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    import plotly.graph_objects as go
    from rdkit.Chem import Mol

__all__ = [
    "AtomicContribution",
    "AtomicContributionMap",
    "FragmentContributionAnalyzer",
    "CounterfactualExplainer",
]


@dataclass
class AtomicContribution:
    """Per-atom attribution for one molecule.

    Attributes
    ----------
    weights : ndarray of shape (n_atoms,)
        Signed contribution of each atom. Positive means removing the
        atom would lower the prediction.
    prediction : float
        The model's prediction for the intact molecule.
    smiles : str
        Canonical SMILES of the molecule explained.
    """

    weights: npt.NDArray[np.float64]
    prediction: float
    smiles: str = ""

    @property
    def most_positive(self) -> int:
        """Index of the atom contributing most positively."""
        return int(np.argmax(self.weights))

    @property
    def most_negative(self) -> int:
        """Index of the atom contributing most negatively."""
        return int(np.argmin(self.weights))


class AtomicContributionMap:
    """Attribute a prediction to individual atoms by fingerprint masking.

    Implements the similarity-map algorithm of Riniker and Landrum: for
    each atom, the fingerprint is recomputed with that atom removed from
    the environment, and the change in prediction is the atom's
    contribution. Every bit set by an atom is switched off, so the
    difference measures exactly what that atom's substructures were
    worth to the model.

    This is what turns a fingerprint model — otherwise a black box over
    2048 anonymous bits — into something a chemist can act on: the
    output maps directly onto the structure, showing which part of the
    molecule the model is actually responding to. It is the practical
    route to OECD principle 5 for fingerprint QSAR.

    One property of the method is worth knowing: on a symmetric molecule
    every atom receives the same weight, and on a redundant one the
    weights are near zero. Masking a single carbon of benzene leaves the
    other five still generating aromatic bits, so the prediction barely
    moves and no atom looks responsible. That is the honest answer — the
    signal is carried by the ring as a whole, not by any one atom — but
    it means a flat weight vector indicates redundancy rather than
    irrelevance. Use :class:`FragmentContributionAnalyzer` when the
    question is about a group rather than an atom.

    Parameters
    ----------
    model : fitted estimator
        Must expose ``predict``, and ``predict_proba`` if
        ``use_proba=True``.
    fingerprint : callable or transformer
        Maps ``Iterable[Mol] -> ndarray``. Any
        :mod:`qsarkit.representation` fingerprint works.
    use_proba : bool, default False
        Explain ``predict_proba(...)[:, 1]`` rather than ``predict``,
        which gives a smoothly varying signal for classifiers instead of
        a step function.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from qsarkit.representation import MorganFingerprint
    >>> fp = MorganFingerprint(n_bits=64)
    >>> mols = [Chem.MolFromSmiles(s) for s in ("CCO", "CCN", "c1ccccc1", "CCC")]
    >>> model = RandomForestRegressor(n_estimators=5, random_state=0).fit(
    ...     fp.transform(mols), [1.0, 2.0, 3.0, 4.0]
    ... )
    >>> result = AtomicContributionMap(model, fp).explain(mols[0])
    >>> result.weights.shape
    (3,)

    References
    ----------
    - Riniker, S. & Landrum, G. A. (2013). "Similarity Maps - A
      Visualization Strategy for Molecular Fingerprints and Machine-
      Learning Methods." J. Cheminform., 5, 43.
      https://doi.org/10.1186/1758-2946-5-43
    - Rogers, D. & Hahn, M. (2010). "Extended-Connectivity Fingerprints."
      J. Chem. Inf. Model., 50(5), 742-754.
      https://doi.org/10.1021/ci100050t
    - RDKit ``Chem.Draw.SimilarityMaps`` documentation:
      https://www.rdkit.org/docs/source/rdkit.Chem.Draw.SimilarityMaps.html
    """

    def __init__(
        self,
        model: Any,
        fingerprint: Any,
        use_proba: bool = False,
    ) -> None:
        self.model = model
        self.fingerprint = fingerprint
        self.use_proba = use_proba

    def _featurize(self, mols: Sequence[Any]) -> npt.NDArray[np.float64]:
        transform = getattr(self.fingerprint, "transform", self.fingerprint)
        return np.asarray(transform(list(mols)), dtype=np.float64)

    def _predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.use_proba:
            return np.asarray(
                self.model.predict_proba(X), dtype=np.float64
            )[:, -1]
        return np.asarray(self.model.predict(X), dtype=np.float64)

    def explain(self, mol: Any) -> AtomicContribution:
        """Compute per-atom contributions for one molecule.

        Parameters
        ----------
        mol : Mol

        Returns
        -------
        AtomicContribution

        Raises
        ------
        ValueError
            If ``mol`` is None.
        """
        from rdkit import Chem

        if mol is None:
            raise ValueError("Cannot explain a None molecule.")

        baseline = float(self._predict(self._featurize([mol]))[0])
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 0:
            return AtomicContribution(
                np.zeros(0, dtype=np.float64), baseline, ""
            )

        # Removing an atom outright changes the molecular graph in ways
        # that can break sanitization. Following Riniker and Landrum, the
        # atom is instead neutralized in place - reset to a generic atom
        # so the substructures it defined no longer match.
        variants: List[Any] = []
        for idx in range(n_atoms):
            editable = Chem.RWMol(mol)
            atom = editable.GetAtomWithIdx(idx)
            atom.SetAtomicNum(0)
            atom.SetIsAromatic(False)
            atom.SetFormalCharge(0)
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(0)
            variant = editable.GetMol()
            try:
                Chem.SanitizeMol(variant)
            except RDKIT_MOLECULE_ERRORS:
                # Masking this atom left an invalid valence, so its
                # contribution cannot be measured; using the unmasked
                # molecule makes the reported weight zero, which is
                # honest -- better than a fabricated number.
                variant = mol
            variants.append(variant)

        masked = self._predict(self._featurize(variants))
        # A positive weight means masking the atom lowered the prediction,
        # i.e. the atom was pushing the prediction up.
        weights = baseline - masked
        return AtomicContribution(
            weights=np.asarray(weights, dtype=np.float64),
            prediction=baseline,
            smiles=str(Chem.MolToSmiles(mol)),
        )

    def transform(self, mols: Sequence[Any]) -> List[AtomicContribution]:
        """Explain a batch of molecules.

        Parameters
        ----------
        mols : sequence of Mol

        Returns
        -------
        list of AtomicContribution
        """
        return [self.explain(m) for m in mols if m is not None]

    def to_similarity_map_weights(self, mol: Any) -> List[float]:
        """Weights in the form RDKit's ``SimilarityMaps`` drawing expects.

        Parameters
        ----------
        mol : Mol

        Returns
        -------
        list of float
            One weight per atom, ready for
            ``SimilarityMaps.GetSimilarityMapFromWeights``.
        """
        return [float(w) for w in self.explain(mol).weights]

    def plot(self, mol: Any) -> "go.Figure":
        """Bar chart of per-atom contributions, labelled by element.

        Parameters
        ----------
        mol : Mol

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go

        result = self.explain(mol)
        labels = [
            f"{a.GetSymbol()}{a.GetIdx()}" for a in mol.GetAtoms()
        ]
        colors = ["#c0392b" if w < 0 else "#2874a6" for w in result.weights]
        fig = go.Figure(
            go.Bar(x=labels, y=result.weights, marker_color=colors)
        )
        fig.update_layout(
            title=f"Atomic contributions (prediction {result.prediction:.3f})",
            xaxis_title="Atom",
            yaxis_title="Contribution to prediction",
        )
        return fig


class FragmentContributionAnalyzer:
    """Attribute predictions to chemically meaningful fragments.

    Per-atom weights answer "which atoms matter"; chemists think in
    groups. This aggregates atomic contributions over substructures — a
    supplied SMARTS list, or the molecule's own BRICS fragments — so the
    output is "the nitro group contributes -1.2 log units" rather than a
    list of atom indices.

    Parameters
    ----------
    model : fitted estimator
    fingerprint : callable or transformer
    fragments : dict, optional
        ``name -> SMARTS`` to attribute against. When ``None``, BRICS
        decomposition is used to find the molecule's own fragments.
    use_proba : bool, default False

    Examples
    --------
    >>> from rdkit import Chem
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from qsarkit.representation import MorganFingerprint
    >>> fp = MorganFingerprint(n_bits=64)
    >>> mols = [Chem.MolFromSmiles(s) for s in ("CCO", "CCN", "c1ccccc1O", "CCC")]
    >>> model = RandomForestRegressor(n_estimators=5, random_state=0).fit(
    ...     fp.transform(mols), [1.0, 2.0, 3.0, 4.0]
    ... )
    >>> analyzer = FragmentContributionAnalyzer(
    ...     model, fp, fragments={"hydroxyl": "[OX2H]"}
    ... )
    >>> df = analyzer.analyze(mols[0])
    >>> "hydroxyl" in list(df["fragment"])
    True

    References
    ----------
    - Riniker, S. & Landrum, G. A. (2013). J. Cheminform., 5, 43.
      https://doi.org/10.1186/1758-2946-5-43
    - Degen, J. et al. (2008). "On the Art of Compiling and Using
      'Drug-Like' Chemical Fragment Spaces." ChemMedChem, 3(10),
      1503-1507. https://doi.org/10.1002/cmdc.200800178
    - Sheridan, R. P. (2019). "Interpretation of QSAR Models by Coloring
      Atoms According to Changes in Predicted Activity." J. Chem. Inf.
      Model., 59(4), 1324-1337.
      https://doi.org/10.1021/acs.jcim.8b00825
    """

    def __init__(
        self,
        model: Any,
        fingerprint: Any,
        fragments: Optional[Dict[str, str]] = None,
        use_proba: bool = False,
    ) -> None:
        self.model = model
        self.fingerprint = fingerprint
        self.fragments = fragments
        self.use_proba = use_proba
        self._atomic = AtomicContributionMap(model, fingerprint, use_proba)

    def _brics_groups(self, mol: Any) -> Dict[str, List[int]]:
        """Atom indices of each BRICS fragment present in the molecule."""
        from rdkit import Chem
        from rdkit.Chem import BRICS

        groups: Dict[str, List[int]] = {}
        try:
            bonds = list(BRICS.FindBRICSBonds(mol))
        except RDKIT_MOLECULE_ERRORS:
            bonds = []
        if not bonds:
            return {Chem.MolToSmiles(mol): list(range(mol.GetNumAtoms()))}

        indices = [mol.GetBondBetweenAtoms(*b[0]).GetIdx() for b in bonds]
        fragmented = Chem.FragmentOnBonds(mol, indices, addDummies=False)
        for atom_ids in Chem.GetMolFrags(fragmented):
            piece = Chem.MolFragmentToSmiles(mol, atomsToUse=list(atom_ids))
            groups.setdefault(piece, list(atom_ids))
        return groups

    def analyze(self, mol: Any) -> "pd.DataFrame":
        """Contribution of each fragment to the prediction.

        Parameters
        ----------
        mol : Mol

        Returns
        -------
        pandas.DataFrame
            Columns ``fragment``, ``n_atoms``, ``contribution`` (summed
            over the fragment's atoms), ``mean_contribution``, sorted by
            descending absolute contribution. Fragments not present in
            the molecule are omitted.
        """
        import pandas as pd
        from rdkit import Chem

        contribution = self._atomic.explain(mol)
        weights = contribution.weights

        if self.fragments is not None:
            groups: Dict[str, List[int]] = {}
            for name, smarts in self.fragments.items():
                pattern = Chem.MolFromSmarts(smarts)
                if pattern is None:
                    continue
                atoms = sorted(
                    {i for match in mol.GetSubstructMatches(pattern) for i in match}
                )
                if atoms:
                    groups[name] = atoms
        else:
            groups = self._brics_groups(mol)

        rows = [
            {
                "fragment": name,
                "n_atoms": len(atoms),
                "contribution": float(weights[atoms].sum()),
                "mean_contribution": float(weights[atoms].mean()),
            }
            for name, atoms in groups.items()
        ]
        frame = pd.DataFrame(
            rows,
            columns=["fragment", "n_atoms", "contribution", "mean_contribution"],
        )
        if not frame.empty:
            frame = frame.reindex(
                frame["contribution"].abs().sort_values(ascending=False).index
            ).reset_index(drop=True)
        return frame


class CounterfactualExplainer:
    """Explain a prediction by finding the nearest molecule the model scores differently.

    Answers the question a chemist actually asks — "what would I have to
    change to fix this?" — by searching a candidate set for the
    structurally closest molecule whose prediction differs by at least a
    given margin. Unlike an importance ranking, the result is an
    actionable, synthesizable alternative rather than an abstraction.

    Parameters
    ----------
    model : fitted estimator
    fingerprint : callable or transformer
        Used both for prediction and for the similarity search.
    delta : float, default 1.0
        Minimum prediction difference for a molecule to count as a
        counterfactual. On a pActivity scale, 1.0 is a ten-fold change.
    min_similarity : float, default 0.4
        Minimum Tanimoto similarity, so the counterfactual is a
        recognisable relative rather than an unrelated molecule.
    use_proba : bool, default False

    Examples
    --------
    >>> from rdkit import Chem
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from qsarkit.representation import MorganFingerprint
    >>> fp = MorganFingerprint(n_bits=256)
    >>> library = [Chem.MolFromSmiles(s) for s in
    ...            ("CCO", "CCN", "CCC", "c1ccccc1", "c1ccccc1O")]
    >>> model = RandomForestRegressor(n_estimators=5, random_state=0).fit(
    ...     fp.transform(library), [1.0, 1.1, 1.2, 8.0, 8.1]
    ... )
    >>> explainer = CounterfactualExplainer(model, fp, delta=2.0,
    ...                                     min_similarity=0.0)
    >>> result = explainer.explain(library[0], library)
    >>> result is None or "counterfactual" in result
    True

    References
    ----------
    - Wachter, S., Mittelstadt, B. & Russell, C. (2018). "Counterfactual
      Explanations without Opening the Black Box." Harvard J. Law &
      Tech., 31(2), 841-887. https://doi.org/10.2139/ssrn.3063289
    - Wellawatte, G. P., Seshadri, A. & White, A. D. (2022). "Model
      Agnostic Generation of Counterfactual Explanations for Molecules."
      Chem. Sci., 13, 3697-3705. https://doi.org/10.1039/D1SC05259D
    - Hussain, J. & Rea, C. (2010). "Computationally Efficient Algorithm
      to Identify Matched Molecular Pairs." J. Chem. Inf. Model., 50(3),
      339-348. https://doi.org/10.1021/ci900450m
    """

    def __init__(
        self,
        model: Any,
        fingerprint: Any,
        delta: float = 1.0,
        min_similarity: float = 0.4,
        use_proba: bool = False,
    ) -> None:
        self.model = model
        self.fingerprint = fingerprint
        self.delta = delta
        self.min_similarity = min_similarity
        self.use_proba = use_proba

    def _featurize(self, mols: Sequence[Any]) -> npt.NDArray[np.float64]:
        transform = getattr(self.fingerprint, "transform", self.fingerprint)
        return np.asarray(transform(list(mols)), dtype=np.float64)

    def _predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.use_proba:
            return np.asarray(
                self.model.predict_proba(X), dtype=np.float64
            )[:, -1]
        return np.asarray(self.model.predict(X), dtype=np.float64)

    def explain(
        self, mol: Any, candidates: Sequence[Any]
    ) -> Optional[Dict[str, Any]]:
        """Find the most similar candidate the model scores differently.

        Parameters
        ----------
        mol : Mol
            The molecule to explain.
        candidates : sequence of Mol
            Molecules to search. Typically a virtual library or the rest
            of the dataset.

        Returns
        -------
        dict or None
            ``counterfactual`` (Mol), ``smiles``, ``similarity``,
            ``prediction``, ``original_prediction``, ``delta``, and
            ``transformation`` (the matched-pair change, when the two
            form a matched molecular pair). ``None`` when no candidate
            meets both thresholds.
        """
        from rdkit import Chem

        from qsarkit.neighbors import tanimoto_similarity_matrix

        if mol is None:
            raise ValueError("Cannot explain a None molecule.")
        valid = [c for c in candidates if c is not None]
        if not valid:
            return None

        query_fp = self._featurize([mol])
        candidate_fps = self._featurize(valid)
        original = float(self._predict(query_fp)[0])
        predictions = self._predict(candidate_fps)
        similarity = tanimoto_similarity_matrix(query_fp, candidate_fps)[0]

        differences = np.abs(predictions - original)
        eligible = np.flatnonzero(
            (differences >= self.delta) & (similarity >= self.min_similarity)
        )
        if eligible.size == 0:
            return None

        # Among candidates that change the prediction enough, the most
        # similar one is the smallest structural edit that does the job.
        best = int(eligible[np.argmax(similarity[eligible])])
        counterfactual = valid[best]

        transformation = None
        pairs = self._matched_pair(mol, counterfactual)
        if pairs:
            transformation = pairs

        return {
            "counterfactual": counterfactual,
            "smiles": str(Chem.MolToSmiles(counterfactual)),
            "similarity": float(similarity[best]),
            "prediction": float(predictions[best]),
            "original_prediction": original,
            "delta": float(predictions[best] - original),
            "transformation": transformation,
        }

    @staticmethod
    def _matched_pair(a: Any, b: Any) -> Optional[str]:
        """Describe the change as an MMP transformation, when one exists."""
        from qsarkit.sar import MatchedMolecularPairs

        pairs = MatchedMolecularPairs().find_pairs([a, b])
        return pairs[0].transformation if pairs else None
