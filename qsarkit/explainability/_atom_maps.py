r"""Project per-feature attributions back onto atoms, and draw them.

SHAP and LIME attribute a prediction to *features*. For a fingerprint
model those features are hash buckets, and "bit 1743 contributed +0.21" is
not an explanation a chemist can act on. What they need is the same
information on the structure.

The bridge is the fingerprint's bit-provenance map: RDKit can report,
for each bit a molecule sets, which atom environments set it. Distributing
each bit's attribution over the atoms in its environments turns a vector of
feature attributions into a vector of atom weights, which RDKit then draws
as a similarity map.

Two caveats that the arithmetic cannot remove:

* **Bit collisions.** A hashed fingerprint folds many environments into
  each bit, so a bit's attribution may belong to an environment other than
  the one present in this molecule. Larger fingerprints collide less;
  :meth:`AttributionAtomMapper.collision_rate` reports how much of a given
  molecule's signal is affected.
* **Shared credit.** An atom appearing in several environments accumulates
  weight from each. That is usually what you want -- an atom central to
  many substructures really is more implicated -- but it means weights are
  not a partition of the prediction and do not sum to it.

References
----------
- Riniker, S. & Landrum, G. A. (2013). "Similarity Maps -- A Visualization
  Strategy for Molecular Fingerprints and Machine-Learning Methods."
  J. Cheminform., 5, 43. https://doi.org/10.1186/1758-2946-5-43
- Lundberg, S. M. & Lee, S.-I. (2017). "A Unified Approach to Interpreting
  Model Predictions." NeurIPS 2017, 4765-4774.
  https://papers.nips.cc/paper/7062
- Ribeiro, M. T., Singh, S. & Guestrin, C. (2016). "Why Should I Trust
  You?: Explaining the Predictions of Any Classifier." KDD 2016,
  1135-1144. https://doi.org/10.1145/2939672.2939778
- Rogers, D. & Hahn, M. (2010). "Extended-Connectivity Fingerprints."
  J. Chem. Inf. Model., 50(5), 742-754. https://doi.org/10.1021/ci100050t
- Polishchuk, P. (2017). "Interpretation of Quantitative
  Structure-Activity Relationship Models: Past, Present, and Future."
  J. Chem. Inf. Model., 57(11), 2618-2639.
  https://doi.org/10.1021/acs.jcim.7b00274
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover
    import plotly.graph_objects as go
    from rdkit.Chem import Mol

__all__ = [
    "bit_atom_environments",
    "bit_weights_to_atom_weights",
    "AttributionAtomMapper",
    "draw_atom_weights",
]

_Distribution = Literal["uniform", "center", "radius_weighted"]


def bit_atom_environments(
    mol: Any, radius: int = 2, n_bits: int = 2048, use_features: bool = False
) -> Dict[int, Tuple[Tuple[int, int], ...]]:
    """Which atom environments set each bit of a molecule's Morgan fingerprint.

    Parameters
    ----------
    mol : Mol
        The molecule.
    radius : int, default 2
        Morgan radius. Must match the fingerprint the attributions came
        from, or the bit numbering will not correspond.
    n_bits : int, default 2048
        Fingerprint length. Must likewise match.
    use_features : bool, default False
        Use the feature-based (FCFP) invariants rather than connectivity.

    Returns
    -------
    dict
        Bit index -> tuple of ``(central_atom_index, radius)`` pairs, one
        per environment in this molecule that sets that bit.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.explainability import bit_atom_environments
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> environments = bit_atom_environments(mol, n_bits=256)
    >>> len(environments) > 0
    True
    >>> all(isinstance(bit, int) for bit in environments)
    True

    Every reported central atom is a real atom of the molecule:

    >>> centres = {atom for envs in environments.values() for atom, _ in envs}
    >>> max(centres) < mol.GetNumAtoms()
    True

    References
    ----------
    - RDKit, "Explaining bits from Morgan fingerprints":
      https://www.rdkit.org/docs/GettingStartedInPython.html#explaining-bits-from-morgan-fingerprints
    """
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        atomInvariantsGenerator=(
            rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
            if use_features
            else None
        ),
    )
    output = rdFingerprintGenerator.AdditionalOutput()
    output.AllocateBitInfoMap()
    generator.GetFingerprint(mol, additionalOutput=output)
    return {int(bit): tuple(envs) for bit, envs in output.GetBitInfoMap().items()}


def _environment_atoms(mol: Any, centre: int, radius: int) -> List[int]:
    """Atoms inside the circular environment of ``centre`` at ``radius``."""
    from rdkit import Chem

    if radius == 0:
        return [centre]
    bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, centre)
    if not bonds:
        return [centre]
    atoms = {centre}
    for bond_index in bonds:
        bond = mol.GetBondWithIdx(bond_index)
        atoms.add(bond.GetBeginAtomIdx())
        atoms.add(bond.GetEndAtomIdx())
    return sorted(atoms)


def bit_weights_to_atom_weights(
    mol: Any,
    bit_weights: npt.ArrayLike,
    radius: int = 2,
    n_bits: int = 2048,
    use_features: bool = False,
    distribution: _Distribution = "uniform",
) -> "npt.NDArray[np.float64]":
    """Spread per-bit attributions over the atoms that produced each bit.

    Parameters
    ----------
    mol : Mol
        The molecule the attributions were computed for.
    bit_weights : array-like of shape (n_bits,)
        One attribution per fingerprint bit -- a row of SHAP values, a LIME
        coefficient vector, or any per-feature importance.
    radius : int, default 2
        Morgan radius used to compute ``bit_weights``.
    n_bits : int, default 2048
        Fingerprint length used to compute ``bit_weights``.
    use_features : bool, default False
        Whether those were FCFP rather than ECFP bits.
    distribution : {"uniform", "center", "radius_weighted"}, default "uniform"
        How a bit's weight is divided among its environment's atoms:

        ``"uniform"``
            Split equally. Neutral, and the right default.
        ``"center"``
            All of it to the central atom. Sharper pictures, but it
            overstates the centre of a large environment.
        ``"radius_weighted"``
            Split equally, then divide by ``radius + 1``, so a bit
            describing a tight environment carries more weight per atom
            than one describing a diffuse environment.

    Returns
    -------
    ndarray of shape (n_atoms,)
        Per-atom weight, ready for :func:`draw_atom_weights`.

    Raises
    ------
    ValueError
        If ``bit_weights`` is not one-dimensional of length ``n_bits``.

    Examples
    --------
    >>> import numpy as np
    >>> from rdkit import Chem
    >>> from qsarkit.explainability import (
    ...     bit_atom_environments, bit_weights_to_atom_weights)
    >>> mol = Chem.MolFromSmiles("CC(=O)Nc1ccc(Cl)cc1")
    >>> weights = np.zeros(256)
    >>> environments = bit_atom_environments(mol, n_bits=256)
    >>> chlorine = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Cl"][0]
    >>> # Attribute to one bit centred on the chlorine, radius 0.
    >>> for bit, envs in environments.items():
    ...     if (chlorine, 0) in envs:
    ...         weights[bit] = 1.0
    >>> atom_weights = bit_weights_to_atom_weights(mol, weights, n_bits=256)
    >>> int(np.argmax(atom_weights)) == chlorine
    True

    Only the atoms in that environment receive weight:

    >>> int((atom_weights != 0).sum())
    1

    References
    ----------
    - Riniker, S. & Landrum, G. A. (2013). J. Cheminform., 5, 43.
      https://doi.org/10.1186/1758-2946-5-43
    """
    values = np.asarray(bit_weights, dtype=np.float64).ravel()
    if values.size != n_bits:
        raise ValueError(
            f"bit_weights has {values.size} entries but n_bits is {n_bits}. "
            "These must match, or bit indices will not line up with the "
            "attributions -- check that the fingerprint used for the "
            "explanation is the one named here."
        )
    if distribution not in ("uniform", "center", "radius_weighted"):
        raise ValueError(
            f"distribution must be 'uniform', 'center' or 'radius_weighted', "
            f"got {distribution!r}."
        )

    environments = bit_atom_environments(
        mol, radius=radius, n_bits=n_bits, use_features=use_features
    )
    return _atom_weights_from_environments(
        mol, values, environments, distribution=distribution
    )


def _atom_weights_from_environments(
    mol: Any,
    values: "npt.NDArray[np.float64]",
    environments: Dict[int, Tuple[Tuple[int, int], ...]],
    distribution: _Distribution = "uniform",
) -> "npt.NDArray[np.float64]":
    """Spread per-bit weights over the atoms of the environments that set them.

    Split out so the bit provenance can come either from a standalone
    recomputation (:func:`bit_atom_environments`) or from the fingerprint
    object itself, without the distribution logic existing twice.
    """
    if distribution not in ("uniform", "center", "radius_weighted"):
        raise ValueError(
            f"distribution must be 'uniform', 'center' or 'radius_weighted', "
            f"got {distribution!r}."
        )

    atom_weights = np.zeros(mol.GetNumAtoms(), dtype=np.float64)
    for bit, envs in environments.items():
        weight = values[bit]
        if weight == 0.0:
            continue
        for centre, env_radius in envs:
            if distribution == "center":
                atom_weights[centre] += weight
                continue
            atoms = _environment_atoms(mol, centre, env_radius)
            share = weight / len(atoms)
            if distribution == "radius_weighted":
                share /= env_radius + 1
            for atom in atoms:
                atom_weights[atom] += share
    return atom_weights


def _bit_index(feature: Any) -> int:
    """Read a bit index out of whatever LIME called the feature.

    LIME labels features ``"x0"``, ``"x1"``, ... when it is given no names,
    which is how :class:`~qsarkit.explainability.LIMEExplainer` runs it. A
    caller who supplied their own ``feature_names`` gets those back instead,
    and names that are not bit indices cannot be placed on atoms.
    """
    if isinstance(feature, (int, np.integer)):
        return int(feature)
    text = str(feature).strip()
    if text.startswith("x") and text[1:].isdigit():
        return int(text[1:])
    if text.isdigit():
        return int(text)
    raise ValueError(
        f"cannot read a fingerprint bit index from the feature label "
        f"{feature!r}. Atom mapping needs the explanation to refer to bits by "
        "index; leave `feature_names` unset on LIMEExplainer so the labels "
        "stay in the form 'x123'."
    )


def _lime_contributions(explanation: Any) -> List[Tuple[Any, float]]:
    """Normalise the shapes a LIME explanation arrives in to (feature, weight)."""
    # The frame LIMEExplainer.explain_one returns.
    columns = getattr(explanation, "columns", None)
    if columns is not None and "feature" in columns and "weight" in columns:
        return [
            (row.feature, float(row.weight))
            for row in explanation.itertuples(index=False)
        ]
    # A LIME Explanation object.
    if hasattr(explanation, "as_map"):
        return [
            (feature, float(weight))
            for pairs in explanation.as_map().values()
            for feature, weight in pairs
        ]
    if isinstance(explanation, Mapping):
        return [(feature, float(weight)) for feature, weight in explanation.items()]
    try:
        return [(feature, float(weight)) for feature, weight in explanation]
    except (TypeError, ValueError):
        raise TypeError(
            f"cannot read a LIME explanation from {type(explanation).__name__}. "
            "Pass the frame from LIMEExplainer.explain_one, a LIME Explanation, "
            "a {bit: weight} mapping, or (bit, weight) pairs."
        ) from None


class AttributionAtomMapper:
    """Turn a SHAP or LIME explainer's output into atom-level weights.

    Wraps any per-feature attribution and the fingerprint that produced
    those features, so a model explanation can be shown on the structure
    rather than as bit indices.

    Parameters
    ----------
    fingerprint : object
        The transformer whose bits the attributions refer to. Its
        ``radius``, ``n_bits`` and (where present) ``use_features``
        attributes are read so the bit numbering matches -- typically a
        :class:`~qsarkit.representation.MorganFingerprint`.
    distribution : {"uniform", "center", "radius_weighted"}, default "uniform"
        Passed to :func:`bit_weights_to_atom_weights`.

    Attributes
    ----------
    radius : int
    n_bits : int
    use_features : bool

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.explainability import AttributionAtomMapper
    >>> from qsarkit.models import QSARRegressor
    >>> from qsarkit.representation import MorganFingerprint
    >>> from rdkit import Chem
    >>> mols = [Chem.MolFromSmiles(s) for s in
    ...         ("CC(=O)Nc1ccccc1", "CC(=O)Nc1ccc(Cl)cc1", "CCO", "CCN")]
    >>> fingerprint = MorganFingerprint(radius=2, n_bits=256)
    >>> X = fingerprint.transform(mols)
    >>> model = QSARRegressor("rf", random_state=0).fit(X, [6.2, 6.7, 5.0, 5.1])
    >>> mapper = AttributionAtomMapper(fingerprint)

    Any per-bit vector maps onto atoms. Here we use permutation
    importance, which works for every backend and needs no optional
    dependency:

    >>> from qsarkit.explainability import PermutationImportance
    >>> importance = PermutationImportance(n_repeats=2, random_state=0).fit(
    ...     model, X, [6.2, 6.7, 5.0, 5.1])
    >>> weights = mapper.atom_weights(mols[1], importance.importances_mean_)
    >>> weights.shape == (mols[1].GetNumAtoms(),)
    True

    Symmetry-equivalent atoms receive identical weight, which is a useful
    correctness check on the mapping:

    >>> chlorobenzene = Chem.MolFromSmiles("Clc1ccccc1")
    >>> import numpy as np
    >>> bits = np.ones(256)
    >>> symmetric = mapper.atom_weights(chlorobenzene, bits)
    >>> bool(np.isclose(symmetric[2], symmetric[6]))
    True

    The collision rate says how much to trust the picture:

    >>> rate = mapper.collision_rate(mols[1])
    >>> 0.0 <= rate <= 1.0
    True

    References
    ----------
    - Riniker, S. & Landrum, G. A. (2013). J. Cheminform., 5, 43.
      https://doi.org/10.1186/1758-2946-5-43
    - Polishchuk, P. (2017). J. Chem. Inf. Model., 57(11), 2618-2639.
      https://doi.org/10.1021/acs.jcim.7b00274
    """

    def __init__(
        self, fingerprint: Any, distribution: _Distribution = "uniform"
    ) -> None:
        if not hasattr(fingerprint, "bit_info"):
            raise TypeError(
                f"{type(fingerprint).__name__} cannot report which atoms set "
                "each bit, so per-bit attributions cannot be placed on atoms. "
                "Atom mapping needs a fingerprint whose bits correspond to "
                "local atom environments and that exposes bit_info(mol) -- "
                "MorganFingerprint and FeatureMorganFingerprint do. A MACCS "
                "or path-based fingerprint has no such correspondence."
            )
        self.fingerprint = fingerprint
        self.distribution = distribution
        self.n_bits = int(getattr(fingerprint, "n_bits", 2048))
        # Kept for `repr` and for callers that read them; the mapping itself
        # no longer derives from these, because re-deriving the bit numbering
        # from loose parameters disagrees with the transformer whenever a
        # setting such as `use_chirality` changes the hashing.
        self.radius = int(getattr(fingerprint, "radius", 2))
        self.use_features = bool(getattr(fingerprint, "use_features", False))

    def atom_weights(
        self, mol: Any, bit_weights: npt.ArrayLike
    ) -> "npt.NDArray[np.float64]":
        """Map one molecule's per-bit attributions onto its atoms.

        The bit provenance comes from the fingerprint object itself, so the
        bit numbering is the one the model was fitted on whatever the
        fingerprint's settings.

        Parameters
        ----------
        mol : Mol
        bit_weights : array-like of shape (n_bits,)

        Returns
        -------
        ndarray of shape (n_atoms,)

        Raises
        ------
        ValueError
            If ``bit_weights`` is not one-dimensional of length ``n_bits``.
        """
        weights = np.asarray(bit_weights, dtype=np.float64).ravel()
        if weights.size != self.n_bits:
            raise ValueError(
                f"bit_weights has {weights.size} entries but the fingerprint "
                f"has {self.n_bits} bits. They must describe the same feature "
                "matrix."
            )
        return _atom_weights_from_environments(
            mol,
            weights,
            self.fingerprint.bit_info(mol),
            distribution=self.distribution,
        )

    def from_shap(
        self, mol: Any, explainer: Any, X: npt.ArrayLike, index: int = 0
    ) -> "npt.NDArray[np.float64]":
        """Atom weights from a :class:`~qsarkit.explainability.SHAPExplainer`.

        Parameters
        ----------
        mol : Mol
            The molecule corresponding to row ``index`` of ``X``.
        explainer : SHAPExplainer
            A fitted explainer.
        X : array-like of shape (n_samples, n_bits)
            The feature matrix the explanation is computed on.
        index : int, default 0
            Which row of ``X`` to explain.

        Returns
        -------
        ndarray of shape (n_atoms,)

        Raises
        ------
        OptionalDependencyError
            If ``shap`` is not installed.
        """
        values = explainer.shap_values(X)
        # Reduce the class axis before picking the sample, so that a raw shap
        # explainer -- which returns one array per class -- does not have its
        # classes read as samples. SHAPExplainer already reduces both forms.
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 3:
            # (n_samples, n_features, n_classes); attribute to the positive
            # class, the one being asked about.
            values = values[:, :, -1]
        return self.atom_weights(mol, values[index])

    def from_lime(
        self, mol: Any, explanation: Any
    ) -> "npt.NDArray[np.float64]":
        """Atom weights from a LIME explanation of one molecule.

        Parameters
        ----------
        mol : Mol
        explanation : DataFrame, Explanation, mapping or pairs
            Any of the forms LIME explanations arrive in:

            * the ``feature``/``weight`` frame returned by
              :meth:`~qsarkit.explainability.LIMEExplainer.explain_one`,
              whose feature labels are of the form ``"x123"``;
            * a LIME ``Explanation`` object, whose ``as_map()`` is read;
            * a mapping of bit index to weight;
            * a sequence of ``(bit index, weight)`` pairs.

        Returns
        -------
        ndarray of shape (n_atoms,)

        Raises
        ------
        ValueError
            If a feature label cannot be read as a bit index, or an index
            falls outside the fingerprint.

        Examples
        --------
        >>> import numpy as np
        >>> from rdkit import Chem
        >>> from qsarkit.explainability import AttributionAtomMapper
        >>> from qsarkit.representation import MorganFingerprint
        >>> mol = Chem.MolFromSmiles("CC(=O)Nc1ccc(Cl)cc1")
        >>> fingerprint = MorganFingerprint(radius=2, n_bits=256)
        >>> mapper = AttributionAtomMapper(fingerprint)
        >>> some_bit = next(iter(fingerprint.bit_info(mol)))
        >>> weights = mapper.from_lime(mol, {some_bit: 1.0})
        >>> weights.shape == (mol.GetNumAtoms(),)
        True
        >>> bool(weights.sum() > 0)
        True
        """
        weights = np.zeros(self.n_bits, dtype=np.float64)
        for feature, weight in _lime_contributions(explanation):
            index = _bit_index(feature)
            if not 0 <= index < self.n_bits:
                raise ValueError(
                    f"LIME reported feature {feature!r} (bit {index}), which "
                    f"is outside the {self.n_bits}-bit fingerprint this "
                    "mapper was built with. The explanation and the mapper "
                    "must describe the same feature matrix."
                )
            weights[index] += float(weight)
        return self.atom_weights(mol, weights)

    def collision_rate(self, mol: Any) -> float:
        """Fraction of this molecule's set bits shared by several environments.

        A bit set by more than one environment carries attribution that
        cannot be assigned to a single substructure, so a high rate means
        the atom-level picture is blurred. Widening the fingerprint lowers
        it.

        Parameters
        ----------
        mol : Mol

        Returns
        -------
        float
            In [0, 1]. ``0.0`` when the molecule sets no bits.
        """
        environments = self.fingerprint.bit_info(mol)
        if not environments:
            return 0.0
        collided = sum(1 for envs in environments.values() if len(envs) > 1)
        return collided / len(environments)

    def __repr__(self) -> str:
        return (
            f"<AttributionAtomMapper radius={self.radius} "
            f"n_bits={self.n_bits} distribution={self.distribution!r}>"
        )


def draw_atom_weights(
    mol: Any,
    atom_weights: npt.ArrayLike,
    size: Tuple[int, int] = (400, 400),
    fmt: Literal["svg", "png"] = "svg",
    normalize: bool = True,
    contour_lines: int = 10,
) -> Any:
    """Render atom weights on the structure as an RDKit similarity map.

    The standard cheminformatics depiction: a green-to-pink field over the
    2D structure, positive contributions in one colour and negative in the
    other.

    Parameters
    ----------
    mol : Mol
        The molecule. A 2D conformer is computed if it has none.
    atom_weights : array-like of shape (n_atoms,)
        Per-atom weight, e.g. from :class:`AttributionAtomMapper`.
    size : tuple of int, default (400, 400)
        Image size in pixels.
    fmt : {"svg", "png"}, default "svg"
        ``"svg"`` returns a string, which renders inline in a notebook and
        embeds in HTML. ``"png"`` returns bytes.
    normalize : bool, default True
        Scale the weights so the largest absolute value maps to the end of
        the colour scale. Keeps the picture readable regardless of the
        attribution's units; turn it off to compare two molecules on one
        absolute scale.
    contour_lines : int, default 10
        Number of contour lines drawn.

    Returns
    -------
    str or bytes
        SVG text, or PNG bytes.

    Raises
    ------
    ValueError
        If ``atom_weights`` does not have one entry per atom, or ``fmt`` is
        not recognized.

    Examples
    --------
    >>> import numpy as np
    >>> from rdkit import Chem
    >>> from qsarkit.explainability import draw_atom_weights
    >>> mol = Chem.MolFromSmiles("CC(=O)Nc1ccc(Cl)cc1")
    >>> weights = np.linspace(-1, 1, mol.GetNumAtoms())
    >>> svg = draw_atom_weights(mol, weights)
    >>> svg.lstrip().startswith("<?xml") or svg.lstrip().startswith("<svg")
    True
    >>> isinstance(draw_atom_weights(mol, weights, fmt="png"), bytes)
    True

    A weight per atom is required, so a mismatched vector is caught rather
    than silently misaligned:

    >>> draw_atom_weights(mol, [0.1, 0.2])
    Traceback (most recent call last):
        ...
    ValueError: atom_weights has 2 entries but the molecule has 11 atoms.

    References
    ----------
    - Riniker, S. & Landrum, G. A. (2013). "Similarity Maps -- A
      Visualization Strategy for Molecular Fingerprints and
      Machine-Learning Methods." J. Cheminform., 5, 43.
      https://doi.org/10.1186/1758-2946-5-43
    - RDKit ``Chem.Draw.SimilarityMaps`` documentation:
      https://www.rdkit.org/docs/source/rdkit.Chem.Draw.SimilarityMaps.html
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    from rdkit.Chem.Draw import SimilarityMaps

    weights = np.asarray(atom_weights, dtype=np.float64).ravel()
    if weights.size != mol.GetNumAtoms():
        raise ValueError(
            f"atom_weights has {weights.size} entries but the molecule has "
            f"{mol.GetNumAtoms()} atoms."
        )
    if fmt not in ("svg", "png"):
        raise ValueError(f"fmt must be 'svg' or 'png', got {fmt!r}.")

    if normalize:
        largest = float(np.abs(weights).max())
        if largest > 0:
            weights = weights / largest

    # The drawing needs 2D coordinates; a molecule parsed from SMILES has none.
    drawn = Chem.Mol(mol)
    if drawn.GetNumConformers() == 0:
        AllChem.Compute2DCoords(drawn)

    drawer = (
        Draw.MolDraw2DSVG(*size) if fmt == "svg" else Draw.MolDraw2DCairo(*size)
    )
    SimilarityMaps.GetSimilarityMapFromWeights(
        drawn,
        [float(w) for w in weights],
        draw2d=drawer,
        contourLines=contour_lines,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()
