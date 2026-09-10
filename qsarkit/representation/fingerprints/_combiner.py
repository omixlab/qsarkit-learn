"""Concatenation of several fingerprint transformers into one feature matrix."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from qsarkit.base import MoleculeTransformer

if TYPE_CHECKING:  # pragma: no cover
    from rdkit.Chem import Mol

__all__ = ["FingerprintCombiner"]


class FingerprintCombiner(MoleculeTransformer):
    """Horizontally concatenate several fingerprint/descriptor transformers.

    Many QSAR pipelines combine a circular fingerprint with a structural-key
    or physicochemical descriptor block (e.g. ECFP4 + MACCS, or ECFP4 +
    Lipinski descriptors) because the two capture complementary information.
    ``FingerprintCombiner`` fits each member transformer independently on
    the same molecules and concatenates their outputs column-wise, exposing
    the combination as a single scikit-learn transformer that fits and
    transforms in one call.

    Parameters
    ----------
    transformers : sequence of (str, MoleculeTransformer)
        Named member transformers. Each must implement ``fit``/``transform``
        (or ``fit_transform``) over ``Iterable[Mol]`` and return a 2-D
        ``numpy.ndarray`` with a consistent number of rows. Names are used
        to prefix ``get_feature_names_out()`` and must be unique.
    weights : sequence of float, optional
        Per-member multiplicative weight applied to that member's output
        block before concatenation. Defaults to 1.0 for every member.

    Attributes
    ----------
    n_features_out_ : int
        Total width of the concatenated output, set after ``fit``.

    Notes
    -----
    This mirrors the intent of ``sklearn.pipeline.FeatureUnion`` but is
    specialized to the ``Iterable[Mol]`` input contract used throughout
    qsarkit (``FeatureUnion`` itself works fine with these transformers too;
    this class exists for a lighter-weight, dependency-free alternative and
    for symmetry with :class:`qsarkit.transform.MoleculeFeatureUnion`, which
    wraps the same pattern for the ``qsarkit.transform`` namespace).

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.representation.fingerprints import (
    ...     FingerprintCombiner, MorganFingerprint, MACCSKeysFingerprint,
    ... )
    >>> combiner = FingerprintCombiner([
    ...     ("morgan", MorganFingerprint(n_bits=32)),
    ...     ("maccs", MACCSKeysFingerprint()),
    ... ])
    >>> X = combiner.fit_transform([Chem.MolFromSmiles("CCO")])
    >>> X.shape
    (1, 199)

    References
    ----------
    - Pedregosa et al. (2011). "Scikit-learn: Machine Learning in Python."
      J. Mach. Learn. Res., 12, 2825-2830.
      https://jmlr.org/papers/v12/pedregosa11a.html
    - scikit-learn ``FeatureUnion`` documentation:
      https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html
    - Nembri, S. et al. (2016). "In Silico Prediction of Cytochrome P450-Drug
      Interaction: QSARs for CYP3A4 and CYP2C9." Int. J. Mol. Sci., 17(6),
      914. https://doi.org/10.3390/ijms17060914 (example of combined
      fingerprint + descriptor QSAR feature sets).
    """

    def __init__(
        self,
        transformers: Sequence[Tuple[str, MoleculeTransformer]],
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        self.transformers = transformers
        self.weights = weights

    def _validate(self) -> None:
        if not self.transformers:
            raise ValueError("FingerprintCombiner needs at least one transformer.")
        names = [name for name, _ in self.transformers]
        if len(set(names)) != len(names):
            raise ValueError(f"Transformer names must be unique, got {names!r}.")
        if self.weights is not None and len(self.weights) != len(self.transformers):
            raise ValueError(
                f"weights has length {len(self.weights)}, expected "
                f"{len(self.transformers)} (one per transformer)."
            )

    def fit(
        self, mols: Any, y: Optional[Any] = None
    ) -> "FingerprintCombiner":
        """Fit every member transformer on the same molecules.

        Parameters
        ----------
        mols : Iterable[rdkit.Chem.Mol]
        y : array-like, optional
            Forwarded to each member's ``fit``.

        Returns
        -------
        FingerprintCombiner
            self.
        """
        from qsarkit.base import ensure_mol_list

        self._validate()
        mol_list = ensure_mol_list(mols)
        for _, transformer in self.transformers:
            transformer.fit(mol_list, y)
        self.n_features_out_ = sum(
            int(np.asarray(t.get_feature_names_out()).shape[0])
            for _, t in self.transformers
        )
        return self

    def _transform(self, mols: List[Optional["Mol"]]) -> npt.NDArray[np.float64]:
        self._validate()
        weights = self.weights if self.weights is not None else [1.0] * len(
            self.transformers
        )
        blocks = [
            np.asarray(transformer.transform(mols), dtype=np.float64) * weight
            for (_, transformer), weight in zip(self.transformers, weights)
        ]
        return np.asarray(np.concatenate(blocks, axis=1), dtype=np.float64)

    def get_feature_names_out(
        self, input_features: Optional[Sequence[str]] = None
    ) -> npt.NDArray[np.object_]:
        """Return prefixed feature names from every member transformer.

        Parameters
        ----------
        input_features : sequence of str, optional
            Ignored; present for scikit-learn API compatibility.

        Returns
        -------
        numpy.ndarray
            Array of ``str`` names, formatted ``"<member_name>__<feature>"``.
        """
        self._validate()
        names: List[str] = []
        for member_name, transformer in self.transformers:
            for feature_name in transformer.get_feature_names_out():
                names.append(f"{member_name}__{feature_name}")
        return np.asarray(names, dtype=object)
